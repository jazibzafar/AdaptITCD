##
import argparse
import json
from pyproj import CRS
import rasterio
import torch
import yaml
from shapely.ops import unary_union
from modules import ConvNeXtBackbone, BackboneWithFPN, ResNet50Backbone, TorchvisionSwinV2Backbone
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import shapely
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import os
import numpy as np
import cv2
from utils import model_replace_prefix
from torch.utils.data import Dataset, DataLoader
from tifffile import imread
import torchvision.transforms.functional as F
from tqdm.auto import tqdm
from torchvision.transforms import Normalize


OAM_NORM_MEAN = [0.452, 0.466, 0.364]
OAM_NORM_STDV = [0.235, 0.227, 0.220]


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def collate_fn(batch):
    return tuple(zip(*batch))


def load_pretrained_maskrcnn_model(model_path):
    """
    Load the pretrained mask rcnn checkpoints into a model used for prediction
    :param model_path: /path/to/model/directory/ OR /path/to/model/directory/checkpoint.ckpt
    :return: pretrained Mask RCNN model
    """
    # Ensuring there's not problems with the paths
    if not model_path.endswith(".ckpt"):
        ckpt_path = os.path.join(model_path, "last.ckpt")
        model_args_path = os.path.join(model_path, "args.yaml")
        model_root, model_name = os.path.split(model_path)
    else:
        ckpt_path = model_path
        _model_root, _ = os.path.split(model_path)
        model_args_path = os.path.join(_model_root, "args.yaml")
        model_root, model_name = os.path.split(_model_root)

    # get the arguments
    with open(model_args_path, 'r') as stream:
        model_args = yaml.safe_load(stream)
    model_args = DotDict(model_args)

    # initialize the backbone
    if model_args.arch_type == "resnet50":
        backbone = ResNet50Backbone(
            checkpoint_path=None,
            apply_lora=True if model_args.strategy == "lora" else False,
            lora_rank=model_args.lora_rank
        )
    elif model_args.arch_type == "convnext":
        backbone = ConvNeXtBackbone(
            backbone_type='small',
            checkpoint_path=None,
            apply_lora=True if model_args.strategy == "lora" else False,
            lora_rank=model_args.lora_rank
        )
    elif model_args.arch_type == "swin":
        backbone = TorchvisionSwinV2Backbone(
            checkpoint_path=None,
            apply_lora=True if model_args.strategy == "lora" else False,
            lora_rank=model_args.lora_rank
        )
    else:
        print(f"Incorrect backbone type. Check args.yaml in {model_path} for issues.")
        sys.exit(1)

    backbone_with_fpn = BackboneWithFPN(body=backbone)

    # create the maskrcnn model
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    model = MaskRCNN(
        backbone=backbone_with_fpn,
        num_classes=model_args.num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=model_args.img_size,
        max_size=model_args.img_size,
        box_detections_per_img=150
    )

    # load state dict from the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict_matched = model_replace_prefix(ckpt['state_dict'], "", "model.")
    msg = model.load_state_dict(ckpt_state_dict_matched, strict=False)
    print(msg)
    return model


class MinimalPredictTifDataset(Dataset):
    def __init__(self, data_path, oam_normalize=True):
        self.data_path = data_path
        self.predict_tile_list = os.listdir(data_path)
        self.predict_tile_list = [tile for tile in self.predict_tile_list if tile.endswith(".tif")]
        self.oam_normalize = oam_normalize
        self.normalize = Normalize(mean=OAM_NORM_MEAN, std=OAM_NORM_STDV)

    def __len__(self):
        return len(self.predict_tile_list)

    def __getitem__(self, idx):
        tile_name = self.predict_tile_list[idx]
        # tile = imread(os.path.join(self.data_path, tile_name))
        with rasterio.open(os.path.join(self.data_path, tile_name)) as src:
            transform = src.transform
            crs = src.crs
            tile = src.read()
        if tile.shape[0] == 3:
            tile = F.to_tensor(tile).permute(1, 0, 2)
        else:
            tile = F.to_tensor(tile)
        # tile = F.to_tensor(tile)
        if self.oam_normalize:
            tile = self.normalize(tile)
        metadata = {
            "tile_name": tile_name,
            "transform": transform,
            "crs": crs
        }
        return tile, metadata


def create_predict_loader(data_path, batch_size, num_workers, oam_normalization=False):
    """
    Create dataloader from the prediction data directory
    :param data_path: full/path/to/prediction/data/directory/
    :return: dataloader
    """
    dataset = MinimalPredictTifDataset(data_path, oam_normalization)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )


def mask_to_polygons(mask):
    """Convert a binary mask to polygon segmentation."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in contours:
        # Polygon needs at least 3 points
        if contour.shape[0] < 3:
            continue
        polygon = contour.reshape(-1, 2).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def collect_preds(tile_names, outputs, score_threshold, filter_tree_groups=True):
    """
    Filter low confidence predictions (based on bbox confidence) and create a dict for each tile.
    :param tile_names: list; names of all the tiles in a batch
    :param outputs: mask rcnn pred outputs
    :param score_threshold: min confidence score to keep the annotation
    :param filter_tree_groups: trigger it true to filter out tree group preds and keep only single tree preds
    :return: dict with key tile_name and values all annotations
    """
    assert len(tile_names) == len(outputs), "len(tile_names) /= len(outputs)"
    # infer batch size from the size of tile_names and outputs
    batch_size = len(outputs)
    pred_list = []
    for i in range(batch_size):
        current_tile = tile_names[i]  # is a string
        current_output = outputs[i]  # is a dict
        boxes = current_output["boxes"]
        labels = current_output["labels"]
        scores = current_output["scores"]
        masks = current_output["masks"]
        annotations = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # optional label filtering for tree groups:
            if filter_tree_groups:
                if label == 1:
                    continue
            # score filtering
            score = score.detach().cpu().item()
            if score < score_threshold:
                continue
            # getting box, label, segmentation, area in the right shape
            box = box.detach().cpu().numpy()
            label = label.detach().cpu().item()
            # mask shape = [1, H, W]
            mask = mask.detach().cpu().numpy()
            mask = mask[0]
            mask_binary = (mask >= 0.5).astype(np.uint8)
            segmentation = mask_to_polygons(mask_binary)
            if len(segmentation) == 0:
                continue
            area = float(mask_binary.sum())
            # append annotations list
            annotations.append({
                "id": len(annotations) + 1,
                "label": label,
                "box": box.tolist(),
                "segmentation": segmentation,
                "area": area,
                "score": score,
            })
        pred_list.append({
            "tile": current_tile,
            "num_detections": len(annotations),
            "annotations": annotations
        })
    return pred_list


def pixel_to_geo(polygon, transform):
    """
    Convert COCO-style polygon coordinates from pixel to map coordinates.
    """
    polygon = np.asarray(polygon).reshape(-1, 2)
    geo_coords = [list(transform * (float(x), float(y))) for x, y in polygon]
    return geo_coords


def collect_preds_geo(outputs, metadata, score_threshold=0.5, mask_threshold=0.5, filter_tree_groups=True):
    """
        Filter Mask R-CNN predictions and create georeferenced annotations.

        Parameters
        ----------
        meta : dict
            Geospatial metadata containing:
                - tile_names
                - transform (or tranform)
                - crs
            provided as lists containing one entry per tile.
        outputs : list
            Mask R-CNN prediction outputs.
        score_threshold : float
            Minimum confidence score to retain an annotation.
        filter_tree_groups : bool
            If True, filter out tree group predictions (label 1).
        mask_threshold : float
            Threshold for converting predicted mask probabilities to binary masks.
        Returns
        -------
        list
            Dictionary for each tile containing georeferenced annotations.
    """
    assert len(metadata) == len(outputs), "len(meta) /= len(outputs)"
    batch_size = len(outputs)

    pred_list = []
    for i, (current_meta, current_output) in enumerate(zip(metadata, outputs)):
        tile_name = current_meta["tile_name"]
        transform = current_meta["transform"]
        crs = CRS.from_user_input(current_meta["crs"])

        boxes = current_output["boxes"]
        labels = current_output["labels"]
        scores = current_output["scores"]
        masks = current_output["masks"]

        annotations = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # Convert tensors to Python/NumPy objects
            label = int(label.detach().cpu().item())
            score = float(score.detach().cpu().item())

            # Filter tree groups
            if filter_tree_groups and label == 1:
                continue

            # Filter low-confidence predictions
            if score < score_threshold:
                continue

            box = box.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()[0]  # Mask shape: [1, H, W]
            mask_binary = (mask >= mask_threshold).astype(np.uint8)
            # Extract polygons in pixel coordinates
            segmentation = mask_to_polygons(mask_binary)
            if len(segmentation) == 0:
                continue

            # Convert segmentation polygons to geographic coordinates
            segmentation_geo = [pixel_to_geo(poly, transform) for poly in segmentation]
            x1, y1, x2, y2 = box
            bbox_geo = [
                list(transform * (x1, y1)),
                list(transform * (x2, y1)),
                list(transform * (x2, y2)),
                list(transform * (x1, y2)),
            ]
            # Pixel area
            area = float(mask_binary.sum())
            annotations.append({
                "id": len(annotations) + 1,
                "label": label,
                "score": score,
                # Original pixel coordinates
                "box": box.tolist(),
                "segmentation": segmentation,
                # Geographic coordinates
                "bbox_geo": bbox_geo,
                "segmentation_geo": segmentation_geo,
                "area": area,
            })
        pred_list.append({
            "tile": tile_name,
            "crs": crs.to_string(),
            "transform": list(transform)[:6],
            "num_detections": len(annotations),
            "annotations": annotations,
        })
    return pred_list


def predict(pred_model, pred_loader, score_threshold=0.5, mask_threshold=0.5, filter_tree_groups=True, device="cuda"):
    """
    Predict the box and mask labels for the desired dataset.
    :param pred_model: prediction model loaded
    :param pred_loader: dataloader for prediction
    :param score_threshold: default=0.5; confidence filter to accept/reject model prediction. based on box score.
    :param mask_threshold: default=0.5; confidence filter for binarizing the mask.
    :param filter_tree_groups: trigger it true to filter out tree group preds and keep only single tree preds
    :param device: "cuda" if available, else "cpu"
    :return: preds: box and mask predictions as a list
    """
    pred_model.to(device)
    pred_model.eval()

    pbar = tqdm(pred_loader, desc="Predicting")
    preds = []
    with torch.no_grad():
        for batch in pbar:
            # get the predictions
            images, metadata = batch
            images = [image.to(device) for image in images]
            outputs = pred_model(images)
            # collect all predictions
            collected_preds = collect_preds_geo(outputs, metadata, score_threshold, mask_threshold, filter_tree_groups)
            preds.extend(collected_preds)
    return preds


def read_json(file_path):
    assert file_path.endswith(".json"), "not a valid json file"
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_preds_json(output_preds, save_path):
    if not save_path.endswith(".json"):
        save_path = os.path.join(save_path, "output.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_preds, f, indent=2)


def export_predictions_geo(pred_list, output_dir, file_format="gpkg"):
    """
        Export predictions to one GeoPackage or GeoJSON per tile.
        Parameters
        ----------
        pred_list : list Output preds
        output_dir : str Directory where files will be saved.
        file_format : str Either 'gpkg' or 'geojson'.

        Returns
        -------
        list: Paths to exported files.
    """
    file_format = file_format.lower().replace(".", "")
    if file_format not in ("gpkg", "geojson"):
        raise ValueError("file_format must be 'gpkg' or 'geojson'")

    exported_files = []
    for tile_data in pred_list:
        tile_name = tile_data["tile"]
        tile_name, tile_ext = tile_name.split(".")
        crs = tile_data["crs"]
        annotations = tile_data["annotations"]
        features = []
        for ann in annotations:
            polygons = []
            for coords in ann["segmentation_geo"]:
                if len(coords) < 3:
                    continue
                polygon = shapely.geometry.Polygon(coords)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon)
            if not polygons:
                continue

            # combine disconnected components
            geometry = unary_union(polygons)
            features.append({
                "id": int(ann["id"]),
                "label": int(ann["label"]),
                "score": float(ann["score"]),
                # Pixel coordinates stored as JSON strings
                "box_px": json.dumps(ann["box"]),
                "seg_px": json.dumps(ann["segmentation"]),
                # Geographic bounding box
                "bbox_geo": json.dumps(ann.get("bbox_geo")),
                "area_px": float(ann["area"]),
                "geometry": geometry,
                "transform": json.dumps(tile_data["transform"]),
                "tile": tile_data["tile"]
            })
        if not features:
            print(f"Skipping {tile_name}: no valid geometries.")
            continue
        gdf = gpd.GeoDataFrame(
            features,
            geometry="geometry",
            crs=crs,
        )
        output_path = os.path.join(output_dir, f"{tile_name}.{file_format}")
        if file_format == "gpkg":
            gdf.to_file(
                output_path,
                driver="GPKG",
                layer="predictions",
                index=False,
            )
        else:
            gdf.to_file(
                output_path,
                driver="GeoJSON",
                index=False,
            )
        exported_files.append(output_path)
        print(f"Exported {tile_name}: {len(gdf)} detections")
    return exported_files


def load_preds_geo(input_file_path):
    """
        Load GeoPackage/GeoJSON predictions and reconstruct pred_list.

        Parameters
        ----------
        input_path : str A single file

        Returns
        -------
        dict: Reconstructed prediction dictionaries, compatible with the output structure of collect_preds().
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(input_file_path)

    gdf = gpd.read_file(input_file_path)
    annotations = []
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if isinstance(geometry, shapely.geometry.Polygon):
            geometries = [geometry]
        elif isinstance(geometry, shapely.geometry.MultiPolygon):
            geometries = list(geometry.geoms)
        else:
            continue

        segmentation_geo = []
        for polygon in geometries:
            coords = [[float(x), float(y)] for x, y in polygon.exterior.coords]
            segmentation_geo.append(coords)

        box = json.loads(row["box_px"])
        segmentation = json.loads(row["seg_px"])
        bbox_geo = json.loads(row["bbox_geo"])
        annotations.append({
            "id": int(row["id"]),
            "label": int(row["label"]),
            "score": float(row["score"]),
            "box": box,
            "segmentation": segmentation,
            "bbox_geo": bbox_geo,
            "segmentation_geo": segmentation_geo,
            "area": float(row["area_px"]),
        })
    return {
        "tile": input_file_path,
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "num_detections": len(annotations),
        "annotations": annotations,
    }


def plot_annotations(image, annotations):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    for anno in annotations:
        # Draw Masks
        polygon = anno['segmentation'][0]
        polygon = np.asarray(polygon).reshape(-1, 2)
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="blue",
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(patch)
        # Draw Boxes
        box = anno['box']
        x1, y1, x2, y2 = box
        rect = Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")
    plt.show()


path_ckpt = "/home/jazib/projects/savedmodels/s1_convnext_lora_full"
model = load_pretrained_maskrcnn_model(path_ckpt)

# data_path = "/home/jazib/projects/data/oam-tcd-3-test-imgs/"
data_path = "/home/jazib/projects/data/MarieFlambeck/tiles/"
# dataset = MinimalPredictTifDataset(data_path)
loader = create_predict_loader(data_path, batch_size=1, num_workers=1, oam_normalization=False)
##
ex = next(iter(loader))
##
imgs, meta = ex
##
def match_histogram_to_reference(target_image, ref_hist):
    """
    Matches the histogram of a grayscale target image to a given reference histogram.

    Parameters:
    - target_image: 2D NumPy array (grayscale image, e.g., uint8)
    - ref_hist: 1D NumPy array representing the desired reference histogram counts

    Returns:
    - matched_image: 2D NumPy array with the histogram matched to ref_hist
    """
    target_image = np.asarray(target_image)
    ref_hist = np.asarray(ref_hist, dtype=np.float64)
    num_bins = len(ref_hist)

    # 1. Compute the histogram and CDF of the target image
    target_hist, _ = np.histogram(target_image.flatten(), bins=num_bins, range=(0, num_bins))
    target_cdf = np.cumsum(target_hist).astype(np.float64)
    if target_cdf[-1] > 0:
        target_cdf /= target_cdf[-1]  # Normalize CDF to [0, 1]

    # 2. Compute the CDF of the reference histogram
    ref_cdf = np.cumsum(ref_hist)
    if ref_cdf[-1] > 0:
        ref_cdf /= ref_cdf[-1]  # Normalize CDF to [0, 1]

    # 3. Create a mapping lookup table using interpolation (Quantile Matching)
    ref_pixels = np.arange(num_bins)
    mapping = np.interp(target_cdf, ref_cdf, ref_pixels)

    # 4. Map the target image pixels to the new values
    matched_image = mapping[target_image.clip(0, num_bins - 1).astype(int)]

    return matched_image.astype(target_image.dtype)


def match_color_histogram(target_rgb, ref_hists):
    """
    ref_hists: A list or tuple of 3 1D numpy arrays [hist_R, hist_G, hist_B]
    """
    matched_channels = []
    for i in range(3):
        matched_channel = match_histogram_to_reference(target_rgb[..., i], ref_hists[i])
        matched_channels.append(matched_channel)

    return np.stack(matched_channels, axis=-1)


img_np = 255*imgs[0].detach().cpu().permute(1,2,0).numpy()

##
from utils import yaml_to_dict_parser

hist_dict = yaml_to_dict_parser("./preds/oam_biome_aligned_4_5_aligned_histcdf.yaml")
hist_tuple = (hist_dict["hist"]["red"], hist_dict["hist"]["green"], hist_dict["hist"]["blue"])

img_np_matched = match_color_histogram(img_np, hist_tuple)
##
plt.imshow(img_np_matched.astype("uint8"))
plt.show()





# crop_img = img[0].permute(1, 0, 2)
# crop_img = crop_img[0:3, 0:1024, 0:1024]
# # norm = Normalize(mean=OAM_NORM_MEAN, std=OAM_NORM_STDV)
# norm = Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )  # dinov3 normalization
# crop_img_norm = norm(crop_img)
##
model.to("cuda")
model.eval()
crop_img = [img.to("cuda") for img in imgs]
outputs = model(crop_img)
##
preds = collect_preds_geo(outputs, meta, score_threshold=0.05, filter_tree_groups=False)
##
exported = export_predictions_geo(preds, "./preds/", "gpkg")
##
# ex = dataset[0]
# tile = ex[0]
# tile = F.to_tensor(tile).permute(1,0,2)

##
# preds = predict(model, loader)
ex = next(iter(loader))

# model.to("cuda")
# model.eval()
# images, tile_names = ex
# images = [image.to("cuda") for image in images]
# outputs = model(images)
# p_list = collect_preds(tile_names, outputs, score_threshold=0.5)


# for batch in loader:
#     print(batch.size())
##



candidate = preds[0]
img = imread(os.path.join(data_path, candidate['tile']))
annotations = candidate['annotations']

plot_annotations(
    image=img,
    annotations=annotations
)
##


def get_pred_args():
    parser = argparse.ArgumentParser(description="Predicting configuration")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("oam_normalization", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_pred_args()
    pred_model = load_pretrained_maskrcnn_model(args.model_path)
    dataloader = create_predict_loader(args.data_path, args.batch_size, args.num_workers, args.oam_normalization)
    predictions = predict(pred_model, dataloader, args.score_threshold)
    save_preds_json(predictions, args.save_path)
