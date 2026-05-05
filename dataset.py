import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from tifffile import imread
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


class OAMTCDCOCODataset(Dataset):
    def __init__(
            self,
            root_dir,
            split="train",
            folds=[0],  # Changed to a list to support multiple folds
            transforms=None,
            return_masks=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.folds = folds if isinstance(folds, list) else [folds]
        self.transforms = transforms
        self.return_masks = return_masks

        self.images = []
        self.annotations = []
        self.img_id_to_fold = {}  # Map images to their specific fold directory

        # -----------------------------
        # Aggregate Data Across Folds
        # -----------------------------
        if split == "train":
            for f_idx in self.folds:
                ann_file = os.path.join(root_dir, f"coco_annotations_fold_{f_idx}.json")
                with open(ann_file, "r") as f:
                    data = json.load(f)

                # We need to track which image belongs to which fold for path resolution
                for img in data["images"]:
                    self.img_id_to_fold[img["id"]] = f_idx

                self.images.extend(data["images"])
                self.annotations.extend(data["annotations"])
        else:
            # Test split logic remains similar
            ann_path = os.path.join(root_dir, "coco_annotations_test.json")
            with open(ann_path, "r") as f:
                data = json.load(f)
            self.images = data["images"]
            self.annotations = data["annotations"]

        # Indexing for fast lookup
        self.imgs_by_id = {img["id"]: img for img in self.images}
        self.anns_by_image = {}
        for ann in self.annotations:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Filter out images without annotations
        self.image_ids = [i for i in self.imgs_by_id if len(self.anns_by_image.get(i, [])) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.imgs_by_id[image_id]
        file_name_prefix = os.path.splitext(img_info["file_name"])[0]

        # Resolve paths based on fold
        if self.split == "train":
            fold = self.img_id_to_fold[image_id]
            img_dir = os.path.join(self.root_dir, "train", f"fold_{fold}")
            mask_dir = os.path.join(self.root_dir, "train_masks", f"fold_{fold}")
        else:
            img_dir = os.path.join(self.root_dir, "test")
            mask_dir = os.path.join(self.root_dir, "test_masks")

        # Load Image
        # image = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        image = imread(os.path.join(img_dir, img_info["file_name"]))
        width, height, _ = image.shape

        # Prepare Target
        anns = self.anns_by_image[image_id]
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        labels = [a["category_id"] for a in anns]

        if self.transforms:
            # Prepare inputs for Albumentations
            input_params = {
                "image": image,
                "bboxes": boxes,
                "labels": labels
            }

            # Load and convert bool masks to uint8 on the fly for Albumentations compatibility
            if self.return_masks:
                mask_path = os.path.join(mask_dir, f"{file_name_prefix}_masks.npz")
                # Convert bool to uint8 here
                input_params["masks"] = np.load(mask_path)['masks'].astype(np.uint8)

            # Apply transforms
            augmented = self.transforms(**input_params)

            image = augmented["image"]
            # Update target values with augmented versions
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

            # Repackage the target dictionary
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4),
                                                                                               dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
                "area": torch.as_tensor([a["area"] for a in anns], dtype=torch.float32),  # Note: see caveat below
                "iscrowd": torch.as_tensor([a.get("iscrowd", 0) for a in anns], dtype=torch.int64),
            }

            if self.return_masks and "masks" in augmented:
                # Albumentations returns a list of masks, we stack them back to (N, H, W)
                target["masks"] = torch.as_tensor(np.array(augmented["masks"]), dtype=torch.uint8)

        else:
            # Fallback for no transforms
            image = F.to_tensor(image)
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
                "area": torch.as_tensor([a["area"] for a in anns], dtype=torch.float32),
                "iscrowd": torch.as_tensor([a.get("iscrowd", 0) for a in anns], dtype=torch.int64),
            }
            if self.return_masks:
                mask_path = os.path.join(mask_dir, f"{file_name_prefix}_masks.npz")
                target["masks"] = torch.as_tensor(np.load(mask_path)['masks'], dtype=torch.uint8)

        return image, target


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
        # A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
        #                            contrast_limit=(0.2, 0.3),
        #                            p=0.2),
        A.RandomGamma(gamma_limit=(100, 140), p=0.2),
        A.RandomToneCurve(scale=0.1, p=0.2),
        A.Normalize(mean=0, std=1, max_pixel_value=255),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))