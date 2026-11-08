##
import math
import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import geopandas as gpd


INPUT_TIF = "/home/jazib/projects/data/MarieFlambeck/big_tile/flambeck_rgb_raw_v1.tif"
POLYGON_FILE = "/home/jazib/projects/data/MarieFlambeck/big_tile/AoI.gpkg"
OUTPUT_DIR = "/home/jazib/projects/data/MarieFlambeck/exp/"

TARGET_RES = 0.1
TILE_SIZE = 1024
OVERLAP = 50
STRIDE = TILE_SIZE - OVERLAP

NODATA = 255  # black, alt: 0 for white but really I want averages or something
LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98


polygon_aoi = gpd.read_file(POLYGON_FILE)
##
with rasterio.open(INPUT_TIF) as tif:
    polygon_aoi = polygon_aoi.to_crs(tif.crs)
    shapes = [
        geom.__geo_interface__
        for geom in polygon_aoi.geometry
        if geom is not None and not geom.is_empty
    ]
    cropped_tif, cropped_transform = mask(tif,
                                          shapes,
                                          crop=True,
                                          filled=False)
    profile = tif.profile.copy()
    profile.update(
        height=cropped_tif.shape[1],
        width=cropped_tif.shape[2],
        transform=cropped_transform,
        compress="deflate",
        BIGTIFF="IF_SAFER"
    )



##
