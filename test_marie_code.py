from pathlib import Path
import math

import numpy as np
import rasterio
import geopandas as gpd

from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import reproject
# ============================================================
# 1. KONFIGURATION
# ============================================================

# Input
INPUT_TIF = Path(r"P:\projects\Flambeck\raster\flambeck_rgb_raw_v1.tif")

POLYGON_FILE = Path(r"P:\projects\Flambeck\Marie\Baumkronenpunkte_SAM\Preprocessing\AoI.gpkg")

# Prüfen, ob die Eingabedateien existieren
if not INPUT_TIF.is_file():
    raise FileNotFoundError(f"TIFF nicht gefunden: {INPUT_TIF}")

if not POLYGON_FILE.is_file():
    raise FileNotFoundError(
        f"Polygonlayer nicht gefunden: {POLYGON_FILE}"
    )

# Output
OUTPUT_DIR = Path(r"C:\Users\AWF\Documents\Masterarbeit Marie\Pre_Baumkronenpunkte_SAM\rgb_raw")

# Unterordner für Zwischenschritte und Tiles
INTERMEDIATE = OUTPUT_DIR / "intermediate"
TILE_DIR = OUTPUT_DIR / "tiles"

# Ordner automatisch erstellen
INTERMEDIATE.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)

# Zielauflösung in Metern pro Pixel
TARGET_RES = 0.1

# Tile-Einstellungen
TILE_SIZE = 1024
OVERLAP = 50
STRIDE = TILE_SIZE - OVERLAP

# Resampling für kontinuierliche Bilddaten
RESAMPLING = Resampling.bilinear

# Beispiel: lineare Skalierung anhand von Perzentilen
LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98

# Achtung: 0 wird hier als NoData reserviert.
NODATA = 0

# Prüfen, ob die Eingabedateien existieren
if not INPUT_TIF.is_file():
    raise FileNotFoundError(f"TIFF nicht gefunden: {INPUT_TIF}")

if not POLYGON_FILE.is_file():
    raise FileNotFoundError(
        f"Polygonlayer nicht gefunden: {POLYGON_FILE}"
    )

print("Eingabe-TIFF:", INPUT_TIF)
print("Polygonlayer:", POLYGON_FILE)
print("Ausgabeordner:", OUTPUT_DIR)
print("Zwischenordner:", INTERMEDIATE)
print("Tile-Ordner:", TILE_DIR)

# ============================================================
# 2. AOI AUSWÄHLEN UND ORIGINAL-TIFF AUSSCHNEIDEN
# ============================================================

aoi_path = INTERMEDIATE / "selected_aoi.gpkg"
cropped_path = INTERMEDIATE / "aoi_raw.tif"

# Polygonlayer direkt laden – ohne Attributfilter
selected = gpd.read_file(POLYGON_FILE)

if selected.empty:
    raise ValueError("Der Polygonlayer ist leer.")

if selected.crs is None:
    raise ValueError("Der Polygonlayer hat kein CRS.")

# Original-TIFF öffnen
with rasterio.open(INPUT_TIF) as src:

    if src.crs is None:
        raise ValueError("Das Eingabe-TIFF hat kein CRS.")

    # Polygon in das CRS des Originalrasters transformieren
    selected = selected.to_crs(src.crs)

    # AOI speichern
    selected.to_file(aoi_path, driver="GPKG")

    # Geometrien vorbereiten
    shapes = [
        geom.__geo_interface__
        for geom in selected.geometry
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        raise ValueError("Keine gültige Polygongeometrie gefunden.")

    # TIFF auf das Polygon zuschneiden
    cropped, cropped_transform = mask(
        src,
        shapes,
        crop=True,
        filled=False
    )

    # Rasterprofil aktualisieren
    profile = src.profile.copy()
    profile.update(
        height=cropped.shape[1],
        width=cropped.shape[2],
        transform=cropped_transform,
        compress="deflate",
        BIGTIFF="IF_SAFER"
    )

print("AOI gespeichert:", aoi_path)
print("Original-TIFF ausgeschnitten:", cropped_path)
print(
    f"AOI-Rastergröße: {cropped.shape[2]} × "
    f"{cropped.shape[1]} Pixel"
)
# Berechnung der Werte anhand p2 und p98 als Ober und Untergrenze
# Wertebreich bei 8bit 0 bis 255
# Brechnung für jedes Band reperat
# NoData=0
# ============================================================
# 2. TIFF IN UINT8 UMWANDELN
# ============================================================

uint8_path = INTERMEDIATE / "raster_uint8.tif"

with rasterio.open(Path(r"C:\Users\AWF\Documents\Masterarbeit Marie\Pre_Baumkronenpunkte_SAM\rgb_raw\intermediate\aoi_raw.tif")) as src:
    profile = src.profile.copy()

    profile.update(
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(uint8_path, "w", **profile) as dst:

        for band in range(1, src.count + 1):

            # Band einlesen
            data = src.read(
                band, masked=True
            ).astype(np.float32)

            # Gültige Pixel bestimmen
            valid = ~np.ma.getmaskarray(data)
            values = data.data[valid]

            if values.size == 0:
                raise ValueError(
                    f"Band {band} enthält keine gültigen Pixel."
                )

            # Perzentile berechnen
            p2, p98 = np.percentile(
                values,
                [2, 98]
            )

            # Ausgabearray mit uint8 erstellen
            scaled = np.zeros(
                data.shape,
                dtype=np.uint8
            )

            # Lineare Skalierung auf 0–255
            if p98 > p2:
                normalized = (
                    (data.data[valid] - p2)
                    / (p98 - p2)
                    * 255.0
                )

                scaled[valid] = np.clip(
                    np.rint(normalized),
                    0,
                    255
                ).astype(np.uint8)

            # Skaliertes Band speichern
            dst.write(scaled, band)

            # Verwendete Parameter dokumentieren
            dst.update_tags(
                band,
                scale_method="percentile_linear",
                lower_percentile="2",
                upper_percentile="98",
                lower_value=str(float(p2)),
                upper_value=str(float(p98))
            )

            print(
                f"Band {band}: "
                f"P2={p2:.3f}, "
                f"P98={p98:.3f}"
            )

print("1/5: uint8-Raster gespeichert:", uint8_path)

# ============================================================
# 3. AUF 0,1 M PRO PIXEL RESAMPLEN
# ============================================================

resampled_path = INTERMEDIATE / "raster_10cm_uint8.tif"

with rasterio.open(uint8_path) as src:
    if src.crs is None:
        raise ValueError("Das Raster hat kein CRS.")

    # 0,1 m erfordert ein projiziertes CRS in Metern.
    if src.crs.is_geographic:
        raise ValueError(
            "Das Raster verwendet geografische Koordinaten. "
            "Zuerst in ein geeignetes metrisches CRS "
            "projizieren."
        )

    bounds = src.bounds
    width = math.ceil((bounds.right - bounds.left) / TARGET_RES)
    height = math.ceil((bounds.top - bounds.bottom) / TARGET_RES)

    dst_transform = from_origin(
        bounds.left, bounds.top, TARGET_RES, TARGET_RES
    )

    profile = src.profile.copy()
    profile.update(
        width=width,
        height=height,
        transform=dst_transform,
        dtype="uint8",
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(resampled_path, "w", **profile) as dst:
        for band in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=rasterio.band(dst, band),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                dst_nodata=NODATA,
                resampling=RESAMPLING
            )

print("2/5: 10-cm-Raster gespeichert:", resampled_path)
# ============================================================
# 6. AOI-AUSSCHNITT IN 1024x1024-TILES ZERLEGEN
# ============================================================

with rasterio.open(resampled_path) as src:
    profile = src.profile.copy()
    tile_number = 0

    for row in range(0, src.height, STRIDE):
        for col in range(0, src.width, STRIDE):
            tile_number += 1

            # Immer vollständige 1024x1024-Tiles erzeugen.
            tile = np.full(
                (src.count, TILE_SIZE, TILE_SIZE),
                NODATA,
                dtype=np.uint8
            )

            # Nur den Bereich lesen, der innerhalb des
            # Quellrasters liegt.
            read_width = min(TILE_SIZE, src.width - col)
            read_height = min(TILE_SIZE, src.height - row)

            window = Window(
                col, row, read_width, read_height
            )

            data = src.read(window=window)

            tile[:, :read_height, :read_width] = data

            tile_transform = src.window_transform(
                Window(col, row, TILE_SIZE, TILE_SIZE)
            )

            tile_profile = profile.copy()
            tile_profile.update(
                width=TILE_SIZE,
                height=TILE_SIZE,
                transform=tile_transform,
                dtype="uint8",
                nodata=NODATA,
                compress="deflate",
                predictor=2
            )

            tile_path = TILE_DIR / f"tile_{tile_number:05d}.tif"

            with rasterio.open(tile_path, "w", **tile_profile) as dst:
                dst.write(tile)

print("5/5: Tiles gespeichert in:", TILE_DIR)