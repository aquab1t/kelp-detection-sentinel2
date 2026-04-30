#!/usr/bin/env python3
"""End-to-end example: run both kelp models on the bundled Cedros scene crop.

The data files in this directory are a 1024×1024 (10 km × 10 km) crop of
S2A_MSIL1C_20180801T181011_T11RPM around a kelp bed off Cedros Island.
The crop is small enough to ship in the repo so anyone can reproduce the
inference pipeline without downloading a full Sentinel-2 SAFE folder.

Inputs (in this directory):
    cedros_2018-08-01_bands.tif   9-band uint16 GeoTIFF (B02,B03,B04,B08,B05,B06,B07,B8A,B11)
    cedros_worldcover.tif         ESA WorldCover 2021 v200, reprojected to the scene grid

Outputs (written next to this script when you run it):
    cedros_roi.tif                Coastal-buffer ROI mask
    cedros_kelp_1dcnn.tif         1D-CNN binary kelp map (1=kelp, 0=else)
    cedros_kelp_2dcnn.tif         2D-CNN binary kelp map (1=kelp, 0=else)

Usage:
    python examples/run_example.py
"""

from pathlib import Path
import sys

import numpy as np
import rasterio

# Make the package importable when running directly from a clone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kelp_detection import Preprocessor, KelpPredictor, create_coastal_buffer


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

BANDS_TIF = HERE / "cedros_2018-08-01_bands.tif"
WORLDCOVER_TIF = HERE / "cedros_worldcover.tif"
OUT_ROI = HERE / "cedros_roi.tif"
OUT_1D = HERE / "cedros_kelp_1dcnn.tif"
OUT_2D = HERE / "cedros_kelp_2dcnn.tif"


def main():
    # 1. Load the bundled 9-band scene crop
    with rasterio.open(BANDS_TIF) as src:
        bands = src.read().astype(np.float32)        # (9, H, W) DN
        height, width = src.height, src.width
        transform, crs = src.transform, src.crs
    print(f"Loaded bands: shape={bands.shape}, dtype={bands.dtype}, "
          f"range=[{bands.min():.0f}, {bands.max():.0f}]")

    # 2. Build a coastal-buffer ROI from the bundled WorldCover crop (water class = 80)
    with rasterio.open(WORLDCOVER_TIF) as src:
        worldcover = src.read(1)
    water_mask = (worldcover == 80)
    roi = create_coastal_buffer(water_mask, buffer_distance=200)   # 200 px × 10 m = 2 km
    print(f"ROI: {int(roi.sum()):,} pixels in 2 km coastal buffer")

    # Save ROI for inspection
    with rasterio.open(
        OUT_ROI, "w", driver="GTiff", dtype="uint8", count=1,
        width=width, height=height, crs=crs, transform=transform, compress="LZW",
    ) as dst:
        dst.write(roi.astype(np.uint8), 1)
    print(f"Wrote {OUT_ROI.name}")

    # 3. Inference - 1D-CNN
    pre1 = Preprocessor(model_type="1dcnn",
                        scaler_path=str(REPO / "models" / "scaler_1dcnn_binary.joblib"))
    X1 = pre1.prepare_for_inference(bands)              # (H*W, 9)
    p1 = KelpPredictor(str(REPO / "models" / "1dcnn_binary_int8.tflite"))
    roi_flat = roi.flatten()
    y1_roi = p1.predict_and_classify(X1[roi_flat], threshold=0.5)
    y1 = np.zeros(height * width, dtype=np.uint8)
    y1[roi_flat] = y1_roi
    map_1d = y1.reshape(height, width)
    print(f"1D-CNN: {int(map_1d.sum()):,} kelp px = {map_1d.sum()/10000:.2f} km²")

    # 4. Inference - 2D-CNN (ROI is passed to the preprocessor to avoid 489 GiB patch expansion)
    pre2 = Preprocessor(model_type="2dcnn",
                        scaler_path=str(REPO / "models" / "scaler_2dcnn_binary.joblib"))
    X2 = pre2.prepare_for_inference(bands, roi_mask=roi)   # (n_roi, 11, 11, 9)
    p2 = KelpPredictor(str(REPO / "models" / "2dcnn_binary_int8.tflite"))
    y2_roi = p2.predict_and_classify(X2, threshold=0.5)
    y2 = np.zeros(height * width, dtype=np.uint8)
    y2[roi_flat] = y2_roi
    map_2d = y2.reshape(height, width)
    print(f"2D-CNN: {int(map_2d.sum()):,} kelp px = {map_2d.sum()/10000:.2f} km²")

    # 5. Write GeoTIFFs (1=kelp, 0=else, no NoData)
    profile = dict(driver="GTiff", dtype="uint8", count=1,
                   width=width, height=height, crs=crs, transform=transform,
                   compress="LZW", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(OUT_1D, "w", **profile) as dst:
        dst.write(map_1d, 1)
    with rasterio.open(OUT_2D, "w", **profile) as dst:
        dst.write(map_2d, 1)
    print(f"Wrote {OUT_1D.name} and {OUT_2D.name}")


if __name__ == "__main__":
    main()
