#!/usr/bin/env python3
"""Run 1D-CNN Binary Kelp Detection Inference."""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from kelp_detection import Sentinel2Loader, Preprocessor, KelpPredictor


def main():
    parser = argparse.ArgumentParser(
        description='Run 1D-CNN binary kelp detection on Sentinel-2 L1C scene'
    )
    parser.add_argument('--scene', '-s', required=True,
                        help='Path to Sentinel-2 L1C .SAFE directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output kelp classification GeoTIFF path (uint8 0/1)')
    parser.add_argument('--model', default='models/1dcnn_binary_int8.tflite',
                        help='Path to TFLite model (default: models/1dcnn_binary_int8.tflite)')
    parser.add_argument('--scaler', default='models/scaler_1dcnn_binary.joblib',
                        help='Path to scaler file (default: models/scaler_1dcnn_binary.joblib)')
    parser.add_argument('--roi-mask',
                        help='Path to ROI mask GeoTIFF (1=process, 0=skip)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Batch size for inference (default: 50000)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Kelp detection threshold (default: 0.5)')
    args = parser.parse_args()

    print(f"Loading Sentinel-2 scene: {args.scene}")
    loader = Sentinel2Loader(args.scene)
    data = loader.load_bands()
    metadata = loader.get_metadata()
    print(f"  Loaded {data.shape[0]} bands, shape: {data.shape[1:]}")

    print("Preprocessing data...")
    preprocessor = Preprocessor(model_type='1dcnn', scaler_path=args.scaler)
    data_ready = preprocessor.prepare_for_inference(data)
    print(f"  Preprocessed shape: {data_ready.shape}")

    roi_mask = None
    if args.roi_mask:
        print(f"Loading ROI mask: {args.roi_mask}")
        with rasterio.open(args.roi_mask) as mask_src:
            roi_mask = mask_src.read(1).astype(bool)

    print("Running 1D-CNN inference...")
    predictor = KelpPredictor(args.model)

    if roi_mask is not None:
        roi_flat = roi_mask.flatten()
        data_roi = data_ready[roi_flat]
        classification_roi = predictor.predict_and_classify(
            data_roi, batch_size=args.batch_size
        )
        classification_1d = np.zeros(data_ready.shape[0], dtype=np.uint8)
        classification_1d[roi_flat] = classification_roi
    else:
        classification_1d = predictor.predict_and_classify(
            data_ready, batch_size=args.batch_size
        )

    classification_2d = predictor.reshape_to_2d(
        classification_1d, metadata['height'], metadata['width']
    )

    stats = predictor.get_class_statistics(classification_2d)
    print("\nClassification statistics:")
    for class_id, stat in stats.items():
        print(f"  {stat['name']}: {stat['pixels']:,} pixels "
              f"({stat['percentage']:.1f}%, {stat['area_km2']:.2f} km²)")

    print(f"\nExporting kelp classification map: {args.output}")
    predictor.export_geotiff(classification_2d, metadata, args.output, nodata=None)
    print("\nDone!")


if __name__ == '__main__':
    main()
