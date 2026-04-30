#!/usr/bin/env python3
"""Create ROI mask for kelp detection from ESA WorldCover data."""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent.parent))

from kelp_detection import LandMask, create_coastal_buffer


def get_reference_metadata(reference_path):
    """Get CRS and transform from a reference GeoTIFF or Sentinel-2 scene."""
    ref = Path(reference_path)

    if ref.is_dir() and ref.suffix == '.SAFE':
        from kelp_detection import Sentinel2Loader
        loader = Sentinel2Loader(str(ref))
        _ = loader.load_bands()
        meta = loader.get_metadata()
        return meta['height'], meta['width'], meta['transform'], meta['crs']
    else:
        with rasterio.open(str(ref)) as src:
            return src.height, src.width, src.transform, src.crs


def main():
    parser = argparse.ArgumentParser(
        description='Create ROI mask for kelp detection from ESA WorldCover'
    )
    parser.add_argument('--worldcover', required=True,
                        help='Path to ESA WorldCover GeoTIFF')
    parser.add_argument('--reference', required=True,
                        help='Reference Sentinel-2 scene (.SAFE) or GeoTIFF '
                             'for CRS/transform')
    parser.add_argument('--output', required=True,
                        help='Output ROI mask GeoTIFF')
    parser.add_argument('--buffer-distance', type=int, default=200,
                        help='Coastal buffer distance in pixels at 10m '
                             '(default: 200 = 2km)')
    args = parser.parse_args()

    print(f"Loading reference metadata from: {args.reference}")
    height, width, transform, crs = get_reference_metadata(args.reference)
    print(f"  Scene dimensions: {width}x{height}")
    print(f"  CRS: {crs}")

    print(f"Creating water mask from ESA WorldCover: {args.worldcover}")
    mask = LandMask(args.worldcover)
    water_mask = mask.get_water_mask((height, width), transform, crs)
    water_pixels = water_mask.sum()
    print(f"  Water pixels: {water_pixels:,} ({100*water_pixels/(height*width):.1f}%)")

    print(f"Applying coastal buffer ({args.buffer_distance} pixels = "
          f"{args.buffer_distance * 10}m)...")
    roi = create_coastal_buffer(water_mask, buffer_distance=args.buffer_distance)
    roi_pixels = roi.sum()
    print(f"  ROI pixels: {roi_pixels:,} ({100*roi_pixels/(height*width):.1f}%)")

    print(f"Exporting ROI mask: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'nodata': 0,
        'compress': 'lzw',
        'tiled': True,
    }

    roi_uint8 = roi.astype(np.uint8)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(roi_uint8, 1)

    print(f"Saved: {output_path}")
    print(f"  ROI coverage: {roi_pixels:,} pixels ({roi_pixels*0.0001:.2f} km²)")
    print("\nDone!")


if __name__ == '__main__':
    main()
