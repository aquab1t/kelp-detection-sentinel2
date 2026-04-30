#!/usr/bin/env python3
"""Batch process multiple Sentinel-2 scenes for kelp detection."""

import argparse
import subprocess
import sys
from pathlib import Path


def process_scene(scene_path, output_dir, model_type, roi_mask=None):
    """Process a single Sentinel-2 scene."""
    scene_name = Path(scene_path).name.replace('.SAFE', '')
    output_path = str(Path(output_dir) / f"{scene_name}_kelp.tif")

    script = Path(__file__).parent / f"run_inference_{model_type}.py"
    cmd = [sys.executable, str(script), "--scene", scene_path,
           "--output", output_path]

    if roi_mask:
        cmd.extend(["--roi-mask", roi_mask])

    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"{'='*60}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing {scene_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch kelp detection')
    parser.add_argument('--scenes-dir', required=True,
                        help='Directory containing .SAFE folders')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for classification GeoTIFFs')
    parser.add_argument('--model', choices=['1dcnn', '2dcnn'], default='2dcnn',
                        help='Model type (default: 2dcnn)')
    parser.add_argument('--roi-mask', help='ROI mask GeoTIFF')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_dirs = sorted(scenes_dir.glob('*.SAFE'))
    if not safe_dirs:
        print(f"No .SAFE directories found in {scenes_dir}")
        sys.exit(1)

    print(f"Found {len(safe_dirs)} scenes to process")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    results = {'success': 0, 'failed': 0}

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_scene, str(s), str(output_dir),
                    args.model, args.roi_mask
                ): s for s in safe_dirs
            }
            for future in as_completed(futures):
                if future.result():
                    results['success'] += 1
                else:
                    results['failed'] += 1
    else:
        for scene in safe_dirs:
            if process_scene(str(scene), str(output_dir),
                             args.model, args.roi_mask):
                results['success'] += 1
            else:
                results['failed'] += 1

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  Success: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total: {len(safe_dirs)}")


if __name__ == '__main__':
    main()
