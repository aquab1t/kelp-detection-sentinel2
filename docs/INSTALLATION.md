# Installation

## Requirements

- Python >= 3.8, < 3.12
- GDAL >= 3.0 (required by rasterio)

## pip

```bash
pip install kelp-detection-sentinel2
```

Or from source:

```bash
git clone https://github.com/aquab1t/kelp-detection-sentinel2.git
cd kelp-detection-sentinel2
pip install .
```

## conda

```bash
conda create -n kelp-detection python=3.10
conda activate kelp-detection
conda install -c conda-forge rasterio
pip install kelp-detection-sentinel2
```

Installing rasterio via conda-forge avoids GDAL compilation issues (see Troubleshooting below).

## TFLite runtime

The package uses `tflite-runtime` by default for lightweight CPU inference. An alternative runtime is available:

| Runtime | Install | Notes |
|---------|---------|-------|
| `tflite-runtime` | `pip install tflite-runtime>=2.7.0` | Default. Small footprint (~2 MB). |
| `ai-edge-litert` | `pip install ai-edge-litert>=1.0.0` | Google's successor runtime. Preferred on ARM/M1 where tflite-runtime wheels may be unavailable. |

The predictor auto-detects which runtime is installed: it tries `ai_edge_litert` first, then falls back to `tflite_runtime`. You do not need to configure anything - just install one of them.

```bash
# Option A: default
pip install tflite-runtime

# Option B: if tflite-runtime fails on your platform
pip install ai-edge-litert
```

## Verification

After installation, verify the package imports correctly:

```bash
python -c "from kelp_detection import Sentinel2Loader, Preprocessor, KelpPredictor, LandMask; print('OK')"
```

## Troubleshooting

### rasterio / GDAL dependencies

rasterio requires GDAL C libraries. If `pip install rasterio` fails with build errors:

1. **Use conda-forge** (recommended): `conda install -c conda-forge rasterio`
2. **Use pre-built wheels**: `pip install rasterio --only-binary=:all:` - available for Linux x86_64, macOS, and Windows.
3. **Install GDAL from system packages**: Install GDAL >= 3.0 via your package manager, then set `GDAL_CONFIG` before pip install.
   - Ubuntu/Debian: `sudo apt install libgdal-dev`
   - macOS: `brew install gdal`
   - Fedora: `sudo dnf install gdal-devel`

### tflite-runtime on ARM / Apple Silicon

`tflite-runtime` does not publish wheels for `aarch64` or macOS ARM in all Python versions. Workarounds:

- Install `ai-edge-litert` instead: `pip install ai-edge-litert`
- Use Rosetta Python on macOS: run under `arch -x86_64`
- Build from source: `pip install tflite-runtime --no-binary=tflite-runtime` (requires Bazel)

### Python version compatibility

| Python | tflite-runtime | ai-edge-litert | Status |
|--------|---------------|----------------|--------|
| 3.8    | Yes           | No             | Supported |
| 3.9    | Yes           | Yes            | Supported |
| 3.10   | Yes           | Yes            | Recommended |
| 3.11   | Partial wheels | Yes           | May need ai-edge-litert |
| 3.12+  | No wheels     | Yes            | Use ai-edge-litert |

If you see `No matching distribution found for tflite-runtime`, switch to `ai-edge-litert` or downgrade to Python 3.10.

### numpy version conflict

Both rasterio and tflite-runtime pin numpy. If you get a version conflict:

```bash
pip install "numpy>=1.20,<2.0"
pip install kelp-detection-sentinel2 --no-deps
pip install rasterio scipy joblib tqdm
```
