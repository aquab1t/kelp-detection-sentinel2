# Usage

## CLI Reference

### `run_inference_1dcnn.py`

Per-pixel binary kelp detection (kelp vs not-kelp):

```bash
python scripts/run_inference_1dcnn.py \
  --scene /path/to/S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM.SAFE \
  --output output/kelp_1dcnn.tif \
  [--model models/1dcnn_binary_int8.tflite] \
  [--scaler models/scaler_1dcnn_binary.joblib] \
  [--roi-mask roi_mask.tif] \
  [--threshold 0.5] \
  [--batch-size 50000]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--scene`, `-s` | Yes | - | Path to Sentinel-2 L1C `.SAFE` directory |
| `--output`, `-o` | Yes | - | Output kelp classification GeoTIFF path |
| `--model` | No | `models/1dcnn_binary_int8.tflite` | Path to TFLite model file |
| `--scaler` | No | `models/scaler_1dcnn_binary.joblib` | StandardScaler `.joblib` path |
| `--roi-mask` | No | - | ROI mask GeoTIFF (1 = process, 0 = skip) |
| `--threshold` | No | 0.5 | Decision threshold on kelp probability |
| `--batch-size` | No | 50000 | Pixels per inference batch |

Output: `uint8` GeoTIFF with `1` = kelp, `0` = everything else (no NoData).

### `run_inference_2dcnn.py`

Patch-based binary kelp detection with 11×11 spatial context:

```bash
python scripts/run_inference_2dcnn.py \
  --scene /path/to/S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM.SAFE \
  --output output/kelp_2dcnn.tif \
  [--model models/2dcnn_binary_int8.tflite] \
  [--scaler models/scaler_2dcnn_binary.joblib] \
  [--roi-mask roi_mask.tif] \
  [--threshold 0.5] \
  [--batch-size 10000] \
  [--no-progress]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--scene`, `-s` | Yes | - | Path to Sentinel-2 L1C `.SAFE` directory |
| `--output`, `-o` | Yes | - | Output kelp classification GeoTIFF path |
| `--model` | No | `models/2dcnn_binary_int8.tflite` | Path to TFLite model file |
| `--scaler` | No | `models/scaler_2dcnn_binary.joblib` | StandardScaler `.joblib` path |
| `--roi-mask` | No | - | ROI mask GeoTIFF (1 = process, 0 = skip) |
| `--threshold` | No | 0.5 | Decision threshold on kelp probability |
| `--batch-size` | No | 10000 | Patches per inference batch |
| `--no-progress` | No | - | Disable progress bar |

> **Memory note.** The 2D-CNN extracts an 11×11×9 patch for every pixel. For a full Sentinel-2 tile (10980×10980 ≈ 121 M pixels) that materialises ~489 GiB of float32 patches and will OOM. Always pass `--roi-mask` for full-scene inference; it tells the preprocessor to extract patches only inside the ROI.

Output: `uint8` GeoTIFF with `1` = kelp, `0` = everything else (no NoData).

### `create_roi_mask.py`

Build a coastal-buffer ROI mask from ESA WorldCover:

```bash
python scripts/create_roi_mask.py \
  --worldcover /path/to/ESA_WorldCover_10m_2021_v200_T11RPM.tif \
  --reference /path/to/S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM.SAFE \
  --output output/roi_mask.tif \
  [--buffer-distance 200]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--worldcover` | Yes | - | Path to ESA WorldCover 2021 GeoTIFF |
| `--reference` | Yes | - | Reference Sentinel-2 `.SAFE` directory or GeoTIFF for CRS/transform |
| `--output` | Yes | - | Output ROI mask GeoTIFF path |
| `--buffer-distance` | No | 200 | Coastal buffer in 10 m pixels (200 = 2 km from shore) |

Output: `uint8` GeoTIFF (1 = ROI, 0 = exclude).

### `batch_process.py`

Process multiple scenes:

```bash
python scripts/batch_process.py \
  --scenes-dir /path/to/safe_directories/ \
  --output-dir /path/to/output/ \
  [--model 2dcnn] \
  [--roi-mask roi_mask.tif] \
  [--workers 4]
```

Output files are named `{scene_name}_kelp.tif`.

---

## Python API

### Minimal example - 1D-CNN with ROI

```python
from kelp_detection import Sentinel2Loader, Preprocessor, KelpPredictor
import rasterio, numpy as np

loader = Sentinel2Loader("S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM.SAFE")
data = loader.load_bands()           # (9, H, W) float32 (DN)
metadata = loader.get_metadata()

with rasterio.open("roi_mask.tif") as r:
    roi = r.read(1).astype(bool)
roi_flat = roi.flatten()

pre = Preprocessor(model_type="1dcnn")
X = pre.prepare_for_inference(data)               # (H*W, 9)
X_roi = X[roi_flat]

predictor = KelpPredictor("models/1dcnn_binary_int8.tflite")
y_roi = predictor.predict_and_classify(X_roi, threshold=0.5)   # uint8 {0, 1}

flat = np.zeros(metadata["height"] * metadata["width"], dtype=np.uint8)
flat[roi_flat] = y_roi
class_map = predictor.reshape_to_2d(flat, metadata["height"], metadata["width"])
predictor.export_geotiff(class_map, metadata, "kelp_1dcnn.tif")
```

### 2D-CNN - pass ROI to the preprocessor

The 2D preprocessor supports `roi_mask` directly so it only extracts patches inside the ROI (avoids the 489 GiB explosion):

```python
from kelp_detection import Sentinel2Loader, Preprocessor, KelpPredictor
import rasterio, numpy as np

loader = Sentinel2Loader("S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM.SAFE")
data = loader.load_bands()
metadata = loader.get_metadata()

with rasterio.open("roi_mask.tif") as r:
    roi = r.read(1).astype(bool)

pre = Preprocessor(model_type="2dcnn")
X = pre.prepare_for_inference(data, roi_mask=roi)            # (n_roi, 11, 11, 9)

predictor = KelpPredictor("models/2dcnn_binary_int8.tflite")
y_roi = predictor.predict_and_classify(X, threshold=0.5)

flat = np.zeros(metadata["height"] * metadata["width"], dtype=np.uint8)
flat[roi.flatten()] = y_roi
class_map = predictor.reshape_to_2d(flat, metadata["height"], metadata["width"])
predictor.export_geotiff(class_map, metadata, "kelp_2dcnn.tif")
```

### Getting raw probabilities instead of a binary mask

```python
probs = predictor.predict_binary(X)        # float32 in [0, 1]
prob_map = np.full(metadata["height"] * metadata["width"], np.nan, dtype=np.float32)
prob_map[roi.flatten()] = probs
prob_map = prob_map.reshape(metadata["height"], metadata["width"])
```

`KelpPredictor.predict_binary()` auto-detects whether the model's TFLite output is already a sigmoid probability (1D-CNN) or a logit (2D-CNN), and applies sigmoid only when needed.

---

## ROI Mask Workflow

1. **Download ESA WorldCover** for your tile (https://esa-worldcover.org). Pick the 2021 v200 10 m product for your Sentinel-2 MGRS tile.
2. **Build the mask** with `create_roi_mask.py`. The script reprojects WorldCover to the Sentinel-2 grid, isolates permanent water bodies (class 80), computes a Euclidean distance transform, and selects water pixels within `--buffer-distance` of shore.
3. **Reuse across scenes** that share the same MGRS tile and processing grid.
4. **Adjust buffer distance** for sites with offshore beds - try `--buffer-distance 400` for ~4 km buffers.

---

## Threshold Tuning

The default decision threshold is `0.5`. Move it to trade kelp precision and recall:

| Threshold | Effect |
|---|---|
| 0.3–0.4 | Higher recall - catches faint canopy and thin edges, more false positives in turbid water and glint |
| 0.5 | Default; balanced |
| 0.6–0.8 | Higher precision - only confident kelp; misses sparse canopy |

For a probability-aware analysis, write the raw probabilities (see Python API above) and threshold downstream - that's cheaper than re-running inference.

---

## Output Format

Both inference scripts produce the same GeoTIFF schema:

| Property | Value |
|---|---|
| Data type | `uint8` |
| Values | `1` = kelp, `0` = everything else |
| Compression | LZW |
| CRS / transform | Inherited from the source Sentinel-2 scene |

### Visualisation

```python
import matplotlib.pyplot as plt, matplotlib.colors as mcolors, rasterio, numpy as np

with rasterio.open("kelp_2dcnn.tif") as src:
    kelp = src.read(1).astype(float)
cmap = mcolors.ListedColormap(["#1F4E79", "#2ECC71"])  # not-kelp, kelp
norm = mcolors.BoundaryNorm([0, 1, 2], cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(kelp, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5])
cbar.ax.set_yticklabels(["Not Kelp", "Kelp"])
plt.tight_layout()
plt.savefig("kelp_classification.png", dpi=150)
```

For QGIS: load the GeoTIFF, set "Paletted/Unique values" symbology with `0`=transparent (or dark blue) and `1`=green.

---

## Applying to New Sites

### 1. ROI

Download WorldCover for the new MGRS tile and build a site-specific ROI mask. Sites with offshore beds may benefit from a wider buffer (`--buffer-distance 400`+).

### 2. Threshold sweep

Run inference at several thresholds (0.3, 0.5, 0.7) on a few well-known scenes and visually compare. Pick the value that matches your priorities (recall vs precision).

### 3. Validation

The training site is Cedros Island, Baja California. Models have not been benchmarked on other sites - validate against:

- **Field surveys** - GPS-tagged kelp presence/absence from boat or drone surveys.
- **High-resolution imagery** - Planet Scope, Worldview, or aerial photos for visual interpretation.
- **Temporal consistency** - *Macrocystis* is perennial, so confident detections should persist across consecutive low-tide, low-cloud dates.

### 4. Known limitations

- **Cloud and glint** - both can drive false positives in coastal pixels. The ROI buffer reduces but does not eliminate this.
- **Turbid plumes** - river outflow with suspended sediment can mimic kelp NIR reflectance.
- **Tide and season** - canopy exposure varies with tide; *Macrocystis* canopy is largest in late summer/fall in the Northern Hemisphere.
- **Site transfer** - the 1D-CNN training set uses a pixel-wise random split, so reported test scores overstate generalisation to unseen sites. Treat the 99% test accuracies as upper bounds.
