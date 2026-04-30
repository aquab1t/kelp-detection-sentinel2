# Methodology

## Sentinel-2 MSI Bands

Both models use 9 bands from the Sentinel-2 Multi-Spectral Instrument (MSI), spanning visible to short-wave infrared:

| Band | Name | Central Wavelength | Native Resolution |
|------|------|--------------------|-------------------|
| B02  | Blue | 492.4 nm | 10 m |
| B03  | Green | 559.8 nm | 10 m |
| B04  | Red | 664.6 nm | 10 m |
| B05  | Red Edge 1 | 704.1 nm | 20 m |
| B06  | Red Edge 2 | 740.5 nm | 20 m |
| B07  | Red Edge 3 | 782.8 nm | 20 m |
| B08  | NIR | 832.8 nm | 10 m |
| B8A  | Narrow NIR | 864.7 nm | 20 m |
| B11  | SWIR 1 | 1613.7 nm | 20 m |

These bands target kelp's spectral signature: red-edge bands (B05–B07) capture vegetation reflectance, NIR (B08, B8A) separates kelp canopy from open water, and B11 (SWIR) helps suppress bright clouds and glint that would otherwise be mistaken for canopy.

## Preprocessing Pipeline

### 1. Band order

Bands are loaded in the order `[B02, B03, B04, B08, B05, B06, B07, B8A, B11]` - 10 m bands first, then 20 m bands. This matches the order used during training; using a different order silently degrades inference quality.

### 2. Band resampling (20 m → 10 m)

Five 20 m bands (B05, B06, B07, B8A, B11) are upsampled 2× to 10 m using bilinear interpolation (`scipy.ndimage.zoom`, `order=1`). The four 10 m bands (B02, B03, B04, B08) pass through. After this step all 9 bands share the same 10 m grid.

### 3. StandardScaler normalization

Each band is independently normalised with a fitted `StandardScaler` saved as `scaler_{1,2}dcnn_binary.joblib`. The two models were trained on differently shaped inputs and have separate scalers; mixing them will produce wrong outputs.

### 4. Patch extraction (2D-CNN only)

The 2D-CNN consumes 11×11×9 patches centred on each pixel. The preprocessor:

1. **Reflect-pads** the normalised image by 5 pixels on each side (avoids zero-padding artefacts at the scene edges).
2. Uses a vectorised sliding-window view to extract patches. When `roi_mask` is provided, only patches centred on ROI pixels are materialised - the full-tile expansion would otherwise need ~489 GiB of float32 memory.

For the 1D-CNN the normalised image is simply reshaped from `(9, H, W)` to `(H·W, 9)` - no spatial context.

## Model Architectures

### 1D-CNN - pixel-wise binary classifier

Each pixel is treated as an independent 9-element spectral vector and classified as kelp or not-kelp.

```
Input: (batch, 9)
  ├─ SepConv1D(32) + BatchNorm + ReLU + SE → (batch, 32)
  ├─ SepConv1D(64) + BatchNorm + ReLU + SE → (batch, 64)
  ├─ SepConv1D(128) + BatchNorm + ReLU + SE → (batch, 128)
  ├─ GlobalMaxPool1D → (batch, 128)
  ├─ Dense(64) + ReLU
  └─ Dense(1) + Sigmoid → (batch, 1)
```

- **Separable Conv1D** - depthwise + pointwise convolution; reduces parameter count vs a standard Conv1D.
- **Squeeze-and-Excitation (SE)** - global average pool → Dense(n/4)+ReLU → Dense(n)+Sigmoid → channel-wise scale; learns which spectral channels matter for each sample.
- **Final activation: sigmoid baked into the graph.** The exported TFLite raw output is already a probability in `[0, 1]`.

### 2D-CNN - patch-based binary classifier

An 11×11×9 spatial neighbourhood is classified as kelp or not-kelp at the centre pixel.

```
Input: (batch, 11, 11, 9)
  ├─ Block 1: Conv2D(32) + BN + ReLU + MaxPool → (batch, 5, 5, 32)
  ├─ Block 2: Conv2D(64) + BN + ReLU + MaxPool → (batch, 2, 2, 64)
  ├─ Block 3: Conv2D(128) + BN + ReLU         → (batch, 2, 2, 128)
  ├─ AdaptiveAvgPool2D(1×1) → Flatten          → (batch, 128)
  ├─ Dense(128) + ReLU + Dropout(0.5)
  └─ Dense(1)                                  → (batch, 1)  [logit]
```

- **Spatial context** - the 11×11 patch lets the model use texture, canopy structure, and gradients at the water surface to separate kelp from glint and macroalgae shadow.
- **Final activation: none in the graph (logit output).** Apply sigmoid (`scipy.special.expit`) after inference to get a probability. `KelpPredictor.predict_binary()` does this for you.

## Quantization

Both models are exported as **INT8 weight-quantized TFLite** with float32 I/O.

- 1D-CNN: TF Keras → TFLite via `tf.lite.TFLiteConverter` with `Optimize.DEFAULT` (54 KB).
- 2D-CNN: PyTorch → ONNX → TFLite via `onnx2tf`, with the same weight-quantization optimization (466 KB).

Activations remain float32 at inference time; only the weights are stored as INT8. This gives roughly 4× file-size reduction with negligible accuracy loss compared to float32 weights.

## Training

| Aspect | 1D-CNN | 2D-CNN |
|---|---|---|
| Framework | TensorFlow / Keras | PyTorch |
| Loss | Binary cross-entropy with class weights | `BCEWithLogitsLoss` |
| Optimizer | Adam | AdamW (lr = 3e-4) |
| LR schedule | Reduce on plateau | CosineAnnealing |
| Batch size | 256 | 512 |
| Augmentation | Gaussian noise (σ = 0.001) | None |
| Training samples | 68,176 pixels (40,391 not-kelp, 27,785 kelp) | 220,800 patches (balanced 4-class → binary) |
| Test split | Random pixel-wise, stratified, 15% (`SEED=42`) | Held-out 4-class patch split mapped to binary |

Training site for both: Cedros Island, Baja California, Mexico (Sentinel-2 L1C imagery, 2015–2025).

### Data leakage caveat for the 1D-CNN

The 1D-CNN training set is split randomly across pixels, **not grouped by scene**. Neighbouring pixels in the same image have nearly identical spectra (spatial autocorrelation), so a random split puts almost-identical samples in both train and test - inflating the test-set score relative to performance on truly novel scenes.

The 2D-CNN avoids this somewhat: its test patches were stratified at the 4-class dataset stage and held out before training. But both models share the same single-site training data - generalisation to other latitudes, seasons, and water types is not measured here.

## ROI Masking

Inference is restricted to a region of interest (ROI) to skip land interiors and deep ocean:

1. **ESA WorldCover 2021 v200** at 10 m. Class 80 = permanent water bodies.
2. The WorldCover map is reprojected (nearest-neighbour) to the Sentinel-2 scene's CRS and grid. To prevent the WorldCover tile boundary from creating a phantom land ring at the edge of the Sentinel-2 grid, the destination buffer is initialised with `WATER_CLASS=80` and reprojection runs with `init_dest_nodata=False`.
3. A **binary water mask** is built (`land_cover == 80`).
4. A Euclidean **distance transform** computes each water pixel's distance to the nearest land pixel.
5. The **coastal buffer** selects water pixels within `buffer_distance` pixels of land. Default 200 px × 10 m = **2 km** from shore.

This focuses computation on the nearshore zone where giant kelp grows.

## Performance

### 1D-CNN binary

Held-out 15% pixel-wise split (n = 10,227):

| Metric | Not Kelp | Kelp |
|---|---|---|
| Precision | 0.9796 | 0.9998 |
| Recall | 0.9998 | 0.9698 |
| F1 | 0.9896 | **0.9845** |
| Overall accuracy | | **98.76%** |

Inference ≈ 30 sec / full Sentinel-2 tile (CPU).

### 2D-CNN binary

Held-out test patches (n = 7,200, 1,800 kelp / 5,400 not-kelp):

| Metric | Not Kelp | Kelp |
|---|---|---|
| Precision | 0.9932 | 0.9855 |
| Recall | 0.9952 | 0.9794 |
| F1 | 0.9942 | **0.9824** |
| Overall accuracy | | **99.12%** |

Inference ≈ 150 sec / full Sentinel-2 tile (CPU).

## Post-processing

A median filter can suppress isolated salt-and-pepper artefacts in the binary map. Not applied by default:

```python
from scipy.ndimage import median_filter
clean = median_filter(class_map, size=3)
```

Be careful: a 3×3 median filter can erase 1–2-pixel kelp slivers. For sparse canopy, prefer leaving the raw output and applying spatial filters downstream where you can tune size to your goal.
