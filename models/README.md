# Kelp Detection Models

INT8-quantized binary kelp detection models for Sentinel-2 imagery.

| Model | Size | Input | Output | Best for |
|-------|------|-------|--------|----------|
| `1dcnn_binary_int8.tflite` | 54 KB | `(batch, 9)` | sigmoid prob `(batch, 1)` | Fast per-pixel screening |
| `2dcnn_binary_int8.tflite` | 466 KB | `(batch, 11, 11, 9)` (NHWC) | logit `(batch, 1)` | Higher-accuracy patch detection |

> **Output convention** - note the asymmetry:
> - `1dcnn_binary_int8.tflite` has the final **sigmoid baked into the graph**, so its raw output is already a probability in `[0, 1]`.
> - `2dcnn_binary_int8.tflite` outputs **raw logits** (typical range −50 to +30); apply sigmoid (`scipy.special.expit`) to get a probability.
>
> `KelpPredictor.predict_binary()` auto-detects the convention by output range and handles both cases - you only need to think about this if you're calling the TFLite interpreter directly.

Threshold the probability at `0.5` (or your chosen value) to get a binary kelp mask.

---

## 1D-CNN Binary

Fast pixel-wise spectral classifier.

| Field | Value |
|---|---|
| Architecture | SeparableConv1D + Squeeze-and-Excitation blocks + GlobalMaxPooling1D |
| Parameters | ~30 K |
| Framework | TensorFlow / Keras → INT8 TFLite |
| Loss | Binary cross-entropy with class weights |
| Augmentation | Gaussian noise (σ = 0.001) |
| Scaler | `scaler_1dcnn_binary.joblib` (StandardScaler on training data) |

### Test performance

Held-out 15% pixel-wise split (`SEED=42`, n = 10,227: 4,168 kelp, 6,059 not-kelp):

| Metric | Not Kelp | Kelp |
|---|---|---|
| Precision | 0.9796 | 0.9998 |
| Recall | 0.9998 | 0.9698 |
| F1 | 0.9896 | **0.9845** |
| **Overall accuracy** | | **98.76%** |

Confusion matrix:

```
                  Predicted
                  Not-Kelp   Kelp
Actual Not-Kelp     6058       1
       Kelp          126    4042
```

> **Caveat - pixel-wise random split.** The 1D-CNN training data is split randomly across pixels, not grouped by scene. Neighbouring pixels in the same scene have nearly identical spectra (spatial autocorrelation), which inflates this score relative to performance on truly novel scenes. Treat 98.76% as an upper bound; expect 95% or lower on unseen sites.

---

## 2D-CNN Binary

Patch-based spatial-spectral classifier examining an 11×11 neighbourhood.

| Field | Value |
|---|---|
| Architecture | Conv2D + BatchNorm + MaxPool blocks + AdaptiveAvgPool + FC head |
| Parameters | ~112 K |
| Framework | PyTorch → ONNX → INT8 TFLite (via onnx2tf) |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW, lr = 3e-4, CosineAnnealing |
| Batch size | 512 |
| Scaler | `scaler_2dcnn_binary.joblib` (StandardScaler on DN-scale training data) |

Architecture detail: 9 → 32 → 64 → 128 channels (Conv2D + BN + ReLU + MaxPool); AdaptiveAvgPool to 1×1; FC head 128 → 128 → 1 with Dropout(0.5).

### Test performance

Held-out 4-class test patches mapped to binary (`patches_test_2dcnn.npz`, n = 7,200: 1,800 kelp, 5,400 not-kelp):

| Metric | Not Kelp | Kelp |
|---|---|---|
| Precision | 0.9932 | 0.9855 |
| Recall | 0.9952 | 0.9794 |
| F1 | 0.9942 | **0.9824** |
| **Overall accuracy** | | **99.12%** |

Confusion matrix:

```
                  Predicted
                  Not-Kelp   Kelp
Actual Not-Kelp     5374      26
       Kelp           37    1763
```

> The 2D-CNN test patches were stratified at the 4-class dataset stage and held out before training. This is closer to a true held-out evaluation than the 1D-CNN's pixel-wise split, but still drawn from the same Cedros Island scene set used in training - generalisation to other sites and seasons is not measured here.

---

## Usage

The recommended path is the inference scripts and the `KelpPredictor` class - see the project root `README.md`. If you must call the TFLite interpreter directly:

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from scipy.special import expit

# 1D-CNN - output is already a sigmoid probability
it = Interpreter(model_path="models/1dcnn_binary_int8.tflite")
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
x = scaled_band_vector.astype(np.float32).reshape(1, 9)  # 9 StandardScaler-normalised bands
it.set_tensor(inp["index"], x)
it.invoke()
kelp_prob = it.get_tensor(out["index"])[0, 0]  # already in [0, 1]

# 2D-CNN - output is a logit; apply sigmoid
it2 = Interpreter(model_path="models/2dcnn_binary_int8.tflite")
it2.allocate_tensors()
inp2, out2 = it2.get_input_details()[0], it2.get_output_details()[0]
patch = scaled_patch.astype(np.float32).reshape(1, 11, 11, 9)  # NHWC
it2.set_tensor(inp2["index"], patch)
it2.invoke()
kelp_prob = float(expit(it2.get_tensor(out2["index"])[0, 0]))
```

---

## Quantization details

Both files use TFLite weight quantization (INT8 weights, float32 I/O). The 1D-CNN was converted from a Keras model with `tf.lite.Optimize.DEFAULT`; the 2D-CNN was converted from PyTorch via ONNX using `onnx2tf` with the same optimization. Activation values stay in float32 at runtime, so quantization-induced accuracy loss is negligible.
