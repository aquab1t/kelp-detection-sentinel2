"""
Predictor Module

Handles TFLite model inference for kelp detection.
Supports the binary 1D-CNN and 2D-CNN kelp detection TFLite models, with auto-detection
of output convention (sigmoid-baked vs raw logits) and optional 4-class fallback.

Author: KelpMap Project
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class KelpPredictor:
    """
    Run kelp detection inference using TFLite models.

    Primary output: binary kelp / not-kelp (model.is_binary == True).
    Also supports 4-class models (Cloud, Kelp, Land, Water) when supplied.
    Model output shape is auto-detected for class count.

    Attributes:
        model_path (Path): Path to TFLite model file
        interpreter: TFLite interpreter instance
        input_shape: Model input shape
        output_shape: Model output shape

    Example:
        >>> predictor = KelpPredictor('models/2dcnn_binary_int8.tflite')
        >>> classification = predictor.predict_and_classify(data)
    """

    CLASS_NAMES = {0: 'Cloud', 1: 'Kelp', 2: 'Land', 3: 'Water'}
    CLASS_COLORS = {
        0: '#95A5A6',
        1: '#2ECC71',
        2: '#E67E22',
        3: '#3498DB',
    }
    BINARY_CLASS_NAMES = {0: 'Not Kelp', 1: 'Kelp'}

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._load_model()

    def _load_model(self):
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=str(self.model_path))
        except ImportError:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_details = input_details[0]
        self.output_details = output_details[0]

        self.input_shape = tuple(self.input_details['shape'])
        self.input_dtype = self.input_details['dtype']
        self.output_shape = tuple(self.output_details['shape'])

        self.quantization_scale = self.input_details.get('quantization_parameters', {}).get('scale', 1.0)
        self.quantization_zero_point = self.input_details.get('quantization_parameters', {}).get('zero_point', 0)

    @property
    def is_binary(self) -> bool:
        """Check if model outputs binary (1 value) vs multi-class (4 values)."""
        return self.output_shape[-1] == 1

    def predict(self, data: np.ndarray, batch_size: int = 10000,
                show_progress: bool = True) -> np.ndarray:
        """
        Run model inference on data with batch processing.

        Args:
            data: Input data array
                - 1D-CNN: (n_samples, n_bands)
                - 2D-CNN: (n_samples, patch_h, patch_w, n_bands)
            batch_size: Number of samples per batch (default: 10000)
            show_progress: Show progress bar (default: True)

        Returns:
            numpy.ndarray: Raw model output
                - 4-class: (n_samples, 4) probabilities
                - Binary: (n_samples, 1) logits
        """
        n_samples = data.shape[0]
        n_classes = self.output_shape[-1]

        predictions = np.zeros((n_samples, n_classes), dtype=np.float32)

        n_batches = int(np.ceil(n_samples / batch_size))

        iterator = range(n_batches)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Running inference", unit="batch")

        current_batch_size = None
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_data = data[start_idx:end_idx]

            if self.input_dtype == np.int8:
                if self.quantization_scale != 1.0:
                    batch_data = (batch_data / self.quantization_scale +
                                  self.quantization_zero_point).astype(np.int8)
                else:
                    batch_data = batch_data.astype(np.int8)
            else:
                batch_data = batch_data.astype(self.input_dtype)

            if batch_data.shape[0] != current_batch_size:
                current_batch_size = batch_data.shape[0]
                self.interpreter.resize_tensor_input(
                    self.input_details['index'], batch_data.shape
                )
                self.interpreter.allocate_tensors()

            self.interpreter.set_tensor(self.input_details['index'], batch_data)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details['index'])
            predictions[start_idx:end_idx] = output

        return predictions

    def predict_binary(self, data: np.ndarray, batch_size: int = 10000,
                       show_progress: bool = True) -> np.ndarray:
        """
        Run inference and return sigmoid probabilities for binary model.

        Args:
            data: Input data array
            batch_size: Number of samples per batch
            show_progress: Show progress bar

        Returns:
            numpy.ndarray: Kelp probabilities of shape (n_samples,) in range [0, 1]
        """
        raw = self.predict(data, batch_size, show_progress)
        raw_squeezed = raw.squeeze(-1).astype(np.float32)
        # Auto-detect: if outputs already in [0,1], the model has sigmoid baked
        # into its graph; otherwise they are logits and need sigmoid applied.
        if raw_squeezed.size and (raw_squeezed.min() >= 0.0) and (raw_squeezed.max() <= 1.0):
            return raw_squeezed
        try:
            from scipy.special import expit
            return expit(raw_squeezed).astype(np.float32)
        except ImportError:
            raw_clipped = np.clip(raw_squeezed, -500, 500)
            return (1.0 / (1.0 + np.exp(-raw_clipped))).astype(np.float32)

    def classify(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to class labels.

        Args:
            probabilities: Array of shape (n_samples, n_classes)

        Returns:
            numpy.ndarray: Class labels of shape (n_samples,)
        """
        return np.argmax(probabilities, axis=1).astype(np.uint8)

    def classify_binary(self, probabilities: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
        """
        Convert binary probabilities to class labels.

        Args:
            probabilities: 1D array of kelp probabilities
            threshold: Kelp detection threshold (default: 0.5)

        Returns:
            numpy.ndarray: Binary class labels (0=Not Kelp, 1=Kelp)
        """
        return (probabilities >= threshold).astype(np.uint8)

    def predict_and_classify(self, data: np.ndarray,
                             batch_size: int = 10000,
                             threshold: float = 0.5) -> np.ndarray:
        """
        Convenience method: predict probabilities and classify in one step.

        For 4-class models: argmax over class probabilities (threshold ignored).
        For binary models: sigmoid + threshold (default 0.5).

        Args:
            data: Input data array
            batch_size: Batch size for inference
            threshold: Decision threshold for binary models

        Returns:
            numpy.ndarray: Class labels
        """
        if self.is_binary:
            probabilities = self.predict_binary(data, batch_size)
            return self.classify_binary(probabilities, threshold=threshold)
        else:
            probabilities = self.predict(data, batch_size)
            return self.classify(probabilities)

    def reshape_to_2d(self, classification: np.ndarray,
                      height: int, width: int) -> np.ndarray:
        """
        Reshape 1D classification array to 2D map.

        Args:
            classification: 1D array of shape (n_pixels,)
            height: Height of the output map
            width: Width of the output map

        Returns:
            numpy.ndarray: 2D classification map of shape (height, width)
        """
        return classification.reshape(height, width)

    def export_geotiff(self, classification: np.ndarray, metadata: Dict,
                       output_path: str, nodata=255, dtype='uint8'):
        """
        Export classification or probability map as GeoTIFF.

        Args:
            classification: 2D array (height, width)
            metadata: Metadata dictionary from Sentinel2Loader
            output_path: Output file path
            nodata: NoData value (default: 255 for uint8, NaN for float32)
            dtype: Output data type ('uint8' or 'float32')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if dtype == 'float32':
            nodata = np.nan if nodata == 255 else nodata

        profile = {
            'driver': 'GTiff',
            'dtype': dtype,
            'width': metadata['width'],
            'height': metadata['height'],
            'count': 1,
            'crs': metadata['crs'],
            'transform': metadata['transform'],
            'nodata': nodata,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classification.astype(dtype), 1)

        print(f"Saved: {output_path}")

    def get_class_statistics(self, classification: np.ndarray,
                             pixel_size_m: float = 10.0) -> Dict:
        """
        Compute statistics for each class in the classification map.

        Args:
            classification: 1D or 2D classification array
            pixel_size_m: Pixel size in meters (default: 10m)

        Returns:
            dict: Statistics for each class
        """
        if classification.ndim == 2:
            classification = classification.flatten()

        unique, counts = np.unique(classification, return_counts=True)
        total = classification.size

        class_names = self.BINARY_CLASS_NAMES if self.is_binary else self.CLASS_NAMES

        stats = {}
        for class_id, count in zip(unique, counts):
            if class_id == 255 or (isinstance(class_id, float) and np.isnan(class_id)):
                continue

            class_name = class_names.get(class_id, f'Class_{class_id}')
            area_km2 = count * pixel_size_m ** 2 / 1e6

            stats[class_id] = {
                'name': class_name,
                'pixels': count,
                'percentage': 100 * count / total,
                'area_km2': area_km2
            }

        return stats
