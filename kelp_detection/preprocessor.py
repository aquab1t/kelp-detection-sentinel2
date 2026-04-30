"""
Preprocessor Module

Handles data preprocessing for kelp detection inference:
- Band resampling (20m → 10m)
- Normalization using StandardScaler
- Patch extraction for 2D-CNN

The preprocessing pipeline transforms raw Sentinel-2 DN values
into normalized patches suitable for CNN inference.

Author: KelpMap Project
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Preprocess Sentinel-2 data for kelp detection inference.

    This class handles:
    1. Band resampling from 20m to 10m resolution
    2. StandardScaler normalization
    3. Patch extraction for 2D-CNN models

    Attributes:
        model_type (str): '1dcnn' or '2dcnn'
        scaler: StandardScaler instance for normalization

    Example:
        >>> preprocessor = Preprocessor(model_type='2dcnn')
        >>> data_resampled = preprocessor.resample_bands(data)
        >>> data_norm = preprocessor.normalize(data_resampled)
    """

    BANDS_10M_INDICES = [0, 1, 2, 6]  # B02, B03, B04, B08
    BANDS_20M_INDICES = [3, 4, 5, 7, 8]  # B05, B06, B07, B8A, B11

    def __init__(self, model_type: str = '2dcnn', scaler_path: Optional[str] = None):
        """
        Initialize preprocessor.

        Args:
            model_type: '1dcnn' or '2dcnn' (default: '2dcnn')
                Both 2dcnn models (4-class and binary) use '2dcnn' since they
                share the same preprocessing (11x11 patch extraction).
            scaler_path: Path to saved scaler file. If None, uses default.

        Raises:
            ValueError: If model_type is not '1dcnn' or '2dcnn'
        """
        if model_type not in ['1dcnn', '2dcnn']:
            raise ValueError(f"model_type must be '1dcnn' or '2dcnn', got {model_type}")

        self.model_type = model_type
        self.scaler = None

        if scaler_path is None:
            scaler_path = self._get_default_scaler_path()

        self.scaler_path = Path(scaler_path)

        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)

    def _get_default_scaler_path(self) -> Path:
        """
        Get default scaler path based on model type.

        Returns:
            Path: Path to the default scaler file
        """
        package_dir = Path(__file__).parent.parent
        if self.model_type == '1dcnn':
            return package_dir / 'models' / 'scaler_1dcnn.joblib'
        else:
            return package_dir / 'models' / 'scaler_2dcnn.joblib'

    def resample_bands(self, data: np.ndarray, target_resolution: int = 10) -> np.ndarray:
        """
        Resample 20m bands to target resolution using bilinear interpolation.

        Sentinel-2 bands have different resolutions:
        - 10m: B02, B03, B04, B08 (already at target)
        - 20m: B05, B06, B07, B8A, B11 (need upsampling)

        Args:
            data: Array of shape (n_bands, height, width)
            target_resolution: Target resolution in meters (default: 10)

        Returns:
            numpy.ndarray: Resampled array with all bands at target resolution

        Example:
            >>> data = loader.load_bands()  # Mixed resolutions
            >>> data_10m = preprocessor.resample_bands(data)
            >>> print(data_10m.shape)  # (9, height, width) at 10m
        """
        n_bands, height, width = data.shape

        if n_bands != 9:
            raise ValueError(f"Expected 9 bands, got {n_bands}")

        result = data.copy()

        for idx in self.BANDS_20M_INDICES:
            band_20m = data[idx]

            if band_20m.shape != (height, width):
                band_10m = self._upsample_2x(band_20m)
                result[idx] = band_10m

        return result

    def _upsample_2x(self, data: np.ndarray) -> np.ndarray:
        """
        Upsample array by 2x using bilinear interpolation.

        Args:
            data: 2D array to upsample

        Returns:
            numpy.ndarray: Upsampled array (2x height, 2x width)
        """
        from scipy.ndimage import zoom

        return zoom(data, 2, order=1)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using StandardScaler.

        Normalization is applied per-band using saved scaler parameters.
        If no scaler is available, uses default normalization (divide by 10000).

        Args:
            data: Array of shape (n_bands, height, width) or (n_samples, n_bands)

        Returns:
            numpy.ndarray: Normalized array

        Example:
            >>> data_norm = preprocessor.normalize(data)
            >>> print(f"Mean: {data_norm.mean():.2f}, Std: {data_norm.std():.2f}")
        """
        if self.scaler is not None:
            scaler_expects_reflectance = float(np.asarray(self.scaler.mean_).flatten()[0]) < 1.5
            if scaler_expects_reflectance and data.max() > 1.5:
                data = data * 0.0001

            if data.ndim == 3:
                n_bands, height, width = data.shape
                data_2d = data.transpose(1, 2, 0).reshape(-1, n_bands)
                data_norm = self.scaler.transform(data_2d)
                return data_norm.reshape(height, width, n_bands).transpose(2, 0, 1)
            else:
                return self.scaler.transform(data)
        else:
            return data / 10000.0 if data.max() > 1.5 else data

    def extract_patches(self, data: np.ndarray, patch_size: int = 11,
                        roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract patches for 2D-CNN inference.

        For 2D-CNN, we need spatial context (neighborhood) around each pixel.
        This extracts overlapping patches of size (patch_size, patch_size, n_bands).

        If `roi_mask` is provided, only patches centred on True pixels are
        materialised - critical for full-tile inference where extracting all
        ~120M patches at 11x11x9 float32 would require ~489 GiB of memory.

        Args:
            data: Normalized array of shape (n_bands, height, width)
            patch_size: Size of patches to extract (default: 11)
            roi_mask: Optional bool array of shape (height, width). When given,
                returns only the patches centred on True pixels.

        Returns:
            numpy.ndarray: Array of shape (n_pixels, patch_size, patch_size, n_bands)
                where n_pixels = height*width if roi_mask is None,
                else int(roi_mask.sum()).
        """
        n_bands, height, width = data.shape
        half_size = patch_size // 2

        data_padded = np.pad(data, ((0, 0), (half_size, half_size), (half_size, half_size)),
                             mode='reflect')
        data_hwc = data_padded.transpose(1, 2, 0)

        windows = np.lib.stride_tricks.sliding_window_view(
            data_hwc, (patch_size, patch_size, n_bands)
        ).squeeze(axis=2)

        if roi_mask is not None:
            ys, xs = np.where(roi_mask)
            patches = windows[ys, xs].copy()
        else:
            patches = windows.reshape(-1, patch_size, patch_size, n_bands)

        return np.ascontiguousarray(patches, dtype=np.float32)

    def prepare_for_inference(self, data: np.ndarray,
                              roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full preprocessing pipeline for inference.

        Applies resampling, normalization, and (for 2D-CNN) patch extraction.

        Args:
            data: Raw data array of shape (n_bands, height, width)

        Returns:
            numpy.ndarray: Preprocessed data ready for model inference
                - 1D-CNN: (n_pixels, n_bands)
                - 2D-CNN: (n_pixels, patch_size, patch_size, n_bands)

        Example:
            >>> data = loader.load_bands()
            >>> data_ready = preprocessor.prepare_for_inference(data)
            >>> predictions = predictor.predict(data_ready)
        """
        data_resampled = self.resample_bands(data)
        data_norm = self.normalize(data_resampled)

        if self.model_type == '2dcnn':
            return self.extract_patches(data_norm, roi_mask=roi_mask)
        else:
            n_bands, height, width = data_norm.shape
            return data_norm.transpose(1, 2, 0).reshape(-1, n_bands)
