"""kelp-detection-sentinel2: Detect giant kelp from Sentinel-2 imagery."""

from .data_loader import Sentinel2Loader
from .preprocessor import Preprocessor
from .predictor import KelpPredictor
from .masks import LandMask, create_coastal_buffer
from .utils import (
    parse_scene_name,
    get_band_wavelengths,
    get_band_names,
    compute_ndwi,
    compute_ndre,
    calculate_kelp_area,
)

__version__ = "2.0.0"

__all__ = [
    "Sentinel2Loader",
    "Preprocessor",
    "KelpPredictor",
    "LandMask",
    "create_coastal_buffer",
    "parse_scene_name",
    "get_band_wavelengths",
    "get_band_names",
    "compute_ndwi",
    "compute_ndre",
    "calculate_kelp_area",
]
