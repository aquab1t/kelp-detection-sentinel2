"""
Data Loader Module

Handles loading Sentinel-2 L1C data from .SAFE directories.
Extracts the 9 MSI bands used for kelp detection.

Bands:
- B2 (490nm): Blue, 10m
- B3 (560nm): Green, 10m
- B4 (665nm): Red, 10m
- B5 (705nm): Red Edge 1, 20m
- B6 (740nm): Red Edge 2, 20m
- B7 (783nm): Red Edge 3, 20m
- B8 (842nm): NIR, 10m
- B8A (865nm): Narrow NIR, 20m
- B11 (1610nm): SWIR 1, 20m

Author: KelpMap Project
"""

import os
import re
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class Sentinel2Loader:
    """
    Load Sentinel-2 L1C data from .SAFE directories.

    This class handles the extraction of the 9 MSI bands required
    for kelp detection from Sentinel-2 L1C products.

    Attributes:
        safe_path (Path): Path to the .SAFE directory
        granule_path (Path): Path to the GRANULE subdirectory
        metadata (dict): Scene metadata including CRS, transform, bounds

    Example:
        >>> loader = Sentinel2Loader('S2A_MSIL1C_20200101.SAFE')
        >>> data = loader.load_bands()
        >>> metadata = loader.get_metadata()
    """

    BANDS_10M = ['B02', 'B03', 'B04', 'B08']
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11']
    BAND_RESOLUTIONS = {
        'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
        'B05': 20, 'B06': 20, 'B07': 20, 'B8A': 20, 'B11': 20
    }
    BAND_ORDER = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11']

    def __init__(self, safe_path: str):
        """
        Initialize loader with path to .SAFE directory.

        Args:
            safe_path: Path to Sentinel-2 .SAFE directory

        Raises:
            FileNotFoundError: If .SAFE directory doesn't exist
            ValueError: If directory structure is invalid
        """
        self.safe_path = Path(safe_path)

        if not self.safe_path.exists():
            raise FileNotFoundError(f"SAFE directory not found: {safe_path}")

        self.granule_path = self._find_granule_path()
        self._metadata = None
        self._data = None

    def _find_granule_path(self) -> Path:
        """
        Find the GRANULE subdirectory within .SAFE structure.

        Returns:
            Path to GRANULE directory

        Raises:
            ValueError: If GRANULE directory not found
        """
        granule_dir = self.safe_path / 'GRANULE'

        if not granule_dir.exists():
            raise ValueError(f"GRANULE directory not found in {self.safe_path}")

        granule_subdirs = list(granule_dir.glob('L1C_*'))
        if not granule_subdirs:
            granule_subdirs = list(granule_dir.glob('L2A_*'))
        if not granule_subdirs:
            granule_subdirs = list(granule_dir.glob('S2*_MSIL1C_*'))
        if not granule_subdirs:
            granule_subdirs = list(granule_dir.glob('S2*_MSIL2A_*'))
        if not granule_subdirs:
            granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]

        if not granule_subdirs:
            raise ValueError(f"No granule subdirectory found in {granule_dir}")

        return granule_subdirs[0]

    def _find_band_path(self, band_name: str) -> Path:
        """
        Find the file path for a specific band.

        Args:
            band_name: Band name (e.g., 'B02', 'B03')

        Returns:
            Path to the band file

        Raises:
            FileNotFoundError: If band file not found
        """
        img_dir = self.granule_path / 'IMG_DATA'

        if not img_dir.exists():
            raise FileNotFoundError(f"IMG_DATA directory not found: {img_dir}")

        matches = list(img_dir.rglob(f'*_{band_name}.jp2'))
        if not matches:
            matches = list(img_dir.rglob(f'*_{band_name}_*.jp2'))
        if not matches:
            matches = [p for p in img_dir.rglob('*.jp2')
                       if p.stem.endswith(f'_{band_name}')
                       or f'_{band_name}_' in p.stem]

        if not matches:
            raise FileNotFoundError(f"Band {band_name} not found in {img_dir}")

        return matches[0]

    def load_bands(self, bands: Optional[list] = None) -> np.ndarray:
        """
        Load specified bands from the Sentinel-2 scene.

        Loads bands as a 3D numpy array with shape (n_bands, height, width).
        The default bands are ordered for kelp detection: B02, B03, B04,
        B05, B06, B07, B08, B8A, B11.

        Args:
            bands: List of band names to load. If None, loads all 9 bands.

        Returns:
            numpy.ndarray: Array of shape (n_bands, height, width)
            Values are digital numbers (DN) in range 0-10000

        Raises:
            FileNotFoundError: If band files not found
            rasterio.errors.RasterioIOError: If band files cannot be read

        Example:
            >>> loader = Sentinel2Loader('S2A_MSIL1C_20200101.SAFE')
            >>> data = loader.load_bands()
            >>> print(data.shape)  # (9, height, width)
        """
        if bands is None:
            bands = self.BAND_ORDER

        from rasterio.enums import Resampling

        target_shape = None
        for band_name in bands:
            if self.BAND_RESOLUTIONS.get(band_name) == 10:
                with rasterio.open(self._find_band_path(band_name)) as src:
                    target_shape = (src.height, src.width)
                    self._metadata = {
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'width': src.width,
                        'height': src.height,
                        'resolution': src.res,
                    }
                break

        if target_shape is None:
            with rasterio.open(self._find_band_path(bands[0])) as src:
                target_shape = (src.height * 2, src.width * 2)

        band_arrays = []
        for band_name in bands:
            band_path = self._find_band_path(band_name)
            with rasterio.open(band_path) as src:
                if (src.height, src.width) == target_shape:
                    band_data = src.read(1).astype(np.float32)
                else:
                    band_data = src.read(
                        1,
                        out_shape=target_shape,
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)
                band_arrays.append(band_data)

        self._data = np.stack(band_arrays, axis=0)
        return self._data

    def get_metadata(self) -> Dict:
        """
        Get scene metadata including CRS, transform, and bounds.

        Returns:
            dict: Metadata dictionary with keys:
                - crs: Coordinate reference system
                - transform: Affine transform
                - bounds: Bounding box (left, bottom, right, top)
                - width: Width in pixels
                - height: Height in pixels
                - resolution: Pixel resolution in meters

        Raises:
            RuntimeError: If load_bands() not called first
        """
        if self._metadata is None:
            raise RuntimeError("Metadata not available. Call load_bands() first.")

        return self._metadata.copy()

    def get_scene_id(self) -> str:
        """
        Extract scene ID from the .SAFE directory name.

        Returns:
            str: Scene ID (e.g., 'S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM')
        """
        return self.safe_path.name.replace('.SAFE', '')

    def get_sensing_date(self) -> str:
        """
        Extract sensing date from scene ID.

        Returns:
            str: Sensing date in YYYYMMDD format
        """
        scene_id = self.get_scene_id()
        match = re.search(r'_(\d{8})T\d{6}_', scene_id)

        if match:
            return match.group(1)

        return None
