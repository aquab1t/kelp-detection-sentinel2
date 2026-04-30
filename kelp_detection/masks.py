"""
Masks Module

Handles land and coastal masking using ESA WorldCover data.
Filters land pixels to focus inference on coastal kelp habitats.

ESA WorldCover 2021 v200:
- Class 80: Permanent water bodies
- All other classes: Land

Author: KelpMap Project
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class LandMask:
    """
    Apply land masking using ESA WorldCover data.
    
    This class handles:
    - Loading ESA WorldCover land cover data
    - Reprojecting to match Sentinel-2 scene
    - Creating binary water/land mask
    
    Attributes:
        worldcover_path (Path): Path to ESA WorldCover GeoTIFF
    
    Example:
        >>> mask = LandMask('ESA_WorldCover_10m_2021_v200.tif')
        >>> water_mask = mask.get_water_mask(scene_bounds, scene_transform, scene_crs)
    """
    
    WATER_CLASS = 80  # ESA WorldCover permanent water bodies
    
    def __init__(self, worldcover_path: str):
        """
        Initialize land mask with ESA WorldCover data.
        
        Args:
            worldcover_path: Path to ESA WorldCover GeoTIFF
        
        Raises:
            FileNotFoundError: If WorldCover file doesn't exist
        """
        self.worldcover_path = Path(worldcover_path)
        
        if not self.worldcover_path.exists():
            raise FileNotFoundError(f"WorldCover file not found: {worldcover_path}")
    
    def get_water_mask(self, target_shape: Tuple[int, int], 
                       target_transform, target_crs) -> np.ndarray:
        """
        Get water mask reprojected to target scene.
        
        Reprojects ESA WorldCover to match the target scene dimensions
        and CRS, then creates a binary mask (True = water).
        
        Args:
            target_shape: (height, width) of target scene
            target_transform: Affine transform of target scene
            target_crs: CRS of target scene
        
        Returns:
            numpy.ndarray: Boolean mask (True = water, False = land)
        
        Example:
            >>> mask = LandMask('ESA_WorldCover.tif')
            >>> water = mask.get_water_mask((10980, 10980), transform, crs)
            >>> print(f"Water pixels: {water.sum():,}")
        """
        height, width = target_shape

        with rasterio.open(self.worldcover_path) as src:
            land_cover = np.full((height, width), self.WATER_CLASS, dtype=np.uint8)

            reproject(
                source=rasterio.band(src, 1),
                destination=land_cover,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                init_dest_nodata=False,
            )

        water_mask = land_cover == self.WATER_CLASS

        return water_mask
    
    def get_land_mask(self, target_shape: Tuple[int, int],
                      target_transform, target_crs) -> np.ndarray:
        """
        Get land mask (inverse of water mask).
        
        Args:
            target_shape: (height, width) of target scene
            target_transform: Affine transform of target scene
            target_crs: CRS of target scene
        
        Returns:
            numpy.ndarray: Boolean mask (True = land, False = water)
        """
        water_mask = self.get_water_mask(target_shape, target_transform, target_crs)
        return ~water_mask
    
    def apply_mask(self, data: np.ndarray, mask: np.ndarray, 
                   fill_value: float = np.nan) -> np.ndarray:
        """
        Apply mask to data array.
        
        Args:
            data: Data array of shape (height, width) or (n_bands, height, width)
            mask: Boolean mask (True = keep, False = mask out)
            fill_value: Value to use for masked pixels (default: np.nan)
        
        Returns:
            numpy.ndarray: Masked data array
        """
        if data.ndim == 3:
            mask = np.broadcast_to(mask, data.shape)
        
        return np.where(mask, data, fill_value)


def create_coastal_buffer(water_mask: np.ndarray, buffer_distance: int = 100) -> np.ndarray:
    """
    Create a coastal buffer zone around the shoreline.
    
    Useful for focusing inference on nearshore kelp habitats.
    
    Args:
        water_mask: Boolean water mask (True = water)
        buffer_distance: Buffer distance in pixels (default: 100)
    
    Returns:
        numpy.ndarray: Boolean mask of coastal buffer zone
    
    Example:
        >>> coastal_zone = create_coastal_buffer(water_mask, buffer_distance=50)
        >>> print(f"Coastal pixels: {coastal_zone.sum():,}")
    """
    from scipy.ndimage import distance_transform_edt
    
    land_mask = ~water_mask
    
    distance_from_land = distance_transform_edt(water_mask)
    
    coastal_buffer = (distance_from_land > 0) & (distance_from_land <= buffer_distance)
    
    return coastal_buffer
