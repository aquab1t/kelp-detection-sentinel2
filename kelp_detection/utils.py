"""
Utility Functions

Helper functions for kelp detection pipeline.

Author: KelpMap Project
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import re


def parse_scene_name(scene_name: str) -> dict:
    """
    Parse Sentinel-2 scene name to extract metadata.
    
    Args:
        scene_name: Scene name (e.g., 'S2A_MSIL1C_20200101T180941_N0200_R084_T11RPM')
    
    Returns:
        dict: Parsed metadata with keys: satellite, product_type, datetime, 
              processing_baseline, relative_orbit, tile_id
    """
    pattern = r'(S2[AB])_MSIL1C_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T([A-Z0-9]{5})'
    match = re.match(pattern, scene_name)
    
    if match:
        return {
            'satellite': match.group(1),
            'datetime': match.group(2),
            'processing_baseline': match.group(3),
            'relative_orbit': match.group(4),
            'tile_id': match.group(5)
        }
    
    return {}


def get_band_wavelengths() -> List[float]:
    """
    Get central wavelengths for Sentinel-2 MSI bands used in kelp detection.
    
    Returns:
        list: Wavelengths in nm for bands B02-B11
    """
    return [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7]


def get_band_names() -> List[str]:
    """
    Get band names for Sentinel-2 MSI bands used in kelp detection.
    
    Returns:
        list: Band names (B02-B11)
    """
    return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11']


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Water Index (NDWI).
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        green: Green band (B03)
        nir: NIR band (B08)
    
    Returns:
        numpy.ndarray: NDWI values (-1 to 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
    
    return ndwi


def compute_ndre(red_edge: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Red Edge (NDRE).
    
    NDRE = (Red Edge - Red) / (Red Edge + Red)
    
    Args:
        red_edge: Red Edge band (B05)
        red: Red band (B04)
    
    Returns:
        numpy.ndarray: NDRE values (-1 to 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ndre = (red_edge - red) / (red_edge + red)
        ndre = np.nan_to_num(ndre, nan=0.0)
    
    return ndre


def calculate_kelp_area(classification: np.ndarray, pixel_size_m: float = 10.0) -> dict:
    """
    Calculate kelp detection area statistics.
    
    Args:
        classification: Classification map (class IDs)
        pixel_size_m: Pixel size in meters (default: 10m)
    
    Returns:
        dict: Area statistics including pixel count and km²
    """
    kelp_pixels = np.sum(classification == 0)
    pixel_area_m2 = pixel_size_m ** 2
    kelp_area_km2 = kelp_pixels * pixel_area_m2 / 1e6
    
    return {
        'kelp_pixels': kelp_pixels,
        'kelp_area_km2': kelp_area_km2,
        'pixel_size_m': pixel_size_m
    }


def format_filesize(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        str: Formatted file size (e.g., '1.5 GB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
