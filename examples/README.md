# Examples

End-to-end reproducible example bundled with the repo. A 1024×1024 (≈10 km × 10 km) crop of `S2A_MSIL1C_20180801T181011_T11RPM` around a kelp bed off Cedros Island, Baja California - small enough to ship in git so anyone can run the pipeline without downloading a Sentinel-2 SAFE folder.

## Files

| File | Size | What it is |
|---|---|---|
| `cedros_2018-08-01_bands.tif` | 12 MB | Source 9-band uint16 GeoTIFF in the order `B02, B03, B04, B08, B05, B06, B07, B8A, B11` (10 m grid, EPSG:32611) |
| `cedros_worldcover.tif` | 15 KB | ESA WorldCover 2021 v200, reprojected to the scene grid |
| `cedros_roi.tif` | 28 KB | 2 km coastal-buffer ROI computed from WorldCover (precomputed reference) |
| `cedros_kelp_1dcnn.tif` | 10 KB | 1D-CNN binary kelp map (precomputed reference) |
| `cedros_kelp_2dcnn.tif` | 10 KB | 2D-CNN binary kelp map (precomputed reference) |
| `run_example.py` | – | Runner script: builds the ROI from WorldCover and runs both models against `bands.tif` |

## Usage

From the repo root:

```bash
python examples/run_example.py
```

Expected output:

```
Loaded bands: shape=(9, 1024, 1024), dtype=float32, range=[1098, 5276]
ROI: 356,187 pixels in 2 km coastal buffer
Wrote cedros_roi.tif
1D-CNN: 14,774 kelp px = 1.48 km²
2D-CNN: 21,646 kelp px = 2.16 km²
Wrote cedros_kelp_1dcnn.tif and cedros_kelp_2dcnn.tif
```

The newly written GeoTIFFs should match the precomputed reference files (`cedros_kelp_{1,2}dcnn.tif`).

## Visualising in QGIS

1. **Load the source bands as RGB**: drag `cedros_2018-08-01_bands.tif` into QGIS, set band rendering to "Multiband color" with R = band 3 (B04), G = band 2 (B03), B = band 1 (B02), and apply a 2 % / 98 % cumulative cut stretch.
2. **Stack the kelp mask**: load `cedros_kelp_2dcnn.tif`, set "Paletted/Unique values" symbology, assign green to value `1` and "no symbol" to `0`. Place it on top of the RGB layer.
3. **(Optional)** stack `cedros_roi.tif` as a transparent overlay to visualise where the model was actually run.

## Output format

Both kelp GeoTIFFs are `uint8` with `1 = kelp` and `0 = everything else`, no NoData. Each pixel is 10 m × 10 m, so kelp area in km² = pixel count ÷ 10,000.
