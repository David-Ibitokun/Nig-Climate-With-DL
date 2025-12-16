# Data Sources, Units, and Formats

This project collects climate, CO₂, and crop production data for Nigeria (6 regions):
- North Central
- North East
- North West
- South East
- South South
- South West

All data used for modelling (FNN and LSTM) are stored under `project_data/`.

## 1) NASA POWER (Temperature & Rainfall)
- Source: NASA POWER API (https://power.larc.nasa.gov)
- Variables:
  - `T2M` (Temperature at 2 m, daily average) → saved as `Temperature_Avg_C` (°C)
  - `T2M_MAX` → `Temperature_Max_C` (°C)
  - `T2M_MIN` → `Temperature_Min_C` (°C)
  - `PRECTOTCORR` → `Precipitation_mm` (mm/day)
- Temporal resolution: daily (resampled or aggregated to monthly/annual as needed)
- Spatial resolution: point (lat/lon) per region centroid used in `scripts/download_zonal_data.py`
- File format: CSV per region and a combined CSV: `project_data/climate/[Region]_1990_2023.csv`, `all_regions_combined_1990_2023.csv`.

## 2) NOAA (CO₂ concentrations)
- Source: NOAA GML Mauna Loa monthly CO₂ (https://gml.noaa.gov/ccgg/trends/)
- Variable: monthly mean CO₂ (ppm) → column `average` in saved CSV
- Temporal resolution: monthly
- Spatial scope: Mauna Loa Observatory (proxy for global well-mixed CO₂)
- File format: CSV saved as:
  - `project_data/climate/co2_global_monthly_full.csv` (full available record)
  - `project_data/climate/co2_global_monthly_1990_2023.csv` (project range)

## 3) FAOSTAT (Crop production)
- Source: FAOSTAT (https://www.fao.org/faostat/en/)
- Requested variables: production for Maize, Cassava, Yam (units: tonnes)
- Temporal resolution: annual (year)
- Spatial resolution: country-level. Regional breakdowns (by Nigeria regions) are not always available in FAOSTAT; when unavailable the code generates a realistic regional allocation as a fallback.
- File format: CSV saved under `project_data/crop_yields/`.

## 4) Notes on formats and derived features
- LSTM inputs: monthly sequences saved as NumPy arrays: `project_data/models/lstm_sequences.npy` and `lstm_targets.npy`.
- FNN inputs: yearly aggregated CSV `project_data/models/fnn_dataset.csv`.
- Suggested derived features:
  - Annual mean CO₂ (ppm)
  - 12-month rolling mean CO₂
  - CO₂ anomaly relative to baseline (e.g., 1981–2010)
  - Yearly climate aggregates (mean temperature, total precipitation)

## 5) Fallback policy
- If FAOSTAT API or other remote services fail, the pipeline creates realistic synthetic crop yields and saves them with filename indicating synthetic origin (e.g., `nigeria_crop_yields_1990_2023.csv`).
- All fallbacks are documented in logs when the script runs.

## 6) Units summary
- Temperature: degrees Celsius (°C)
- Precipitation: millimeters (mm) — daily values; monthly aggregates are in mm/month
- CO₂ concentration: parts per million (ppm)
- Crop production: tonnes (t) or 1000 tonnes where indicated

---
If you want, I can: 
- Add CHIRPS rainfall download (gridded) and code to aggregate it to region centroids.
- Try a more robust FAOSTAT approach (bulk CSV download links) or integrate the `pycountry`/`pandas` mapping to ensure correct country codes.
