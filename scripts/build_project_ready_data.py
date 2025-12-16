"""
Script: build_project_ready_data.py
Creates ProjectReadyData CSVs required for hybrid LSTM–FNN modeling.
Outputs (saved under ProjectReadyData/):
 - regional_monthly_climate_nigeria.csv  (monthly per region)
 - regional_annual_climate_features.csv (annual aggregates for FNN)
 - nigeria_co2_emissions.csv             (World Bank EN.ATM.CO2E.KT, 1990-2023)
 - nigeria_crop_yields.csv               (FAOSTAT attempt, fallback to local synthetic file)

Usage:
    python scripts/build_project_ready_data.py

Dependencies: requests, pandas, numpy
"""

import os
import sys
import requests
import json
import numpy as np
import pandas as pd
from io import StringIO

# Reproducible
np.random.seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA = os.path.join(ROOT, 'project_data')
OUT_DIR = os.path.join(ROOT, 'ProjectReadyData')
os.makedirs(OUT_DIR, exist_ok=True)

# Config
START_YEAR = 1990
END_YEAR = 2023
REGIONS = [
    'North_Central', 'North_East', 'North_West',
    'South_East', 'South_South', 'South_West'
]


def build_monthly_climate():
    """Read existing combined daily climate and produce monthly per-region CSV."""
    src = os.path.join(PROJECT_DATA, 'climate', 'all_regions_combined_1990_2023.csv')
    if not os.path.exists(src):
        raise FileNotFoundError(f"Daily climate file not found: {src}")

    print('Loading daily combined climate...')
    df = pd.read_csv(src, index_col=0, parse_dates=True)

    # Ensure Region column exists
    if 'Region' not in df.columns:
        raise RuntimeError('Region column missing in climate file')

    # Resample to monthly per region: temp mean, temp max/min aggregation, precipitation sum
    print('Resampling to monthly per region...')
    monthly_list = []
    for region in df['Region'].unique():
        region_df = df[df['Region'] == region].copy()
        # Select relevant columns and resample
        r = region_df.resample('M').agg({
            'Temperature_Avg_C': 'mean',
            'Temperature_Max_C': 'max',
            'Temperature_Min_C': 'min',
            'Precipitation_mm': 'sum'
        })
        r['Region'] = region
        r['Year'] = r.index.year
        r['Month'] = r.index.month
        monthly_list.append(r.reset_index(drop=True))

    monthly = pd.concat(monthly_list, ignore_index=True)
    # Reorder
    monthly = monthly[['Region', 'Year', 'Month', 'Temperature_Avg_C', 'Temperature_Max_C', 'Temperature_Min_C', 'Precipitation_mm']]

    out_path = os.path.join(OUT_DIR, 'regional_monthly_climate_nigeria.csv')
    monthly.to_csv(out_path, index=False)
    print('Saved:', out_path)
    return out_path


def build_annual_climate_features(monthly_csv_path=None):
    """From monthly climate, compute annual rainfall total, mean temperature, max temperature, rainfall variability."""
    if monthly_csv_path is None:
        monthly_csv_path = os.path.join(OUT_DIR, 'regional_monthly_climate_nigeria.csv')
    if not os.path.exists(monthly_csv_path):
        raise FileNotFoundError(monthly_csv_path)

    print('Loading monthly climate...')
    m = pd.read_csv(monthly_csv_path)

    # Group and compute
    print('Computing annual features...')
    agg = m.groupby(['Region', 'Year']).agg(
        annual_rainfall_mm=('Precipitation_mm', 'sum'),
        rainfall_std_mm=('Precipitation_mm', 'std'),
        mean_temp_C=('Temperature_Avg_C', 'mean'),
        max_temp_C=('Temperature_Max_C', 'max'),
        min_temp_C=('Temperature_Min_C', 'min')
    ).reset_index()

    # Fill NaN std (single-month or constant) with 0
    agg['rainfall_std_mm'] = agg['rainfall_std_mm'].fillna(0.0)

    out_path = os.path.join(OUT_DIR, 'regional_annual_climate_features.csv')
    agg.to_csv(out_path, index=False)
    print('Saved:', out_path)
    return out_path


def fetch_worldbank_co2(start=START_YEAR, end=END_YEAR):
    """Fetch EN.ATM.CO2E.KT for Nigeria from World Bank.
    Returns CSV path. If fetch fails, raises but does not crash the script.
    """
    print('Fetching World Bank CO2 emissions (EN.ATM.CO2E.KT) ...')
    url = f'http://api.worldbank.org/v2/country/NG/indicator/EN.ATM.CO2E.KT?format=json&date={start}:{end}&per_page=1000'
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            raise RuntimeError('Unexpected World Bank response')

        records = []
        for rec in data[1]:
            year = rec.get('date')
            val = rec.get('value')
            if year is not None:
                records.append({'Year': int(year), 'CO2_Emissions_kt': val if val is not None else np.nan})

        df = pd.DataFrame(records).drop_duplicates(subset=['Year']).sort_values('Year')
        out_path = os.path.join(OUT_DIR, 'nigeria_co2_emissions.csv')
        df.to_csv(out_path, index=False)
        print('Saved:', out_path)
        return out_path

    except Exception as e:
        print('World Bank CO2 fetch failed:', str(e))
        # If fetch failed, attempt to create a proxy from local NOAA ppm by converting ppm trend to kt is non-trivial.
        # We'll save an empty file with Year and NaNs so downstream code can handle it.
        years = list(range(start, end + 1))
        df = pd.DataFrame({'Year': years, 'CO2_Emissions_kt': [np.nan] * len(years)})
        out_path = os.path.join(OUT_DIR, 'nigeria_co2_emissions.csv')
        df.to_csv(out_path, index=False)
        print('Saved placeholder:', out_path)
        return out_path


def fetch_faostat_crop_yields(start=START_YEAR, end=END_YEAR):
    """Attempt to fetch FAOSTAT yields for Maize, Cassava, Yam. If fails, fall back to local synthetic file.
    Output: nigeria_crop_yields.csv with columns Year,Crop,Yield_kg_per_ha
    """
    crops = ['Maize', 'Cassava', 'Yam']
    print('Attempting FAOSTAT fetch (best-effort)...')

    # Best-effort endpoint used earlier; FAO APIs are unstable so we fall back on failure
    base = 'https://fenixservices.fao.org/faostat/api/v1/en/data/production'
    params = {'country': 'Nigeria', 'year': f'{start}-{end}'}
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        recs = []
        for it in js.get('data', []):
            item = it.get('item') or it.get('crop') or it.get('commodity')
            if not item:
                continue
            if any(c.lower() in str(item).lower() for c in crops):
                year = it.get('year')
                value = it.get('value')
                # FAOSTAT 'value' may be production; yields may not be provided here.
                recs.append({'Year': int(year), 'Crop': item, 'Yield_kg_per_ha': value})

        if recs:
            df = pd.DataFrame(recs)
            out_path = os.path.join(OUT_DIR, 'nigeria_crop_yields.csv')
            df.to_csv(out_path, index=False)
            print('Saved FAOSTAT results:', out_path)
            return out_path
        else:
            raise RuntimeError('FAOSTAT returned no matching crop yield records')

    except Exception as e:
        print('FAOSTAT fetch failed or incomplete:', e)
        # Fallback: use local synthetic file (tons/ha) and convert to kg/ha
        local = os.path.join(PROJECT_DATA, 'crop_yields', 'nigeria_crop_yields_1990_2023.csv')
        if not os.path.exists(local):
            raise FileNotFoundError('No FAOSTAT data and no local fallback file')

        df_local = pd.read_csv(local)
        # df_local columns: Year,Region,Crop,Yield_tons_per_ha,Production_1000_tons
        if 'Yield_tons_per_ha' in df_local.columns:
            df_local['Yield_kg_per_ha'] = df_local['Yield_tons_per_ha'] * 1000.0
        elif 'Yield' in df_local.columns:
            df_local['Yield_kg_per_ha'] = df_local['Yield']  # assume already kg/ha
        else:
            raise RuntimeError('Local crop yields missing expected columns')

        df_out = df_local[['Year', 'Crop', 'Yield_kg_per_ha']].copy()
        out_path = os.path.join(OUT_DIR, 'nigeria_crop_yields.csv')
        df_out.to_csv(out_path, index=False)
        print('Saved fallback crop yields:', out_path)
        return out_path


def main():
    print('Building ProjectReadyData in', OUT_DIR)

    # 1) monthly climate
    try:
        monthly_path = build_monthly_climate()
    except Exception as e:
        print('Failed to build monthly climate:', e)
        monthly_path = None

    # 2) annual climate features
    if monthly_path:
        try:
            annual_path = build_annual_climate_features(monthly_path)
        except Exception as e:
            print('Failed to build annual climate features:', e)
            annual_path = None

    # 3) World Bank CO2
    try:
        co2_path = fetch_worldbank_co2()
    except Exception as e:
        print('Failed to fetch World Bank CO2:', e)
        co2_path = None

    # 4) FAOSTAT crop yields (or fallback)
    try:
        crop_path = fetch_faostat_crop_yields()
    except Exception as e:
        print('Failed to fetch/create crop yields:', e)
        crop_path = None

    print('\nSummary of outputs saved to ProjectReadyData:')
    for fname in ['regional_monthly_climate_nigeria.csv', 'regional_annual_climate_features.csv', 'nigeria_co2_emissions.csv', 'nigeria_crop_yields.csv']:
        p = os.path.join(OUT_DIR, fname)
        print(' -', p, '->', 'OK' if os.path.exists(p) else 'MISSING')


if __name__ == '__main__':
    main()
