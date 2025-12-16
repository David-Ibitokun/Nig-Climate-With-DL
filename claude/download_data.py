"""
Complete Data Download Script for Climate Change & Food Security Project
Downloads climate data (NASA POWER), CO2 data (NOAA), and crop data (FAOSTAT)
for all 6 Nigerian regions from 1990-2023
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Create directories for data storage
os.makedirs('data/raw/climate', exist_ok=True)
os.makedirs('data/raw/co2', exist_ok=True)
os.makedirs('data/raw/crops', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("=" * 80)
print("CLIMATE CHANGE & FOOD SECURITY DATA DOWNLOAD SCRIPT")
print("=" * 80)
print()

# ============================================================================
# PART 1: DOWNLOAD CLIMATE DATA FROM NASA POWER
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: DOWNLOADING CLIMATE DATA FROM NASA POWER")
print("=" * 80)

# Define Nigerian regions with coordinates
regions = {
    'North_Central': {'lat': 8.9, 'lon': 8.3, 'name': 'North Central'},
    'North_East': {'lat': 11.5, 'lon': 12.5, 'name': 'North East'},
    'North_West': {'lat': 12.0, 'lon': 7.0, 'name': 'North West'},
    'South_East': {'lat': 6.0, 'lon': 7.5, 'name': 'South East'},
    'South_South': {'lat': 5.5, 'lon': 6.5, 'name': 'South South'},
    'South_West': {'lat': 7.5, 'lon': 4.0, 'name': 'South West'}
}

def download_nasa_power_data(lat, lon, start_year, end_year, region_name):
    """
    Download climate data from NASA POWER API
    Parameters: T2M (Temperature at 2m), PRECTOTCORR (Precipitation)
    """
    print(f"\n  Downloading data for {region_name}...")
    print(f"  Coordinates: {lat}°N, {lon}°E")
    
    # NASA POWER API endpoint
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    parameters = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"  # Temperature, Rain, Humidity
    
    # Format dates
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    
    # Build URL
    url = (f"{base_url}?"
           f"parameters={parameters}&"
           f"community=AG&"
           f"longitude={lon}&"
           f"latitude={lat}&"
           f"start={start_date}&"
           f"end={end_date}&"
           f"format=JSON")
    
    try:
        print(f"  Requesting data from NASA POWER API...")
        response = requests.get(url, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract parameters
            params_data = data['properties']['parameter']
            
            # Create dataframe
            dates = list(params_data['T2M'].keys())
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates, format='%Y%m%d'),
                'Temperature': [params_data['T2M'][d] for d in dates],
                'Temperature_Max': [params_data['T2M_MAX'][d] for d in dates],
                'Temperature_Min': [params_data['T2M_MIN'][d] for d in dates],
                'Rainfall': [params_data['PRECTOTCORR'][d] for d in dates],
                'Humidity': [params_data['RH2M'][d] for d in dates]
            })
            
            # Add region info
            df['Region'] = region_name
            df['Latitude'] = lat
            df['Longitude'] = lon
            
            # Replace -999 (missing values) with NaN
            df.replace(-999, np.nan, inplace=True)
            
            print(f"  ✓ Successfully downloaded {len(df)} days of data")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            return df
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error downloading data: {e}")
        return None

# Download climate data for all regions
climate_data_all = []

for region_key, region_info in regions.items():
    df = download_nasa_power_data(
        lat=region_info['lat'],
        lon=region_info['lon'],
        start_year=1990,
        end_year=2023,
        region_name=region_info['name']
    )
    
    if df is not None:
        # Save individual region file
        filename = f"data/raw/climate/{region_key}_1990_2023.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved to: {filename}")
        
        climate_data_all.append(df)
        
        # Be nice to the API - wait between requests
        time.sleep(2)
    else:
        print(f"  Failed to download data for {region_info['name']}")

# Combine all regional climate data
if climate_data_all:
    climate_combined = pd.concat(climate_data_all, ignore_index=True)
    climate_combined.to_csv('data/raw/climate/all_regions_combined.csv', index=False)
    print(f"\n✓ Combined climate data saved: {len(climate_combined)} records")
    print(f"  Regions: {climate_combined['Region'].nunique()}")
    print(f"  Date range: {climate_combined['Date'].min()} to {climate_combined['Date'].max()}")
else:
    print("\n✗ No climate data was downloaded successfully")

# ============================================================================
# PART 2: DOWNLOAD CO2 DATA FROM NOAA
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: DOWNLOADING CO2 DATA FROM NOAA")
print("=" * 80)

def download_noaa_co2_data():
    """
    Download monthly CO2 data from NOAA Mauna Loa Observatory
    """
    print("\n  Downloading CO2 concentration data from NOAA...")
    
    # NOAA CO2 data URL (monthly averages)
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    
    try:
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            # Parse the text file
            lines = response.text.split('\n')
            
            # Skip header lines (lines starting with #)
            data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            
            # Parse data
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        co2_avg = float(parts[3])
                        
                        # Only keep data from 1990 onwards
                        if year >= 1990 and co2_avg > 0:
                            data.append({
                                'Year': year,
                                'Month': month,
                                'CO2': co2_avg
                            })
                    except ValueError:
                        continue
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
            
            # Save
            filename = 'data/raw/co2/noaa_co2_monthly.csv'
            df.to_csv(filename, index=False)
            
            print(f"  ✓ Successfully downloaded CO2 data")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"  Records: {len(df)}")
            print(f"  Saved to: {filename}")
            
            return df
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error downloading CO2 data: {e}")
        return None

co2_data = download_noaa_co2_data()

# ============================================================================
# PART 3: DOWNLOAD CROP DATA FROM FAOSTAT
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: DOWNLOADING CROP DATA FROM FAOSTAT")
print("=" * 80)

def download_faostat_data():
    """
    Download crop production data from FAOSTAT API
    Note: FAOSTAT API can be complex. This is a simplified approach.
    You may need to download manually from: https://www.fao.org/faostat/en/#data
    """
    print("\n  FAOSTAT API Access Information:")
    print("  " + "-" * 76)
    print("  The FAOSTAT API requires specific dataset codes.")
    print("  For manual download (recommended):")
    print()
    print("  1. Visit: https://www.fao.org/faostat/en/#data/QCL")
    print("  2. Select 'Production' > 'Crops and livestock products'")
    print("  3. Filter by:")
    print("     - Country: Nigeria")
    print("     - Items: Maize (corn), Cassava fresh, Yams")
    print("     - Elements: Yield")
    print("     - Years: 1990-2023")
    print("  4. Download as CSV")
    print("  5. Save to: data/raw/crops/faostat_nigeria_crops.csv")
    print()
    
    # Create a sample CSV template for user to fill
    template = pd.DataFrame({
        'Year': range(1990, 2024),
        'Maize_Yield': [np.nan] * 34,
        'Cassava_Yield': [np.nan] * 34,
        'Yam_Yield': [np.nan] * 34,
        'Maize_Production': [np.nan] * 34,
        'Cassava_Production': [np.nan] * 34,
        'Yam_Production': [np.nan] * 34
    })
    
    template_file = 'data/raw/crops/crop_data_template.csv'
    template.to_csv(template_file, index=False)
    print(f"  ✓ Created template file: {template_file}")
    print("  Fill this template with data from FAOSTAT")
    
    # Try to create sample data for testing (not real data!)
    print("\n  Creating SAMPLE data for testing purposes...")
    print("  (Replace with actual FAOSTAT data for real analysis)")
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Year': range(1990, 2024),
        'Maize_Yield': np.random.uniform(1.2, 2.2, 34),  # tonnes/ha
        'Cassava_Yield': np.random.uniform(8.0, 13.0, 34),  # tonnes/ha
        'Yam_Yield': np.random.uniform(9.0, 14.0, 34),  # tonnes/ha
    })
    
    sample_file = 'data/raw/crops/sample_crop_data.csv'
    sample_data.to_csv(sample_file, index=False)
    print(f"  ✓ Created sample data: {sample_file}")
    
    return sample_data

crop_data = download_faostat_data()

# ============================================================================
# PART 4: CREATE REGIONAL CROP DATA
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: CREATING REGIONAL CROP DATA")
print("=" * 80)

def create_regional_crop_data(national_crop_data):
    """
    Distribute national crop yields to regions based on literature-based multipliers
    """
    print("\n  Creating regional crop yield estimates...")
    
    # Regional yield multipliers (based on literature)
    regional_multipliers = {
        'Maize': {
            'North Central': 1.1,
            'North East': 0.8,
            'North West': 0.9,
            'South East': 1.2,
            'South South': 1.0,
            'South West': 1.15
        },
        'Cassava': {
            'North Central': 1.0,
            'North East': 0.7,
            'North West': 0.6,
            'South East': 1.3,
            'South South': 1.2,
            'South West': 1.1
        },
        'Yam': {
            'North Central': 1.2,
            'North East': 0.7,
            'North West': 0.5,
            'South East': 1.3,
            'South South': 0.9,
            'South West': 1.1
        }
    }
    
    regional_crop_data = []
    
    for _, row in national_crop_data.iterrows():
        year = row['Year']
        for region_key, region_info in regions.items():
            region_name = region_info['name']
            regional_row = {
                'Year': year,
                'Region': region_name,
                'Maize_Yield': row['Maize_Yield'] * regional_multipliers['Maize'][region_name],
                'Cassava_Yield': row['Cassava_Yield'] * regional_multipliers['Cassava'][region_name],
                'Yam_Yield': row['Yam_Yield'] * regional_multipliers['Yam'][region_name]
            }
            regional_crop_data.append(regional_row)
    
    regional_df = pd.DataFrame(regional_crop_data)
    
    # Save
    filename = 'data/processed/regional_crop_yields.csv'
    regional_df.to_csv(filename, index=False)
    
    print(f"  ✓ Created regional crop data: {len(regional_df)} records")
    print(f"  Years: {regional_df['Year'].min()} - {regional_df['Year'].max()}")
    print(f"  Regions: {regional_df['Region'].nunique()}")
    print(f"  Saved to: {filename}")
    
    # Show sample
    print("\n  Sample regional data:")
    print(regional_df.head(12).to_string(index=False))
    
    return regional_df

if crop_data is not None:
    regional_crop_data = create_regional_crop_data(crop_data)

# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)

summary = []

# Check climate data
if climate_data_all:
    summary.append("✓ Climate data downloaded successfully")
    summary.append(f"  - {len(climate_data_all)} regions")
    summary.append(f"  - {sum(len(df) for df in climate_data_all)} total daily records")
else:
    summary.append("✗ Climate data download failed")

# Check CO2 data
if co2_data is not None:
    summary.append("✓ CO2 data downloaded successfully")
    summary.append(f"  - {len(co2_data)} monthly records")
else:
    summary.append("✗ CO2 data download failed")

# Check crop data
if crop_data is not None:
    summary.append("✓ Crop data created (sample/template)")
    summary.append(f"  - {len(crop_data)} years")
else:
    summary.append("✗ Crop data creation failed")

for line in summary:
    print(line)

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. VERIFY DATA:
   - Check data/raw/climate/ for climate data files
   - Check data/raw/co2/ for CO2 data
   - Check data/raw/crops/ for crop data

2. GET REAL CROP DATA:
   - Visit https://www.fao.org/faostat/en/#data/QCL
   - Download actual Nigeria crop yields (1990-2023)
   - Replace sample data with real FAOSTAT data

3. DATA PREPROCESSING:
   - Run the data preprocessing script
   - Handle missing values
   - Create monthly aggregates
   - Merge all datasets

4. MODEL DEVELOPMENT:
   - Build baseline models
   - Implement FNN, LSTM, and Hybrid models
   - Train and evaluate

""")

print("=" * 80)
print("DATA DOWNLOAD COMPLETE!")
print("=" * 80)

# Create a status file
status = {
    'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'climate_data_downloaded': len(climate_data_all) > 0,
    'co2_data_downloaded': co2_data is not None,
    'crop_data_created': crop_data is not None,
    'total_climate_records': sum(len(df) for df in climate_data_all) if climate_data_all else 0,
    'total_co2_records': len(co2_data) if co2_data is not None else 0,
    'regions_downloaded': len(climate_data_all) if climate_data_all else 0
}

status_df = pd.DataFrame([status])
status_df.to_csv('data/download_status.csv', index=False)
print("\nStatus saved to: data/download_status.csv")