"""
download_zonal_data.py
Downloads climate data from NASA POWER for Nigeria's 6 geopolitical zones
"""

import pandas as pd
import numpy as np
from nasapower import nasapower
import time
import json
import os
from datetime import datetime

class NigeriaClimateDownloader:
    def __init__(self):
        """Initialize with Nigeria's geopolitical zones"""
        self.zones = {
            'North_Central': {
                'states': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'FCT'],
                'lat_range': (7.5, 11),
                'lon_range': (5.5, 10),
                'cities': [
                    ('Abuja', 9.0765, 7.3986),
                    ('Jos', 9.8965, 8.8583),
                    ('Minna', 9.5833, 6.5500)
                ]
            },
            'North_East': {
                'states': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
                'lat_range': (10, 13.5),
                'lon_range': (9, 14),
                'cities': [
                    ('Maiduguri', 11.8333, 13.1500),
                    ('Yola', 9.2035, 12.4954),
                    ('Bauchi', 10.3103, 9.8439)
                ]
            },
            'North_West': {
                'states': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
                'lat_range': (11, 13.5),
                'lon_range': (3.5, 9),
                'cities': [
                    ('Kano', 12.0022, 8.5920),
                    ('Kaduna', 10.5264, 7.4388),
                    ('Sokoto', 13.0059, 5.2476)
                ]
            },
            'South_East': {
                'states': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
                'lat_range': (5.5, 7),
                'lon_range': (6.5, 9),
                'cities': [
                    ('Enugu', 6.4528, 7.5108),
                    ('Owerri', 5.4836, 7.0266),
                    ('Umuahia', 5.5333, 7.4833)
                ]
            },
            'South_South': {
                'states': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
                'lat_range': (4.5, 7),
                'lon_range': (5, 9),
                'cities': [
                    ('Port Harcourt', 4.8156, 7.0498),
                    ('Calabar', 4.9757, 8.3417),
                    ('Benin City', 6.3176, 5.6145)
                ]
            },
            'South_West': {
                'states': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo'],
                'lat_range': (6, 9),
                'lon_range': (2.5, 6),
                'cities': [
                    ('Lagos', 6.5244, 3.3792),
                    ('Ibadan', 7.3775, 3.9470),
                    ('Abeokuta', 7.1475, 3.3619)
                ]
            }
        }
        
        # Climate parameters to download
        self.parameters = [
            "T2M",           # Temperature at 2 meters
            "T2M_MAX",       # Maximum temperature
            "T2M_MIN",       # Minimum temperature
            "PRECTOTCORR",   # Precipitation corrected
            "RH2M",          # Relative humidity at 2 meters
            "WS2M",          # Wind speed at 2 meters
            "ALLSKY_SFC_SW_DWN"  # Solar radiation
        ]
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            'data/raw',
            'data/processed',
            'data/visualizations',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def download_city_data(self, city_name, lat, lon, start_year=1990, end_year=2023):
        """Download data for a single city/location"""
        try:
            print(f"  Downloading {city_name} ({lat:.4f}, {lon:.4f})...")
            
            data = nasapower.power_point_data(
                community="AG",
                lonlat=(lon, lat),
                dates=f"{start_year}0101",
                end=f"{end_year}1231",
                temporal_api="daily",
                parameters=self.parameters
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data['properties']['parameter'])
            
            # Add metadata
            df['city'] = city_name
            df['latitude'] = lat
            df['longitude'] = lon
            
            # Convert index to proper date
            df['date'] = pd.to_datetime(df.index, format='%Y%m%d')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            return df
            
        except Exception as e:
            print(f"    Error downloading {city_name}: {str(e)}")
            return None
    
    def download_zone_data(self, zone_name, method='cities'):
        """Download data for an entire zone"""
        print(f"\n{'='*60}")
        print(f"Downloading data for {zone_name} zone")
        print(f"{'='*60}")
        
        zone_info = self.zones[zone_name]
        all_data = []
        
        if method == 'cities':
            # Download from representative cities
            for city_name, lat, lon in zone_info['cities']:
                city_data = self.download_city_data(city_name, lat, lon)
                if city_data is not None:
                    city_data['zone'] = zone_name
                    all_data.append(city_data)
                    time.sleep(1)  # Rate limiting
        
        elif method == 'grid':
            # Download from grid points (more comprehensive but slower)
            lat_step = 0.5
            lon_step = 0.5
            
            lats = np.arange(zone_info['lat_range'][0], 
                            zone_info['lat_range'][1], 
                            lat_step)
            lons = np.arange(zone_info['lon_range'][0], 
                            zone_info['lon_range'][1], 
                            lon_step)
            
            for lat in lats:
                for lon in lons:
                    city_data = self.download_city_data(
                        f"Grid_{lat:.2f}_{lon:.2f}", lat, lon
                    )
                    if city_data is not None:
                        city_data['zone'] = zone_name
                        all_data.append(city_data)
                        time.sleep(1)  # Rate limiting
        
        if all_data:
            # Combine all data for this zone
            zone_df = pd.concat(all_data, ignore_index=True)
            
            # Save raw data
            raw_filename = f"data/raw/{zone_name}_raw.csv"
            zone_df.to_csv(raw_filename, index=False)
            print(f"✓ Saved {len(zone_df)} records to {raw_filename}")
            
            return zone_df
        else:
            print(f"✗ No data downloaded for {zone_name}")
            return None
    
    def download_all_zones(self):
        """Download data for all zones"""
        print("="*60)
        print("NIGERIA CLIMATE DATA DOWNLOADER")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        all_zones_data = {}
        download_log = []
        
        for zone_name in self.zones.keys():
            start_time = time.time()
            
            try:
                zone_data = self.download_zone_data(zone_name, method='cities')
                
                if zone_data is not None:
                    all_zones_data[zone_name] = zone_data
                    
                    # Log download
                    elapsed = time.time() - start_time
                    download_log.append({
                        'zone': zone_name,
                        'records': len(zone_data),
                        'start_year': zone_data['year'].min(),
                        'end_year': zone_data['year'].max(),
                        'time_seconds': round(elapsed, 2),
                        'status': 'success'
                    })
                    
                    print(f"✓ Completed {zone_name} in {elapsed:.2f} seconds")
                else:
                    download_log.append({
                        'zone': zone_name,
                        'records': 0,
                        'status': 'failed'
                    })
                    
            except Exception as e:
                print(f"✗ Error in {zone_name}: {str(e)}")
                download_log.append({
                    'zone': zone_name,
                    'records': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Combine all zones data
        if all_zones_data:
            combined_df = pd.concat(all_zones_data.values(), ignore_index=True)
            combined_filename = "data/raw/all_zones_combined.csv"
            combined_df.to_csv(combined_filename, index=False)
            print(f"\n✓ Saved combined data: {len(combined_df)} records")
        
        # Save download log
        log_df = pd.DataFrame(download_log)
        log_filename = f"logs/download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_df.to_csv(log_filename, index=False)
        
        # Save zone metadata
        with open('data/zone_metadata.json', 'w') as f:
            json.dump(self.zones, f, indent=2)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print(f"Total zones processed: {len([x for x in download_log if x['status'] == 'success'])}")
        print(f"Log saved to: {log_filename}")
        print("="*60)
        
        return all_zones_data

def main():
    """Main execution function"""
    print("Initializing Nigeria Climate Data Downloader...")
    
    # Initialize downloader
    downloader = NigeriaClimateDownloader()
    
    # Download all zones
    downloader.download_all_zones()
    
    print("\nData download completed successfully!")

if __name__ == "__main__":
    # Install required package if not present
    try:
        import nasapower
    except ImportError:
        print("Installing nasapower package...")
        import subprocess
        subprocess.check_call(["pip", "install", "nasapower"])
    
    main()