"""
process_zonal_data.py
Processes raw climate data for analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class ClimateDataProcessor:
    def __init__(self):
        """Initialize processor"""
        self.zones = [
            'North_Central', 'North_East', 'North_West',
            'South_East', 'South_South', 'South_West'
        ]
        
        # Create output directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/processed/monthly', exist_ok=True)
        os.makedirs('data/processed/annual', exist_ok=True)
        os.makedirs('data/processed/statistics', exist_ok=True)
    
    def load_raw_data(self, filename='data/raw/all_zones_combined.csv'):
        """Load combined raw data"""
        try:
            print(f"Loading data from {filename}...")
            df = pd.read_csv(filename, low_memory=False)
            print(f"✓ Loaded {len(df):,} records")
            return df
        except FileNotFoundError:
            print("Warning: Combined file not found, loading individual zone files...")
            return self.load_individual_zone_files()
    
    def load_individual_zone_files(self):
        """Load data from individual zone files"""
        all_data = []
        
        for zone in self.zones:
            filename = f"data/raw/{zone}_raw.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, low_memory=False)
                df['zone'] = zone  # Ensure zone column exists
                all_data.append(df)
                print(f"  Loaded {zone}: {len(df):,} records")
            else:
                print(f"  Warning: {filename} not found")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ Combined {len(all_data)} zones: {len(combined_df):,} total records")
            return combined_df
        else:
            raise FileNotFoundError("No raw data files found")
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("\nCleaning data...")
        
        # Ensure date column exists
        if 'date' not in df.columns and 'YEAR' in df.columns and 'MO' in df.columns:
            # NASA POWER format
            df['date'] = pd.to_datetime(
                df['YEAR'].astype(str) + '-' + 
                df['MO'].astype(str).str.zfill(2) + '-01'
            )
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['season'] = df['month'].apply(self.get_season)
        
        # Standardize column names
        column_mapping = {
            'T2M': 'temperature',
            'T2M_MAX': 'temperature_max',
            'T2M_MIN': 'temperature_min',
            'PRECTOTCORR': 'precipitation',
            'RH2M': 'relative_humidity',
            'WS2M': 'wind_speed',
            'ALLSKY_SFC_SW_DWN': 'solar_radiation'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Separate grouping for interpolation
        for col in numeric_cols:
            if col not in ['year', 'month', 'quarter', 'day', 'latitude', 'longitude']:
                # Group by location and interpolate
                if 'city' in df.columns:
                    for city in df['city'].unique():
                        mask = df['city'] == city
                        df.loc[mask, col] = df.loc[mask, col].interpolate(method='linear')
                else:
                    df[col] = df[col].interpolate(method='linear')
        
        # Fill any remaining NaN with column mean
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['date', 'city', 'zone'], keep='first')
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  Removed {removed} duplicate records")
        
        print(f"✓ Data cleaning complete: {len(df):,} clean records")
        return df
    
    def get_season(self, month):
        """Classify month into season for Nigeria"""
        if month in [12, 1, 2]:
            return 'Dry Season (Dec-Feb)'
        elif month in [3, 4, 5]:
            return 'Early Rainy Season (Mar-May)'
        elif month in [6, 7, 8, 9]:
            return 'Peak Rainy Season (Jun-Sep)'
        else:  # 10, 11
            return 'Late Rainy Season (Oct-Nov)'
    
    def aggregate_monthly(self, df):
        """Aggregate daily data to monthly level"""
        print("\nAggregating to monthly level...")
        
        # Group by zone, city, year, month
        agg_dict = {
            'temperature': ['mean', 'max', 'min', 'std'],
            'precipitation': ['sum', 'mean', 'max', 'std'],
            'relative_humidity': 'mean',
            'wind_speed': 'mean',
            'solar_radiation': 'mean',
            'latitude': 'first',
            'longitude': 'first'
        }
        
        monthly_df = df.groupby(['zone', 'city', 'year', 'month']).agg(agg_dict).reset_index()
        
        # Flatten multi-index columns
        monthly_df.columns = ['_'.join(col).strip('_') for col in monthly_df.columns.values]
        monthly_df = monthly_df.rename(columns={
            'zone_': 'zone',
            'city_': 'city',
            'year_': 'year',
            'month_': 'month',
            'latitude_first': 'latitude',
            'longitude_first': 'longitude'
        })
        
        # Create month-year column
        monthly_df['month_year'] = pd.to_datetime(
            monthly_df['year'].astype(str) + '-' + 
            monthly_df['month'].astype(str).str.zfill(2) + '-01'
        )
        
        # Save monthly data
        monthly_df.to_csv('data/processed/monthly/climate_monthly.csv', index=False)
        
        # Save by zone
        for zone in monthly_df['zone'].unique():
            zone_monthly = monthly_df[monthly_df['zone'] == zone]
            zone_monthly.to_csv(f'data/processed/monthly/{zone}_monthly.csv', index=False)
        
        print(f"✓ Monthly aggregation complete: {len(monthly_df):,} monthly records")
        return monthly_df
    
    def aggregate_annual(self, df):
        """Aggregate to annual level"""
        print("\nAggregating to annual level...")
        
        # Group by zone, city, year
        annual_agg = {
            'temperature_mean': ['mean', 'max', 'min', 'std'],
            'temperature_max_mean': 'mean',
            'temperature_min_mean': 'mean',
            'precipitation_sum': ['sum', 'mean', 'max', 'std'],
            'relative_humidity_mean': 'mean',
            'wind_speed_mean': 'mean',
            'solar_radiation_mean': 'mean'
        }
        
        annual_df = df.groupby(['zone', 'city', 'year']).agg(annual_agg).reset_index()
        
        # Flatten columns
        annual_df.columns = ['_'.join(col).strip('_') for col in annual_df.columns.values]
        annual_df = annual_df.rename(columns={
            'zone_': 'zone',
            'city_': 'city',
            'year_': 'year'
        })
        
        # Clean column names
        annual_df.columns = [col.replace('_mean_mean', '_mean')
                           .replace('_sum_sum', '_total')
                           for col in annual_df.columns]
        
        # Calculate derived metrics
        annual_df['precipitation_anomaly'] = (
            annual_df['precipitation_sum_total'] - 
            annual_df['precipitation_sum_total'].mean()
        )
        
        annual_df['temperature_anomaly'] = (
            annual_df['temperature_mean_mean'] - 
            annual_df['temperature_mean_mean'].mean()
        )
        
        # Save annual data
        annual_df.to_csv('data/processed/annual/climate_annual.csv', index=False)
        
        # Save by zone
        for zone in annual_df['zone'].unique():
            zone_annual = annual_df[annual_df['zone'] == zone]
            zone_annual.to_csv(f'data/processed/annual/{zone}_annual.csv', index=False)
        
        print(f"✓ Annual aggregation complete: {len(annual_df):,} annual records")
        return annual_df
    
    def calculate_statistics(self, monthly_df, annual_df):
        """Calculate statistical summaries"""
        print("\nCalculating statistics...")
        
        # Zone-level statistics
        zone_stats = []
        
        for zone in self.zones:
            zone_monthly = monthly_df[monthly_df['zone'] == zone]
            zone_annual = annual_df[annual_df['zone'] == zone]
            
            if len(zone_monthly) > 0:
                stats = {
                    'zone': zone,
                    'records': len(zone_monthly),
                    'cities': zone_monthly['city'].nunique(),
                    'years': zone_annual['year'].nunique(),
                    'avg_temperature': zone_annual['temperature_mean_mean'].mean(),
                    'avg_precipitation': zone_annual['precipitation_sum_total'].mean(),
                    'max_temperature': zone_annual['temperature_mean_max'].max(),
                    'min_temperature': zone_annual['temperature_mean_min'].min(),
                    'precipitation_std': zone_annual['precipitation_sum_total'].std(),
                    'temperature_std': zone_annual['temperature_mean_mean'].std(),
                    'start_year': zone_annual['year'].min(),
                    'end_year': zone_annual['year'].max()
                }
                
                # Calculate trends (linear regression slope)
                years = zone_annual['year'].values
                temp = zone_annual['temperature_mean_mean'].values
                rain = zone_annual['precipitation_sum_total'].values
                
                if len(years) > 1:
                    stats['temp_trend_per_decade'] = np.polyfit(years, temp, 1)[0] * 10
                    stats['rain_trend_per_decade'] = np.polyfit(years, rain, 1)[0] * 10
                else:
                    stats['temp_trend_per_decade'] = 0
                    stats['rain_trend_per_decade'] = 0
                
                zone_stats.append(stats)
        
        stats_df = pd.DataFrame(zone_stats)
        stats_df.to_csv('data/processed/statistics/zone_statistics.csv', index=False)
        
        # City-level statistics
        city_stats = []
        for (zone, city), group in monthly_df.groupby(['zone', 'city']):
            stats = {
                'zone': zone,
                'city': city,
                'latitude': group['latitude'].iloc[0],
                'longitude': group['longitude'].iloc[0],
                'avg_temperature': group['temperature_mean'].mean(),
                'avg_precipitation': group['precipitation_sum'].mean(),
                'records': len(group)
            }
            city_stats.append(stats)
        
        city_stats_df = pd.DataFrame(city_stats)
        city_stats_df.to_csv('data/processed/statistics/city_statistics.csv', index=False)
        
        print("✓ Statistics calculated and saved")
        return stats_df, city_stats_df
    
    def create_summary_report(self, stats_df):
        """Create a summary report of the data processing"""
        print("\nCreating summary report...")
        
        report = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'zones_processed': len(stats_df),
            'total_cities': stats_df['cities'].sum(),
            'total_records': stats_df['records'].sum(),
            'time_period': f"{stats_df['start_year'].min()}-{stats_df['end_year'].max()}",
            'zone_summary': {}
        }
        
        for _, row in stats_df.iterrows():
            report['zone_summary'][row['zone']] = {
                'cities': int(row['cities']),
                'records': int(row['records']),
                'avg_temperature': round(row['avg_temperature'], 2),
                'avg_precipitation': round(row['avg_precipitation'], 2),
                'temp_trend_per_decade': round(row.get('temp_trend_per_decade', 0), 3),
                'rain_trend_per_decade': round(row.get('rain_trend_per_decade', 0), 1)
            }
        
        # Save report
        with open('data/processed/processing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✓ Summary report created")
        
        # Print summary to console
        print("\n" + "="*60)
        print("DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Zones processed: {len(stats_df)}")
        print(f"Total cities: {stats_df['cities'].sum()}")
        print(f"Time period: {stats_df['start_year'].min()}-{stats_df['end_year'].max()}")
        print(f"Total monthly records: {stats_df['records'].sum():,}")
        
        for zone in report['zone_summary']:
            print(f"\n{zone}:")
            print(f"  Cities: {report['zone_summary'][zone]['cities']}")
            print(f"  Avg Temp: {report['zone_summary'][zone]['avg_temperature']}°C")
            print(f"  Avg Rain: {report['zone_summary'][zone]['avg_precipitation']:.0f} mm/yr")
        
        print("="*60)
        
        return report
    
    def run_full_processing(self):
        """Run complete data processing pipeline"""
        print("="*60)
        print("NIGERIA CLIMATE DATA PROCESSOR")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Step 1: Load data
        df = self.load_raw_data()
        
        # Step 2: Clean data
        cleaned_df = self.clean_data(df)
        
        # Step 3: Save cleaned data
        cleaned_df.to_csv('data/processed/cleaned_climate_data.csv', index=False)
        
        # Step 4: Aggregate to monthly
        monthly_df = self.aggregate_monthly(cleaned_df)
        
        # Step 5: Aggregate to annual
        annual_df = self.aggregate_annual(monthly_df)
        
        # Step 6: Calculate statistics
        stats_df, city_stats_df = self.calculate_statistics(monthly_df, annual_df)
        
        # Step 7: Create report
        report = self.create_summary_report(stats_df)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        
        return {
            'cleaned': cleaned_df,
            'monthly': monthly_df,
            'annual': annual_df,
            'statistics': stats_df,
            'city_stats': city_stats_df,
            'report': report
        }

def main():
    """Main execution function"""
    print("Starting Nigeria Climate Data Processing...")
    
    processor = ClimateDataProcessor()
    results = processor.run_full_processing()
    
    print("\nAll data processing completed successfully!")
    print("Output files saved in 'data/processed/' directory")

if __name__ == "__main__":
    main()