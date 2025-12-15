"""
analyze_zonal_trends.py
Analyzes climate trends for Nigeria's geopolitical zones
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClimateTrendAnalyzer:
    def __init__(self):
        """Initialize analyzer"""
        self.zones = [
            'North_Central', 'North_East', 'North_West',
            'South_East', 'South_South', 'South_West'
        ]
        
        # Create output directories
        os.makedirs('data/analysis', exist_ok=True)
        os.makedirs('data/analysis/trends', exist_ok=True)
        os.makedirs('data/analysis/anomalies', exist_ok=True)
        
        # Color scheme for zones
        self.zone_colors = {
            'North_Central': '#1f77b4',    # Blue
            'North_East': '#ff7f0e',       # Orange
            'North_West': '#2ca02c',       # Green
            'South_East': '#d62728',       # Red
            'South_South': '#9467bd',      # Purple
            'South_West': '#8c564b'        # Brown
        }
    
    def load_processed_data(self):
        """Load processed climate data"""
        print("Loading processed data...")
        
        # Load annual data
        annual_path = 'data/processed/annual/climate_annual.csv'
        if os.path.exists(annual_path):
            annual_df = pd.read_csv(annual_path)
            print(f"✓ Loaded annual data: {len(annual_df):,} records")
        else:
            raise FileNotFoundError(f"Annual data not found at {annual_path}")
        
        # Load monthly data
        monthly_path = 'data/processed/monthly/climate_monthly.csv'
        if os.path.exists(monthly_path):
            monthly_df = pd.read_csv(monthly_path)
            print(f"✓ Loaded monthly data: {len(monthly_df):,} records")
        else:
            monthly_df = None
            print("Warning: Monthly data not found")
        
        # Load statistics
        stats_path = 'data/processed/statistics/zone_statistics.csv'
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path)
            print(f"✓ Loaded zone statistics")
        else:
            stats_df = None
        
        return {
            'annual': annual_df,
            'monthly': monthly_df,
            'statistics': stats_df
        }
    
    def calculate_linear_trends(self, annual_df):
        """Calculate linear trends for each zone"""
        print("\nCalculating linear trends...")
        
        trends = []
        
        for zone in self.zones:
            zone_data = annual_df[annual_df['zone'] == zone]
            
            if len(zone_data) < 5:  # Need sufficient data
                print(f"  Warning: Insufficient data for {zone}")
                continue
            
            # Group by year for zone-level trends
            zone_yearly = zone_data.groupby('year').agg({
                'temperature_mean_mean': 'mean',
                'precipitation_sum_total': 'mean'
            }).reset_index()
            
            years = zone_yearly['year'].values
            n_years = len(years)
            
            # Temperature trend
            if 'temperature_mean_mean' in zone_yearly.columns:
                temp = zone_yearly['temperature_mean_mean'].values
                temp_slope, temp_intercept, temp_r, temp_p, temp_std_err = stats.linregress(years, temp)
                
                # Precipitation trend
                if 'precipitation_sum_total' in zone_yearly.columns:
                    rain = zone_yearly['precipitation_sum_total'].values
                    rain_slope, rain_intercept, rain_r, rain_p, rain_std_err = stats.linregress(years, rain)
                else:
                    rain_slope = rain_intercept = rain_r = rain_p = rain_std_err = np.nan
            
            # Calculate per decade trends
            temp_trend_per_decade = temp_slope * 10
            rain_trend_per_decade = rain_slope * 10
            
            # Classify trend significance
            temp_significant = temp_p < 0.05
            rain_significant = rain_p < 0.05
            
            # Classify trend direction and magnitude
            temp_direction = self.classify_trend(temp_trend_per_decade, temp_significant)
            rain_direction = self.classify_trend(rain_trend_per_decade, rain_significant, is_temp=False)
            
            trend_info = {
                'zone': zone,
                'n_years': n_years,
                'start_year': years.min(),
                'end_year': years.max(),
                
                # Temperature trends
                'temp_slope': temp_slope,
                'temp_intercept': temp_intercept,
                'temp_r_squared': temp_r**2,
                'temp_p_value': temp_p,
                'temp_std_error': temp_std_err,
                'temp_trend_per_decade': temp_trend_per_decade,
                'temp_significant': temp_significant,
                'temp_direction': temp_direction,
                
                # Precipitation trends
                'rain_slope': rain_slope,
                'rain_intercept': rain_intercept,
                'rain_r_squared': rain_r**2,
                'rain_p_value': rain_p,
                'rain_std_error': rain_std_err,
                'rain_trend_per_decade': rain_trend_per_decade,
                'rain_significant': rain_significant,
                'rain_direction': rain_direction,
                
                # Current climate (last 10 years average)
                'recent_temp': zone_yearly[zone_yearly['year'] >= years.max() - 10]['temperature_mean_mean'].mean(),
                'recent_rain': zone_yearly[zone_yearly['year'] >= years.max() - 10]['precipitation_sum_total'].mean()
            }
            
            trends.append(trend_info)
        
        trends_df = pd.DataFrame(trends)
        
        # Save trends
        trends_df.to_csv('data/analysis/trends/linear_trends.csv', index=False)
        print(f"✓ Calculated trends for {len(trends_df)} zones")
        
        return trends_df
    
    def classify_trend(self, trend_per_decade, significant, is_temp=True):
        """Classify trend direction and magnitude"""
        if not significant:
            return 'No significant trend'
        
        if is_temp:
            if trend_per_decade > 0.5:
                return 'Strong warming'
            elif trend_per_decade > 0.2:
                return 'Moderate warming'
            elif trend_per_decade > 0:
                return 'Slight warming'
            elif trend_per_decade < -0.5:
                return 'Strong cooling'
            elif trend_per_decade < -0.2:
                return 'Moderate cooling'
            else:
                return 'Slight cooling'
        else:
            if trend_per_decade > 50:
                return 'Strong wetting'
            elif trend_per_decade > 20:
                return 'Moderate wetting'
            elif trend_per_decade > 0:
                return 'Slight wetting'
            elif trend_per_decade < -50:
                return 'Strong drying'
            elif trend_per_decade < -20:
                return 'Moderate drying'
            else:
                return 'Slight drying'
    
    def calculate_anomalies(self, annual_df, baseline_start=1991, baseline_end=2020):
        """Calculate climate anomalies relative to baseline period"""
        print("\nCalculating climate anomalies...")
        
        anomalies_data = []
        
        for zone in self.zones:
            zone_data = annual_df[annual_df['zone'] == zone].copy()
            
            # Calculate baseline (1991-2020 climatology)
            baseline_mask = (zone_data['year'] >= baseline_start) & (zone_data['year'] <= baseline_end)
            
            if baseline_mask.sum() > 0:
                baseline_temp = zone_data.loc[baseline_mask, 'temperature_mean_mean'].mean()
                baseline_rain = zone_data.loc[baseline_mask, 'precipitation_sum_total'].mean()
                
                # Calculate anomalies for all years
                zone_data['temp_anomaly'] = zone_data['temperature_mean_mean'] - baseline_temp
                zone_data['temp_anomaly_percent'] = (zone_data['temp_anomaly'] / baseline_temp) * 100
                zone_data['rain_anomaly'] = zone_data['precipitation_sum_total'] - baseline_rain
                zone_data['rain_anomaly_percent'] = (zone_data['rain_anomaly'] / baseline_rain) * 100
                
                # Classify anomaly years
                zone_data['temp_anomaly_class'] = zone_data['temp_anomaly'].apply(
                    lambda x: 'Extreme Heat' if x > 1.5 else
                             'Moderate Heat' if x > 0.5 else
                             'Normal' if x > -0.5 else
                             'Moderate Cool' if x > -1.5 else 'Extreme Cool'
                )
                
                zone_data['rain_anomaly_class'] = zone_data['rain_anomaly_percent'].apply(
                    lambda x: 'Extreme Wet' if x > 30 else
                             'Moderate Wet' if x > 10 else
                             'Normal' if x > -10 else
                             'Moderate Dry' if x > -30 else 'Extreme Dry'
                )
                
                # Calculate frequency of extreme events
                extreme_heat_years = (zone_data['temp_anomaly'] > 1.5).sum()
                extreme_dry_years = (zone_data['rain_anomaly_percent'] < -30).sum()
                
                anomalies_data.append({
                    'zone': zone,
                    'baseline_temp': baseline_temp,
                    'baseline_rain': baseline_rain,
                    'extreme_heat_frequency': extreme_heat_years,
                    'extreme_dry_frequency': extreme_dry_years,
                    'avg_temp_anomaly': zone_data['temp_anomaly'].mean(),
                    'avg_rain_anomaly': zone_data['rain_anomaly'].mean()
                })
                
                # Save zone anomalies
                zone_data.to_csv(f'data/analysis/anomalies/{zone}_anomalies.csv', index=False)
        
        anomalies_summary = pd.DataFrame(anomalies_data)
        anomalies_summary.to_csv('data/analysis/anomalies/anomalies_summary.csv', index=False)
        
        print(f"✓ Calculated anomalies for {len(anomalies_summary)} zones")
        return anomalies_summary
    
    def calculate_extreme_indices(self, monthly_df):
        """Calculate climate extreme indices"""
        print("\nCalculating extreme climate indices...")
        
        if monthly_df is None:
            print("  Skipping: Monthly data not available")
            return None
        
        extreme_indices = []
        
        for zone in self.zones:
            zone_data = monthly_df[monthly_df['zone'] == zone].copy()
            
            if len(zone_data) == 0:
                continue
            
            # Convert to datetime if needed
            if 'month_year' in zone_data.columns:
                zone_data['date'] = pd.to_datetime(zone_data['month_year'])
            elif 'year' in zone_data.columns and 'month' in zone_data.columns:
                zone_data['date'] = pd.to_datetime(
                    zone_data['year'].astype(str) + '-' + 
                    zone_data['month'].astype(str).str.zfill(2) + '-01'
                )
            
            zone_data.set_index('date', inplace=True)
            
            # Calculate annual indices
            yearly_indices = []
            
            for year in zone_data['year'].unique():
                year_data = zone_data[zone_data['year'] == year]
                
                if len(year_data) >= 10:  # Need most months
                    indices = {
                        'zone': zone,
                        'year': year,
                        'tx90p': self.calculate_tx90p(year_data),  # Warm days percentage
                        'tn90p': self.calculate_tn90p(year_data),  # Warm nights percentage
                        'r95p': self.calculate_r95p(year_data),    # Very wet days
                        'cdd': self.calculate_cdd(year_data),      # Consecutive dry days
                        'cwd': self.calculate_cwd(year_data),      # Consecutive wet days
                        'rx1day': self.calculate_rx1day(year_data), # Max 1-day precipitation
                        'rx5day': self.calculate_rx5day(year_data)  # Max 5-day precipitation
                    }
                    
                    yearly_indices.append(indices)
            
            if yearly_indices:
                # Calculate trends in extremes
                yearly_df = pd.DataFrame(yearly_indices)
                
                # Simple trend calculation for key indices
                for index in ['tx90p', 'r95p', 'cdd']:
                    if index in yearly_df.columns:
                        slope, _ = np.polyfit(yearly_df['year'], yearly_df[index], 1)
                        extreme_indices.append({
                            'zone': zone,
                            'index': index,
                            'trend_per_decade': slope * 10,
                            'mean_value': yearly_df[index].mean()
                        })
        
        if extreme_indices:
            indices_df = pd.DataFrame(extreme_indices)
            indices_df.to_csv('data/analysis/trends/extreme_indices.csv', index=False)
            print(f"✓ Calculated extreme indices for {indices_df['zone'].nunique()} zones")
            return indices_df
        else:
            return None
    
    def calculate_tx90p(self, data):
        """Percentage of days with Tmax > 90th percentile"""
        if 'temperature_max' in data.columns:
            threshold = np.percentile(data['temperature_max'], 90)
            return (data['temperature_max'] > threshold).mean() * 100
        return np.nan
    
    def calculate_tn90p(self, data):
        """Percentage of days with Tmin > 90th percentile"""
        if 'temperature_min' in data.columns:
            threshold = np.percentile(data['temperature_min'], 90)
            return (data['temperature_min'] > threshold).mean() * 100
        return np.nan
    
    def calculate_r95p(self, data):
        """Precipitation from very wet days (>95th percentile)"""
        if 'precipitation' in data.columns:
            threshold = np.percentile(data['precipitation'], 95)
            return data[data['precipitation'] > threshold]['precipitation'].sum()
        return np.nan
    
    def calculate_cdd(self, data):
        """Maximum consecutive dry days (precipitation < 1mm)"""
        if 'precipitation' in data.columns:
            dry_series = (data['precipitation'] < 1).astype(int)
            max_consecutive = 0
            current = 0
            for val in dry_series:
                if val == 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 0
            return max_consecutive
        return np.nan
    
    def calculate_cwd(self, data):
        """Maximum consecutive wet days (precipitation >= 1mm)"""
        if 'precipitation' in data.columns:
            wet_series = (data['precipitation'] >= 1).astype(int)
            max_consecutive = 0
            current = 0
            for val in wet_series:
                if val == 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 0
            return max_consecutive
        return np.nan
    
    def calculate_rx1day(self, data):
        """Maximum 1-day precipitation"""
        if 'precipitation' in data.columns:
            return data['precipitation'].max()
        return np.nan
    
    def calculate_rx5day(self, data):
        """Maximum 5-day precipitation"""
        if 'precipitation' in data.columns:
            return data['precipitation'].rolling(5).sum().max()
        return np.nan
    
    def perform_mann_kendall_test(self, annual_df):
        """Perform Mann-Kendall trend test for each zone"""
        print("\nPerforming Mann-Kendall trend tests...")
        
        mk_results = []
        
        for zone in self.zones:
            zone_data = annual_df[annual_df['zone'] == zone]
            
            if len(zone_data) < 10:
                continue
            
            # Group by year
            yearly = zone_data.groupby('year').agg({
                'temperature_mean_mean': 'mean',
                'precipitation_sum_total': 'mean'
            }).reset_index()
            
            # Temperature trend test
            if 'temperature_mean_mean' in yearly.columns:
                temp_trend, temp_p, temp_z = self.mann_kendall(yearly['temperature_mean_mean'].values)
                
                # Precipitation trend test
                if 'precipitation_sum_total' in yearly.columns:
                    rain_trend, rain_p, rain_z = self.mann_kendall(yearly['precipitation_sum_total'].values)
                else:
                    rain_trend = rain_p = rain_z = np.nan
                
                mk_results.append({
                    'zone': zone,
                    'temp_trend': temp_trend,
                    'temp_p_value': temp_p,
                    'temp_z_score': temp_z,
                    'temp_significant': temp_p < 0.05,
                    'rain_trend': rain_trend,
                    'rain_p_value': rain_p,
                    'rain_z_score': rain_z,
                    'rain_significant': rain_p < 0.05
                })
        
        mk_df = pd.DataFrame(mk_results)
        mk_df.to_csv('data/analysis/trends/mann_kendall_results.csv', index=False)
        
        print(f"✓ Mann-Kendall tests completed for {len(mk_df)} zones")
        return mk_df
    
    def mann_kendall(self, x):
        """Mann-Kendall trend test"""
        n = len(x)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(x[j] - x[i])
        
        # Calculate variance
        var_s = n * (n-1) * (2*n + 5) / 18
        
        # Calculate Z
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        
        trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
        
        return trend, p, z
    
    def calculate_vulnerability_scores(self, trends_df, anomalies_df):
        """Calculate climate vulnerability scores for each zone"""
        print("\nCalculating vulnerability scores...")
        
        vulnerability_data = []
        
        for zone in self.zones:
            # Get trend data
            trend_data = trends_df[trends_df['zone'] == zone]
            anomaly_data = anomalies_df[anomalies_df['zone'] == zone]
            
            if len(trend_data) == 0 or len(anomaly_data) == 0:
                continue
            
            trend_row = trend_data.iloc[0]
            anomaly_row = anomaly_data.iloc[0]
            
            # Normalize indicators (0-1 scale, higher = more vulnerable)
            indicators = {}
            
            # Temperature trend vulnerability
            temp_trend = abs(trend_row['temp_trend_per_decade'])
            indicators['temp_trend'] = min(temp_trend / 1.0, 1)  # Cap at 1°C/decade
            
            # Precipitation trend vulnerability
            rain_trend = abs(trend_row['rain_trend_per_decade'])
            indicators['rain_trend'] = min(rain_trend / 100, 1)  # Cap at 100mm/decade
            
            # Extreme heat frequency
            if 'extreme_heat_frequency' in anomaly_row:
                heat_freq = anomaly_row['extreme_heat_frequency']
                indicators['heat_freq'] = min(heat_freq / 10, 1)  # Cap at 10 years
            
            # Extreme dry frequency
            if 'extreme_dry_frequency' in anomaly_row:
                dry_freq = anomaly_row['extreme_dry_frequency']
                indicators['dry_freq'] = min(dry_freq / 10, 1)  # Cap at 10 years
            
            # Temperature variability
            if 'temp_std_error' in trend_row:
                temp_var = trend_row['temp_std_error']
                indicators['temp_var'] = min(temp_var / 0.5, 1)  # Cap at 0.5°C
            
            # Calculate weighted vulnerability score
            weights = {
                'temp_trend': 0.3,
                'rain_trend': 0.2,
                'heat_freq': 0.2,
                'dry_freq': 0.2,
                'temp_var': 0.1
            }
            
            # Calculate weighted sum
            vulnerability_score = 0
            weight_sum = 0
            
            for indicator, weight in weights.items():
                if indicator in indicators:
                    vulnerability_score += indicators[indicator] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                vulnerability_score = vulnerability_score / weight_sum * 100
            
            # Classify vulnerability
            if vulnerability_score >= 70:
                vulnerability_class = 'Very High'
            elif vulnerability_score >= 50:
                vulnerability_class = 'High'
            elif vulnerability_score >= 30:
                vulnerability_class = 'Moderate'
            else:
                vulnerability_class = 'Low'
            
            vulnerability_data.append({
                'zone': zone,
                'vulnerability_score': round(vulnerability_score, 1),
                'vulnerability_class': vulnerability_class,
                **indicators
            })
        
        vulnerability_df = pd.DataFrame(vulnerability_data)
        vulnerability_df.to_csv('data/analysis/trends/vulnerability_scores.csv', index=False)
        
        print(f"✓ Vulnerability scores calculated for {len(vulnerability_df)} zones")
        return vulnerability_df
    
    def generate_analysis_report(self, trends_df, anomalies_df, vulnerability_df):
        """Generate comprehensive analysis report"""
        print("\nGenerating analysis report...")
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'zones_analyzed': len(trends_df),
            'key_findings': {},
            'zone_insights': {},
            'recommendations': {}
        }
        
        # Overall findings
        warming_zones = trends_df[trends_df['temp_trend_per_decade'] > 0.2].shape[0]
        drying_zones = trends_df[trends_df['rain_trend_per_decade'] < 0].shape[0]
        
        report['key_findings'] = {
            'zones_warming': int(warming_zones),
            'zones_drying': int(drying_zones),
            'avg_temp_increase': round(trends_df['temp_trend_per_decade'].mean(), 3),
            'avg_rain_change': round(trends_df['rain_trend_per_decade'].mean(), 1),
            'most_vulnerable_zone': vulnerability_df.loc[vulnerability_df['vulnerability_score'].idxmax(), 'zone'],
            'least_vulnerable_zone': vulnerability_df.loc[vulnerability_df['vulnerability_score'].idxmin(), 'zone']
        }
        
        # Zone-specific insights
        for _, row in trends_df.iterrows():
            zone = row['zone']
            
            # Get vulnerability for this zone
            vuln_row = vulnerability_df[vulnerability_df['zone'] == zone]
            vulnerability = vuln_row.iloc[0] if len(vuln_row) > 0 else {}
            
            insight = {
                'temperature_trend': f"{row['temp_trend_per_decade']:.3f}°C/decade",
                'temperature_significant': bool(row['temp_significant']),
                'precipitation_trend': f"{row['rain_trend_per_decade']:.1f} mm/decade",
                'precipitation_significant': bool(row['rain_significant']),
                'vulnerability_score': vulnerability.get('vulnerability_score', 'N/A'),
                'vulnerability_class': vulnerability.get('vulnerability_class', 'N/A')
            }
            
            report['zone_insights'][zone] = insight
        
        # Policy recommendations
        report['recommendations'] = {
            'North_West': 'Focus on drought-resistant crops and water harvesting',
            'North_East': 'Integrated climate adaptation with conflict resolution',
            'North_Central': 'Sustainable land management and irrigation',
            'South_West': 'Urban heat island mitigation and flood control',
            'South_South': 'Coastal protection and mangrove restoration',
            'South_East': 'Erosion control and sustainable agriculture'
        }
        
        # Save report
        report_path = 'data/analysis/analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS REPORT SUMMARY")
        print("="*60)
        print(f"Average temperature increase: {report['key_findings']['avg_temp_increase']}°C/decade")
        print(f"Average precipitation change: {report['key_findings']['avg_rain_change']} mm/decade")
        print(f"Most vulnerable zone: {report['key_findings']['most_vulnerable_zone']}")
        print(f"Least vulnerable zone: {report['key_findings']['least_vulnerable_zone']}")
        print("="*60)
        
        return report
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("="*60)
        print("NIGERIA CLIMATE TREND ANALYZER")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load data
        data = self.load_processed_data()
        
        # Perform analyses
        trends_df = self.calculate_linear_trends(data['annual'])
        anomalies_df = self.calculate_anomalies(data['annual'])
        extreme_indices = self.calculate_extreme_indices(data['monthly'])
        mk_results = self.perform_mann_kendall_test(data['annual'])
        vulnerability_df = self.calculate_vulnerability_scores(trends_df, anomalies_df)
        
        # Generate report
        report = self.generate_analysis_report(trends_df, anomalies_df, vulnerability_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'trends': trends_df,
            'anomalies': anomalies_df,
            'extremes': extreme_indices,
            'mann_kendall': mk_results,
            'vulnerability': vulnerability_df,
            'report': report
        }

def main():
    """Main execution function"""
    print("Starting Nigeria Climate Trend Analysis...")
    
    analyzer = ClimateTrendAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\nAll analyses completed successfully!")
    print("Analysis results saved in 'data/analysis/' directory")

if __name__ == "__main__":
    main()