"""
visualize_zonal_data.py
Creates visualizations for Nigeria's zonal climate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ClimateDataVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.zones = [
            'North_Central', 'North_East', 'North_West',
            'South_East', 'South_South', 'South_West'
        ]
        
        # Color scheme
        self.zone_colors = {
            'North_Central': '#1f77b4',    # Blue
            'North_East': '#ff7f0e',       # Orange
            'North_West': '#2ca02c',       # Green
            'South_East': '#d62728',       # Red
            'South_South': '#9467bd',      # Purple
            'South_West': '#8c564b'        # Brown
        }
        
        # Create output directories
        os.makedirs('data/visualizations/static', exist_ok=True)
        os.makedirs('data/visualizations/interactive', exist_ok=True)
        os.makedirs('data/visualizations/reports', exist_ok=True)
        
        # Font settings
        self.font_dict = {
            'family': 'Arial',
            'size': 12,
            'color': 'black'
        }
    
    def load_analysis_data(self):
        """Load analysis results"""
        print("Loading analysis data...")
        
        data = {}
        
        # Load trends
        trends_path = 'data/analysis/trends/linear_trends.csv'
        if os.path.exists(trends_path):
            data['trends'] = pd.read_csv(trends_path)
            print(f"✓ Loaded trends data")
        
        # Load anomalies
        anomalies_path = 'data/analysis/anomalies/anomalies_summary.csv'
        if os.path.exists(anomalies_path):
            data['anomalies'] = pd.read_csv(anomalies_path)
            print(f"✓ Loaded anomalies data")
        
        # Load vulnerability
        vulnerability_path = 'data/analysis/trends/vulnerability_scores.csv'
        if os.path.exists(vulnerability_path):
            data['vulnerability'] = pd.read_csv(vulnerability_path)
            print(f"✓ Loaded vulnerability data")
        
        # Load annual data
        annual_path = 'data/processed/annual/climate_annual.csv'
        if os.path.exists(annual_path):
            data['annual'] = pd.read_csv(annual_path)
            print(f"✓ Loaded annual data")
        
        return data
    
    def plot_temperature_trends(self, trends_df):
        """Plot temperature trends by zone"""
        print("Creating temperature trend visualization...")
        
        fig = plt.figure(figsize=(14, 8))
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        
        # Flatten axes array
        axs_flat = axs.flatten()
        
        for idx, zone in enumerate(self.zones):
            ax = axs_flat[idx]
            
            # Get zone data
            zone_data = trends_df[trends_df['zone'] == zone]
            
            if len(zone_data) > 0:
                row = zone_data.iloc[0]
                
                # Create scatter plot
                years = np.array([row['start_year'], row['end_year']])
                temps = np.array([row['temp_intercept'] + row['temp_slope'] * years[0],
                                 row['temp_intercept'] + row['temp_slope'] * years[1]])
                
                # Plot trend line
                ax.plot(years, temps, 'r-', linewidth=3, alpha=0.7,
                       label=f"Trend: {row['temp_trend_per_decade']:.2f}°C/decade")
                
                # Add significance indicator
                if row['temp_significant']:
                    ax.text(0.05, 0.95, '***', transform=ax.transAxes,
                           fontsize=14, fontweight='bold', color='red',
                           verticalalignment='top')
                
                # Styling
                ax.set_title(f'{zone.replace("_", " ")}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Year', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=9)
                
                # Set color based on trend
                if row['temp_trend_per_decade'] > 0:
                    ax.set_facecolor('#FFF0F0')  # Light red for warming
                else:
                    ax.set_facecolor('#F0F8FF')  # Light blue for cooling
        
        # Set common y-label
        fig.text(0.04, 0.5, 'Temperature (°C)', va='center', 
                rotation='vertical', fontsize=12, fontweight='bold')
        
        # Set super title
        plt.suptitle('Temperature Trends Across Nigerian Geopolitical Zones', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Adjust layout and save
        plt.tight_layout()
        save_path = 'data/visualizations/static/temperature_trends_by_zone.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def plot_precipitation_trends(self, trends_df):
        """Plot precipitation trends by zone"""
        print("Creating precipitation trend visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, zone in enumerate(self.zones):
            ax = axes[idx]
            
            # Get zone data
            zone_data = trends_df[trends_df['zone'] == zone]
            
            if len(zone_data) > 0:
                row = zone_data.iloc[0]
                
                # Create bar for trend
                trend_value = row['rain_trend_per_decade']
                color = 'blue' if trend_value > 0 else 'red'
                
                # Create bar chart
                bars = ax.bar(['Trend'], [trend_value], color=color, alpha=0.7)
                
                # Add value label
                ax.text(0, trend_value, 
                       f'{trend_value:+.1f} mm/decade',
                       ha='center', va='bottom' if trend_value > 0 else 'top',
                       fontweight='bold', fontsize=10)
                
                # Add significance indicator
                if row['rain_significant']:
                    ax.text(0.5, 0.95 if trend_value > 0 else 0.05, 
                           '***', transform=ax.transAxes,
                           fontsize=14, fontweight='bold', color='darkblue',
                           ha='center', va='top' if trend_value > 0 else 'bottom')
                
                # Add baseline average precipitation
                if 'recent_rain' in row:
                    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
                    ax.text(0.5, 0.1, f"Avg: {row['recent_rain']:.0f} mm/yr",
                           transform=ax.transAxes, ha='center',
                           fontsize=9, style='italic')
            
            # Styling
            ax.set_title(f'{zone.replace("_", " ")}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Precipitation Trend (mm/decade)', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set background color
            if 'rain_direction' in row:
                if 'drying' in str(row['rain_direction']):
                    ax.set_facecolor('#FFF0F0')
                elif 'wetting' in str(row['rain_direction']):
                    ax.set_facecolor('#F0F8FF')
        
        plt.suptitle('Precipitation Trends Across Nigerian Geopolitical Zones', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = 'data/visualizations/static/precipitation_trends_by_zone.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def create_vulnerability_heatmap(self, vulnerability_df):
        """Create vulnerability heatmap"""
        print("Creating vulnerability heatmap...")
        
        # Prepare data for heatmap
        heatmap_data = vulnerability_df.copy()
        heatmap_data = heatmap_data.set_index('zone')
        
        # Select vulnerability indicators
        indicators = ['vulnerability_score']
        if 'temp_trend' in heatmap_data.columns:
            indicators.append('temp_trend')
        if 'rain_trend' in heatmap_data.columns:
            indicators.append('rain_trend')
        if 'heat_freq' in heatmap_data.columns:
            indicators.append('heat_freq')
        
        heatmap_data = heatmap_data[indicators]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create numeric values for heatmap
        heatmap_values = heatmap_data.values
        
        # Create heatmap
        im = ax.imshow(heatmap_values, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        
        # Set labels
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index.str.replace('_', ' '))
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}',
                                  ha="center", va="center",
                                  color="black" if abs(value) < 0.5 else "white",
                                  fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Value (Normalized 0-1)', rotation=-90, va="bottom")
        
        # Title
        ax.set_title("Climate Vulnerability Heatmap - Nigerian Zones", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        save_path = 'data/visualizations/static/vulnerability_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def create_interactive_time_series(self, annual_df):
        """Create interactive time series plot"""
        print("Creating interactive time series...")
        
        # Prepare data
        plot_data = annual_df.copy()
        plot_data['zone'] = plot_data['zone'].str.replace('_', ' ')
        
        # Create interactive plot with Plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Trends', 'Precipitation Trends'),
            vertical_spacing=0.15
        )
        
        # Add temperature traces for each zone
        for zone in plot_data['zone'].unique():
            zone_data = plot_data[plot_data['zone'] == zone]
            
            fig.add_trace(
                go.Scatter(
                    x=zone_data['year'],
                    y=zone_data['temperature_mean_mean'],
                    mode='lines+markers',
                    name=zone,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Year: %{x}<br>' +
                                 'Temperature: %{y:.2f}°C<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add precipitation traces for each zone
        for zone in plot_data['zone'].unique():
            zone_data = plot_data[plot_data['zone'] == zone]
            
            fig.add_trace(
                go.Scatter(
                    x=zone_data['year'],
                    y=zone_data['precipitation_sum_total'],
                    mode='lines+markers',
                    name=zone,
                    line=dict(width=2, dash='dash'),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Year: %{x}<br>' +
                                 'Precipitation: %{y:.0f} mm<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Nigeria Climate Trends by Geopolitical Zone (1990-2023)',
                font=dict(size=20, family='Arial', color='black'),
                x=0.5
            ),
            height=800,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
        
        # Save interactive plot
        save_path = 'data/visualizations/interactive/climate_trends_interactive.html'
        fig.write_html(save_path)
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def create_zone_comparison_dashboard(self, data):
        """Create comprehensive comparison dashboard"""
        print("Creating zone comparison dashboard...")
        
        # Extract data
        trends_df = data.get('trends', pd.DataFrame())
        vulnerability_df = data.get('vulnerability', pd.DataFrame())
        
        if trends_df.empty or vulnerability_df.empty:
            print("Warning: Insufficient data for dashboard")
            return None
        
        # Create dashboard with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature Trends (°C/decade)',
                'Precipitation Trends (mm/decade)',
                'Vulnerability Scores',
                'Recent Average Temperature (°C)',
                'Recent Average Precipitation (mm/yr)',
                'Climate Risk Profile'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatterpolar"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        # Sort zones by vulnerability (highest first)
        vulnerability_df = vulnerability_df.sort_values('vulnerability_score', ascending=False)
        zones_sorted = vulnerability_df['zone'].tolist()
        
        # 1. Temperature trends
        temp_trends = []
        for zone in zones_sorted:
            zone_data = trends_df[trends_df['zone'] == zone]
            if len(zone_data) > 0:
                temp_trends.append(zone_data.iloc[0]['temp_trend_per_decade'])
            else:
                temp_trends.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[z.replace('_', ' ') for z in zones_sorted],
                y=temp_trends,
                marker_color=[self.zone_colors.get(z, 'gray') for z in zones_sorted],
                name='Temp Trend',
                hovertemplate='%{x}<br>Trend: %{y:.2f}°C/decade<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Precipitation trends
        rain_trends = []
        for zone in zones_sorted:
            zone_data = trends_df[trends_df['zone'] == zone]
            if len(zone_data) > 0:
                rain_trends.append(zone_data.iloc[0]['rain_trend_per_decade'])
            else:
                rain_trends.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[z.replace('_', ' ') for z in zones_sorted],
                y=rain_trends,
                marker_color=[self.zone_colors.get(z, 'gray') for z in zones_sorted],
                name='Rain Trend',
                hovertemplate='%{x}<br>Trend: %{y:.1f} mm/decade<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Vulnerability scores
        vuln_scores = vulnerability_df['vulnerability_score'].tolist()
        
        fig.add_trace(
            go.Bar(
                x=[z.replace('_', ' ') for z in zones_sorted],
                y=vuln_scores,
                marker_color=[self.zone_colors.get(z, 'gray') for z in zones_sorted],
                name='Vulnerability',
                hovertemplate='%{x}<br>Score: %{y:.1f}/100<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Recent temperatures
        recent_temps = []
        for zone in zones_sorted:
            zone_data = trends_df[trends_df['zone'] == zone]
            if len(zone_data) > 0 and 'recent_temp' in zone_data.columns:
                recent_temps.append(zone_data.iloc[0]['recent_temp'])
            else:
                recent_temps.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[z.replace('_', ' ') for z in zones_sorted],
                y=recent_temps,
                marker_color=[self.zone_colors.get(z, 'gray') for z in zones_sorted],
                name='Recent Temp',
                hovertemplate='%{x}<br>Temp: %{y:.1f}°C<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Recent precipitation
        recent_rains = []
        for zone in zones_sorted:
            zone_data = trends_df[trends_df['zone'] == zone]
            if len(zone_data) > 0 and 'recent_rain' in zone_data.columns:
                recent_rains.append(zone_data.iloc[0]['recent_rain'])
            else:
                recent_rains.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[z.replace('_', ' ') for z in zones_sorted],
                y=recent_rains,
                marker_color=[self.zone_colors.get(z, 'gray') for z in zones_sorted],
                name='Recent Rain',
                hovertemplate='%{x}<br>Rain: %{y:.0f} mm/yr<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Radar chart for one zone (most vulnerable)
        if not vulnerability_df.empty:
            most_vulnerable = zones_sorted[0]
            mv_data = vulnerability_df[vulnerability_df['zone'] == most_vulnerable]
            
            if not mv_data.empty:
                indicators = ['temp_trend', 'rain_trend', 'heat_freq', 'dry_freq']
                values = []
                
                for indicator in indicators:
                    if indicator in mv_data.columns:
                        values.append(mv_data.iloc[0][indicator] * 100)
                    else:
                        values.append(0)
                
                # Close the radar chart
                values = values + [values[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=indicators + [indicators[0]],
                        fill='toself',
                        name=most_vulnerable.replace('_', ' '),
                        line=dict(color=self.zone_colors.get(most_vulnerable, 'gray')),
                        hovertemplate='%{theta}: %{r:.1f}%<extra></extra>'
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Nigeria Climate Zones Comparison Dashboard',
                font=dict(size=24, family='Arial', color='black'),
                x=0.5
            ),
            height=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        # Save dashboard
        save_path = 'data/visualizations/interactive/zone_comparison_dashboard.html'
        fig.write_html(save_path)
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def create_geospatial_visualization(self, data):
        """Create geospatial visualization of climate data"""
        print("Creating geospatial visualization...")
        
        # Load city statistics
        city_stats_path = 'data/processed/statistics/city_statistics.csv'
        if not os.path.exists(city_stats_path):
            print("Warning: City statistics not found")
            return None
        
        city_df = pd.read_csv(city_stats_path)
        
        # Create map
        fig = go.Figure()
        
        # Add city points
        for zone in city_df['zone'].unique():
            zone_data = city_df[city_df['zone'] == zone]
            
            fig.add_trace(go.Scattergeo(
                lon=zone_data['longitude'],
                lat=zone_data['latitude'],
                text=zone_data['city'] + '<br>' + 
                     'Avg Temp: ' + zone_data['avg_temperature'].round(1).astype(str) + '°C<br>' +
                     'Avg Rain: ' + zone_data['avg_precipitation'].round(0).astype(str) + ' mm',
                mode='markers',
                marker=dict(
                    size=zone_data['avg_precipitation'] / 50,  # Scale by precipitation
                    color=zone_data['avg_temperature'],  # Color by temperature
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Temperature (°C)"),
                    line=dict(width=1, color='white')
                ),
                name=zone.replace('_', ' '),
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Update layout for Nigeria focus
        fig.update_geos(
            visible=False,
            resolution=50,
            showcountries=True,
            countrycolor="Black",
            showsubunits=True,
            subunitcolor="Blue",
            center=dict(lon=8, lat=9),
            projection_scale=6
        )
        
        fig.update_layout(
            title=dict(
                text='Nigeria Climate Monitoring Stations<br>' +
                     '<sup>Marker size = Precipitation, Color = Temperature</sup>',
                font=dict(size=20, family='Arial'),
                x=0.5
            ),
            height=700,
            geo=dict(
                scope='africa',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                subunitcolor='rgb(217, 217, 217)',
                lonaxis_range=[2, 15],
                lataxis_range=[4, 14],
                projection_type='mercator'
            )
        )
        
        # Save map
        save_path = 'data/visualizations/interactive/nigeria_climate_map.html'
        fig.write_html(save_path)
        
        print(f"✓ Saved: {save_path}")
        return save_path
    
    def create_summary_report_pdf(self, data):
        """Create a PDF summary report of visualizations"""
        print("Creating summary report...")
        
        # This would typically use a library like ReportLab or WeasyPrint
        # For now, we'll create an HTML report that can be converted to PDF
        
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nigeria Climate Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .viz {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; 
                         font-size: 0.9em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>Nigeria Climate Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents climate change analysis for Nigeria's six geopolitical zones, 
                covering the period 1990-2023. The analysis includes temperature and precipitation trends, 
                vulnerability assessments, and zone-specific climate profiles.</p>
            </div>
            
            <h2>Key Visualizations</h2>
            
            <div class="viz">
                <h3>Temperature Trends by Zone</h3>
                <img src="temperature_trends_by_zone.png" alt="Temperature Trends">
                <p><em>Figure 1: Temperature trends across Nigerian geopolitical zones</em></p>
            </div>
            
            <div class="viz">
                <h3>Precipitation Trends</h3>
                <img src="precipitation_trends_by_zone.png" alt="Precipitation Trends">
                <p><em>Figure 2: Precipitation trends across Nigerian geopolitical zones</em></p>
            </div>
            
            <div class="viz">
                <h3>Vulnerability Assessment</h3>
                <img src="vulnerability_heatmap.png" alt="Vulnerability Heatmap">
                <p><em>Figure 3: Climate vulnerability heatmap</em></p>
            </div>
            
            <h2>Zone Comparison</h2>
            <table>
                <tr>
                    <th>Zone</th>
                    <th>Temp Trend (°C/decade)</th>
                    <th>Rain Trend (mm/decade)</th>
                    <th>Vulnerability Score</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        # Add data rows
        if 'trends' in data and 'vulnerability' in data:
            trends_df = data['trends']
            vulnerability_df = data['vulnerability']
            
            for zone in self.zones:
                trend_row = trends_df[trends_df['zone'] == zone]
                vuln_row = vulnerability_df[vulnerability_df['zone'] == zone]
                
                if len(trend_row) > 0 and len(vuln_row) > 0:
                    temp_trend = trend_row.iloc[0]['temp_trend_per_decade']
                    rain_trend = trend_row.iloc[0]['rain_trend_per_decade']
                    vuln_score = vuln_row.iloc[0]['vulnerability_score']
                    vuln_class = vuln_row.iloc[0]['vulnerability_class']
                    
                    report_content += f"""
                    <tr>
                        <td>{zone.replace('_', ' ')}</td>
                        <td>{temp_trend:.3f}</td>
                        <td>{rain_trend:+.1f}</td>
                        <td>{vuln_score:.1f}</td>
                        <td>{vuln_class}</td>
                    </tr>
                    """
        
        report_content += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li><strong>North West:</strong> Implement drought-resistant agriculture and water harvesting</li>
                <li><strong>North East:</strong> Integrate climate adaptation with conflict resolution</li>
                <li><strong>North Central:</strong> Promote sustainable land management</li>
                <li><strong>South West:</strong> Develop urban heat island mitigation strategies</li>
                <li><strong>South South:</strong> Enhance coastal protection measures</li>
                <li><strong>South East:</strong> Focus on erosion control and sustainable farming</li>
            </ul>
            
            <div class="footer">
                <p><strong>Report generated by:</strong> Nigeria Climate Analysis System</p>
                <p><strong>Data Source:</strong> NASA POWER Climate Data (1990-2023)</p>
                <p><strong>Analysis Method:</strong> Linear regression, Mann-Kendall test, Vulnerability scoring</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = 'data/visualizations/reports/climate_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ Saved: {report_path}")
        return report_path
    
    def run_all_visualizations(self):
        """Run complete visualization pipeline"""
        print("="*60)
        print("NIGERIA CLIMATE DATA VISUALIZER")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load data
        data = self.load_analysis_data()
        
        if data.get('trends') is None or data['trends'].empty:
            print("Error: No trend data available for visualization")
            return
        
        # Create visualizations
        visualization_files = []
        
        # Static visualizations
        print("\nCreating static visualizations...")
        temp_viz = self.plot_temperature_trends(data['trends'])
        rain_viz = self.plot_precipitation_trends(data['trends'])
        heatmap_viz = self.create_vulnerability_heatmap(data.get('vulnerability', pd.DataFrame()))
        
        visualization_files.extend([temp_viz, rain_viz, heatmap_viz])
        
        # Interactive visualizations
        print("\nCreating interactive visualizations...")
        if 'annual' in data:
            time_series_viz = self.create_interactive_time_series(data['annual'])
            dashboard_viz = self.create_zone_comparison_dashboard(data)
            geospatial_viz = self.create_geospatial_visualization(data)
            
            if time_series_viz:
                visualization_files.append(time_series_viz)
            if dashboard_viz:
                visualization_files.append(dashboard_viz)
            if geospatial_viz:
                visualization_files.append(geospatial_viz)
        
        # Create summary report
        print("\nCreating summary report...")
        report_viz = self.create_summary_report_pdf(data)
        if report_viz:
            visualization_files.append(report_viz)
        
        # Print summary
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Total visualizations created: {len(visualization_files)}")
        print("\nGenerated files:")
        for viz_file in visualization_files:
            print(f"  ✓ {viz_file}")
        print("\nVisualizations saved in 'data/visualizations/' directory")
        print("="*60)
        
        return visualization_files

def main():
    """Main execution function"""
    print("Starting Nigeria Climate Data Visualization...")
    
    visualizer = ClimateDataVisualizer()
    visualizations = visualizer.run_all_visualizations()
    
    print("\nAll visualizations created successfully!")
    
    # Optional: Open the dashboard in browser
    import webbrowser
    dashboard_path = 'data/visualizations/interactive/zone_comparison_dashboard.html'
    if os.path.exists(dashboard_path):
        print(f"\nOpening dashboard in browser: {dashboard_path}")
        webbrowser.open(f'file://{os.path.abspath(dashboard_path)}')

if __name__ == "__main__":
    # Install required packages
    try:
        import plotly
    except ImportError:
        print("Installing visualization packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "plotly", "kaleido"])
    
    main()