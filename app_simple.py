"""
Simple Flask Web Frontend for Sea Surface Temperature Detection
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import os
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global storage for analysis results
analysis_results = {}


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generate synthetic SST data"""
    try:
        days = request.json.get('days', 30)
        lat_points, lon_points = 180, 360
        
        print(f"Generating SST data for {days} days...")
        
        # Generate synthetic data
        lat = np.linspace(-90, 90, lat_points)
        lon = np.linspace(-180, 180, lon_points)
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        
        # Synthetic SST with seasonal variation and anomalies
        np.random.seed(42)
        sst = np.zeros((days, lat_points, lon_points))
        
        for day in range(days):
            base_temp = 15 + 10 * np.sin(2 * np.pi * day / 365)
            lat_gradient = 20 - 0.2 * np.abs(np.repeat(lat[:, np.newaxis], lon_points, axis=1))
            sst[day] = lat_gradient + base_temp + np.random.normal(0, 0.5, (lat_points, lon_points))
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(sst)),
            'std': float(np.std(sst)),
            'min': float(np.min(sst)),
            'max': float(np.max(sst)),
            'count': int(sst.size)
        }
        
        # Store results
        analysis_results['data'] = {
            'sst': sst,
            'latitude': lat,
            'longitude': lon,
            'dates': dates
        }
        analysis_results['stats'] = stats
        analysis_results['sst_mean_series'] = np.mean(sst, axis=(1, 2))
        
        return jsonify({
            'success': True,
            'message': f'Generated {days} days of SST data',
            'stats': {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'days': days
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/analyze-temperature', methods=['POST'])
def analyze_temperature():
    """Analyze temperature and return visualizations"""
    try:
        if 'data' not in analysis_results:
            return jsonify({'success': False, 'error': 'No data generated. Generate data first.'}), 400
        
        data = analysis_results['data']
        sst_mean_series = analysis_results['sst_mean_series']
        
        visualizations = {}
        
        # 1. Temperature Map
        fig, ax = plt.subplots(figsize=(10, 6))
        temp_map = ax.contourf(data['longitude'], data['latitude'], data['sst'][0], 
                               levels=20, cmap='RdYlBu_r')
        plt.colorbar(temp_map, ax=ax, label='Temperature (°C)')
        ax.set_title('Sea Surface Temperature Map (Day 1)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        visualizations['temperature_map'] = fig_to_base64(fig)
        
        # 2. Time Series
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(sst_mean_series)), sst_mean_series, 'b-', linewidth=2, label='SST')
        ax.fill_between(range(len(sst_mean_series)), 
                        sst_mean_series - np.std(sst_mean_series),
                        sst_mean_series + np.std(sst_mean_series),
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax.set_xlabel('Days')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Sea Surface Temperature Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        visualizations['time_series'] = fig_to_base64(fig)
        
        # 3. Anomalies
        anomalies = sst_mean_series - np.mean(sst_mean_series)
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['red' if x > 0 else 'blue' for x in anomalies]
        ax.bar(range(len(anomalies)), anomalies, color=colors, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('Anomaly (°C)')
        ax.set_title('Temperature Anomalies')
        ax.grid(True, alpha=0.3, axis='y')
        visualizations['anomalies'] = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'visualizations': visualizations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/detect-events', methods=['POST'])
def detect_events():
    """Detect Marine Heatwave events"""
    try:
        if 'sst_mean_series' not in analysis_results:
            return jsonify({'success': False, 'error': 'No data available. Generate data first.'}), 400
        
        threshold = request.json.get('threshold', 25)
        duration = request.json.get('duration', 3)
        
        sst_mean_series = analysis_results['sst_mean_series']
        
        # Detect MHW events
        above_threshold = sst_mean_series > threshold
        mhw_events = []
        in_event = False
        event_start: int | None = None
        event_temps = []
        
        for i, is_hot in enumerate(above_threshold):
            if is_hot and not in_event:
                in_event = True
                event_start = i
                event_temps = [sst_mean_series[i]]
            elif is_hot and in_event:
                event_temps.append(sst_mean_series[i])
            elif not is_hot and in_event:
                event_duration = i - (event_start if event_start is not None else 0)
                if event_duration >= duration:
                    mhw_events.append({
                        'start_day': event_start,
                        'end_day': i,
                        'duration': event_duration,
                        'max_temp': max(event_temps),
                        'mean_temp': np.mean(event_temps),
                        'intensity': max(event_temps) - threshold
                    })
                in_event = False
                event_temps = []
        
        # Handle event at end of series
        if in_event and len(event_temps) >= duration:
            mhw_events.append({
                'start_day': event_start if event_start is not None else 0,
                'end_day': len(sst_mean_series),
                'duration': len(sst_mean_series) - (event_start if event_start is not None else 0),
                'max_temp': max(event_temps),
                'mean_temp': np.mean(event_temps),
                'intensity': max(event_temps) - threshold
            })
        
        # Calculate trend
        days_numeric = np.arange(len(sst_mean_series))
        z = np.polyfit(days_numeric, sst_mean_series, 1)
        slope = z[0]
        intercept = z[1]
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(sst_mean_series)), sst_mean_series, 'b-', linewidth=2, label='SST')
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold}°C)')
        
        # Highlight events
        for event in mhw_events:
            ax.axvspan(event['start_day'], event['end_day'], alpha=0.2, color='red', 
                      label='MHW Event' if event == mhw_events[0] else '')
        
        # Trend line
        trend_line = slope * days_numeric + intercept
        ax.plot(range(len(sst_mean_series)), trend_line, 'g--', linewidth=2, label='Trend')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Marine Heatwave Detection (Threshold: {threshold}°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        visualization = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'events': [
                {
                    'start_day': int(e['start_day']),
                    'end_day': int(e['end_day']),
                    'duration': int(e['duration']),
                    'max_temp': float(e['max_temp']),
                    'mean_temp': float(e['mean_temp']),
                    'intensity': float(e['intensity'])
                } for e in mhw_events
            ],
            'trend': {
                'slope': float(slope),
                'p_value': 0.0  # Simplified for demo
            },
            'visualization': visualization
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/thermal-image', methods=['POST'])
def thermal_image_analysis():
    """Analyze thermal image"""
    try:
        # Generate synthetic thermal image
        thermal_img = np.random.rand(128, 128) * 65535
        temp_image = (thermal_img / 65535.0) * 50 - 20  # Convert to -20 to 30°C
        
        hotspot_threshold = request.json.get('hotspot_threshold', 15)
        
        # Find hotspots
        hotspot_mask = temp_image > hotspot_threshold
        coldspot_mask = temp_image < -10
        
        visualizations = {}
        
        # Temperature distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(temp_image, cmap='RdYlBu_r')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title('Temperature Distribution')
        visualizations['temperature'] = fig_to_base64(fig)
        
        # Hotspot detection
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(temp_image, cmap='gray', alpha=0.7)
        mask = hotspot_mask.astype(float)
        ax.imshow(mask, cmap='Reds', alpha=0.5)
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title(f'Hotspot Detection (Threshold: {hotspot_threshold}°C)')
        visualizations['hotspots'] = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'hotspots_count': int(np.sum(hotspot_mask)),
            'coldspots_count': int(np.sum(coldspot_mask)),
            'temperature_range': {
                'min': float(np.min(temp_image)),
                'max': float(np.max(temp_image)),
                'mean': float(np.mean(temp_image))
            },
            'visualizations': visualizations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get analysis summary"""
    try:
        if not analysis_results or 'stats' not in analysis_results:
            return jsonify({
                'success': False,
                'error': 'No analysis data available'
            }), 400
        
        stats = analysis_results['stats']
        
        return jsonify({
            'success': True,
            'summary': {
                'mean_temp': float(stats['mean']),
                'std_dev': float(stats['std']),
                'min_temp': float(stats['min']),
                'max_temp': float(stats['max']),
                'data_points': int(stats['count'])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" SEA SURFACE TEMPERATURE DETECTION - WEB INTERFACE")
    print("="*70)
    print("\n🌊 Server starting on http://localhost:5000")
    print("   Open your browser and navigate to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)
