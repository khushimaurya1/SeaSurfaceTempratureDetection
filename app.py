"""
Flask Web Frontend for Sea Surface Temperature Detection
"""

from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import os
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

# Import SST modules from seasurface.py
import sys
sys.path.insert(0, os.path.dirname(__file__))

from seasurface import (
    SSTConfig, SSTDataHandler, SSTAnalyzer, 
    ThermalImageProcessor, SSTVisualizer
)

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
        
        print(f"Generating SST data for {days} days...")
        data = SSTDataHandler.generate_sample_sst_data(days=days)
        
        # Calculate statistics
        stats = SSTAnalyzer.calculate_sst_statistics(data['sst'])
        
        # Time series
        sst_mean_series = np.mean(data['sst'], axis=(1, 2))
        anomalies = sst_mean_series - np.mean(sst_mean_series)
        
        # Store results
        analysis_results['data'] = data
        analysis_results['stats'] = stats
        analysis_results['sst_mean_series'] = sst_mean_series
        analysis_results['anomalies'] = anomalies
        
        return jsonify({
            'success': True,
            'message': f'Generated {days} days of SST data',
            'stats': {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
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
        anomalies = analysis_results['anomalies']
        
        # Generate visualizations
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
        data = analysis_results['data']
        
        # Detect MHW events
        mhw_events = SSTAnalyzer.identify_mhw_events(sst_mean_series, threshold=threshold, duration=duration)
        
        # Calculate trend
        trend = SSTAnalyzer.calculate_sst_trend(sst_mean_series, data['dates'])
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(sst_mean_series)), sst_mean_series, 'b-', linewidth=2, label='SST')
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold}°C)')
        
        # Highlight events
        for event in mhw_events:
            ax.axvspan(event['start_day'], event['end_day'], alpha=0.2, color='red', 
                      label='MHW Event' if event == mhw_events[0] else '')
        
        # Trend line
        trend_line = trend['slope'] * np.arange(len(sst_mean_series)) + trend['intercept']
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
                'slope': float(trend['slope']),
                'r_squared': float(trend['r_squared']),
                'p_value': float(trend['p_value'])
            },
            'visualization': visualization
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/thermal-image', methods=['POST'])
def thermal_image_analysis():
    """Analyze thermal image"""
    try:
        # Generate thermal image
        thermal_img = ThermalImageProcessor.generate_thermal_image()
        temp_image = ThermalImageProcessor.thermal_to_temperature(thermal_img)
        
        hotspot_threshold = request.json.get('hotspot_threshold', 15)
        coldspot_threshold = request.json.get('coldspot_threshold', -250)
        
        # Detect anomalies
        hotspots = ThermalImageProcessor.detect_hot_spots(temp_image, threshold=hotspot_threshold)
        coldspots = ThermalImageProcessor.detect_cold_spots(temp_image, threshold=coldspot_threshold)
        gradients = ThermalImageProcessor.calculate_temperature_gradient(temp_image)
        
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
        mask = hotspots['mask'].astype(float)
        ax.imshow(mask, cmap='Reds', alpha=0.5)
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title(f'Hotspot Detection (Threshold: {hotspot_threshold}°C)')
        visualizations['hotspots'] = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'hotspots_count': len(hotspots['hotspots']),
            'coldspots_count': len(coldspots['coldspots']),
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


# ============================================================================
# API ENDPOINTS - ALL ROUTES REQUIRED BY FRONTEND
# ============================================================================

@app.route('/api/sst-analysis')
def get_sst_analysis():
    """Generate SST statistics and trend analysis"""
    try:
        # Generate sample data
        data = SSTDataHandler.generate_sample_sst_data(days=30)
        
        # Calculate statistics
        stats = SSTAnalyzer.calculate_sst_statistics(data['sst'])
        
        # Calculate anomalies
        anomalies = SSTAnalyzer.calculate_anomalies(data['sst'])
        
        # Time series analysis
        sst_mean_series = np.mean(data['sst'], axis=(1, 2))
        mhw_events = SSTAnalyzer.identify_mhw_events(sst_mean_series, threshold=25, duration=3)
        trend = SSTAnalyzer.calculate_sst_trend(sst_mean_series, data['dates'])
        
        # Store data for other endpoints
        analysis_results['data'] = data
        analysis_results['sst_mean_series'] = sst_mean_series
        
        return jsonify({
            'statistics': {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
            },
            'trend': {
                'slope': float(trend['slope']),
                'total_change': float(trend['slope'] * 30),
                'r_squared': float(trend['r_squared']),
            },
            'mhw_events': len(mhw_events),
            'data_days': len(data['dates']),
        })
    except Exception as e:
        print(f"Error in get_sst_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sst-map')
def get_sst_map():
    """Generate SST temperature map visualization"""
    try:
        if 'data' not in analysis_results:
            data = SSTDataHandler.generate_sample_sst_data(days=30)
            analysis_results['data'] = data
        else:
            data = analysis_results['data']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data['sst'][0], cmap='RdYlBu_r', extent=[-180, 180, -90, 90], aspect='auto')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title('Global Sea Surface Temperature')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        
        plot_url = fig_to_base64(fig)
        return jsonify({'image': plot_url})
    except Exception as e:
        print(f"Error in get_sst_map: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/time-series')
def get_time_series():
    """Generate time series visualization"""
    try:
        if 'data' not in analysis_results:
            data = SSTDataHandler.generate_sample_sst_data(days=30)
            analysis_results['data'] = data
        else:
            data = analysis_results['data']
        
        if 'sst_mean_series' not in analysis_results:
            sst_mean_series = np.mean(data['sst'], axis=(1, 2))
            analysis_results['sst_mean_series'] = sst_mean_series
        else:
            sst_mean_series = analysis_results['sst_mean_series']
        
        anomalies = sst_mean_series - np.mean(sst_mean_series)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['dates'], sst_mean_series, 'b-', linewidth=2, label='SST Mean')
        ax.fill_between(data['dates'], sst_mean_series - anomalies.std(), 
                         sst_mean_series + anomalies.std(), alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Sea Surface Temperature Time Series')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_url = fig_to_base64(fig)
        return jsonify({'image': plot_url})
    except Exception as e:
        print(f"Error in get_time_series: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/thermal-analysis')
def get_thermal_analysis():
    """Generate thermal image analysis"""
    try:
        # Generate thermal image
        thermal_img = ThermalImageProcessor.generate_thermal_image()
        temp_image = ThermalImageProcessor.thermal_to_temperature(thermal_img)
        
        # Detect anomalies
        hotspots = ThermalImageProcessor.detect_hot_spots(temp_image, threshold=15)
        coldspots = ThermalImageProcessor.detect_cold_spots(temp_image, threshold=-250)
        
        return jsonify({
            'hotspots_detected': len(hotspots['hotspots']),
            'coldspots_detected': len(coldspots['coldspots']),
            'largest_hotspot': int(max([hs['size'] for hs in hotspots['hotspots']], default=0)),
            'highest_temp': float(max([hs['max_temp'] for hs in hotspots['hotspots']], default=0)),
        })
    except Exception as e:
        print(f"Error in get_thermal_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/thermal-image')
def get_thermal_image():
    """Generate thermal image visualization"""
    try:
        thermal_img = ThermalImageProcessor.generate_thermal_image()
        temp_image = ThermalImageProcessor.thermal_to_temperature(thermal_img)
        hotspots = ThermalImageProcessor.detect_hot_spots(temp_image, threshold=15)
        
        fig = plt.figure(figsize=(12, 6))
        im = plt.imshow(temp_image, cmap='hot')
        plt.colorbar(im, label='Temperature (°C)')
        plt.title('Thermal Image Temperature Map')
        
        # Plot hotspots
        for hotspot in hotspots['hotspots']:
            y, x = hotspot['center']
            plt.plot(x, y, 'c*', markersize=15)
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'image': plot_url})
    except Exception as e:
        print(f"Error in get_thermal_image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/temperature-distribution')
def get_temperature_distribution():
    """Generate temperature distribution histogram"""
    try:
        if 'data' not in analysis_results:
            data = SSTDataHandler.generate_sample_sst_data(days=30)
            analysis_results['data'] = data
        else:
            data = analysis_results['data']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['sst'].flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Frequency')
        ax.set_title('Sea Surface Temperature Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_url = fig_to_base64(fig)
        return jsonify({'image': plot_url})
    except Exception as e:
        print(f"Error in get_temperature_distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" SEA SURFACE TEMPERATURE DETECTION - WEB APPLICATION")
    print("="*70)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\n" + "="*70 + "\n")
    
    SSTConfig.setup_directories()
    app.run(debug=True, host='0.0.0.0', port=5000)
