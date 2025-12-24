# ============================================================================
# SEA SURFACE TEMPERATURE (SST) DETECTION PROJECT - COMPLETE PYTHON CODE
# ============================================================================
# This project includes multiple approaches:
# 1. Download SST data from NOAA
# 2. Process satellite thermal images
# 3. Visualize SST data on maps
# 4. Time series analysis
# 5. Temperature anomaly detection
# ============================================================================

# PART 1: REQUIRED LIBRARIES INSTALLATION
# ============================================================================
# Run this in terminal:
# pip install numpy pandas matplotlib xarray netCDF4 requests opencv-python
# pip install scipy scikit-image pillow cartopy seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
import json
from scipy import ndimage
from scipy.stats import linregress
# cv2 is optional; import only when needed
from PIL import Image
import warnings
from typing import Tuple
warnings.filterwarnings('ignore')

# ============================================================================
# PART 2: CONFIGURATION
# ============================================================================

class SSTConfig:
    """Configuration for SST Detection Project"""
    
    # NOAA Data endpoints
    NOAA_SST_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
    
    # Data storage directory
    DATA_DIR = "./sst_data"
    OUTPUT_DIR = "./sst_output"
    
    # Geographic boundaries (latitude, longitude)
    REGION_GLOBAL = {
        'lat_min': -90, 'lat_max': 90,
        'lon_min': -180, 'lon_max': 180,
        'name': 'Global'
    }
    
    REGION_INDIAN_OCEAN = {
        'lat_min': -40, 'lat_max': 20,
        'lon_min': 30, 'lon_max': 120,
        'name': 'Indian Ocean'
    }
    
    REGION_PACIFIC = {
        'lat_min': -60, 'lat_max': 60,
        'lon_min': 120, 'lon_max': 280,
        'name': 'Pacific Ocean'
    }
    
    # Temperature thresholds
    NORMAL_TEMP = 20  # °C
    HOT_THRESHOLD = 25  # °C - Marine Heatwave
    COLD_THRESHOLD = 10  # °C - Cold anomaly
    
    # Color mapping for temperature
    CMAP_NAME = 'RdYlBu_r'  # Red (hot) to Blue (cold)
    
    @staticmethod
    def setup_directories():
        """Create necessary directories"""
        os.makedirs(SSTConfig.DATA_DIR, exist_ok=True)
        os.makedirs(SSTConfig.OUTPUT_DIR, exist_ok=True)
        print(f"Directories ready: {SSTConfig.DATA_DIR}, {SSTConfig.OUTPUT_DIR}")


# ============================================================================
# PART 3: DATA DOWNLOAD AND PREPROCESSING
# ============================================================================

class SSTDataHandler:
    """Download and process SST data"""
    
    @staticmethod
    def generate_sample_sst_data(days=30, lat_points=180, lon_points=360):
        """
        Generate synthetic SST data for demonstration
        (When NOAA data is unavailable)
        
        Returns:
            dict: Contains latitude, longitude, and SST data
        """
        print("\nGenerating synthetic SST data...")
        
        lat = np.linspace(-90, 90, lat_points)
        lon = np.linspace(-180, 180, lon_points)
        
        # Create realistic temperature pattern
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        
        # Base temperature increases towards equator
        base_temp = 15 + 10 * np.sin(np.radians(lat_mesh))
        
        # Add coastal variations
        coastal_effect = 5 * np.sin(np.radians(lon_mesh) * 2) * np.cos(np.radians(lat_mesh))
        
        # Random daily variations
        np.random.seed(42)
        daily_variation = np.random.randn(days, lat_points, lon_points) * 0.5
        
        # Combine effects
        sst_data = []
        dates = []
        
        for day in range(days):
            temp = base_temp + coastal_effect + daily_variation[day]
            sst_data.append(temp)
            dates.append(datetime.now() - timedelta(days=days-day))
        
        sst_data = np.array(sst_data)
        
        print(f"Generated {days} days of SST data")
        print(f"  Shape: {sst_data.shape} (days, lat, lon)")
        print(f"  Temperature range: {sst_data.min():.2f}°C to {sst_data.max():.2f}°C")
        
        return {
            'latitude': lat,
            'longitude': lon,
            'sst': sst_data,
            'dates': dates,
            'units': 'Celsius'
        }
    
    @staticmethod
    def download_noaa_data(date_str, region='global'):
        """
        Attempt to download real NOAA SST data
        
        Args:
            date_str: Date in format 'YYYY-MM-DD'
            region: 'global', 'atlantic', 'pacific'
        
        Returns:
            Downloaded file path or None if failed
        """
        try:
            print(f"\nDownloading NOAA SST data for {date_str}...")
            
            # Parse date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            month = date_obj.month
            day = date_obj.day
            
            # Construct URL
            filename = f"sst.day.mean.{year}{month:02d}{day:02d}.nc"
            url = f"{SSTConfig.NOAA_SST_URL}/{year}/{month:02d}/{filename}"
            
            filepath = os.path.join(SSTConfig.DATA_DIR, filename)
            
            # Download
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
                return filepath
            else:
                print(f"File not found (Status: {response.status_code})")
                return None
                
        except Exception as e:
            print(f"Download failed: {str(e)}")
            return None


# ============================================================================
# PART 4: THERMAL IMAGE PROCESSING
# ============================================================================

class ThermalImageProcessor:
    """Process thermal images to extract temperature data"""
    
    @staticmethod
    def generate_thermal_image(width=640, height=480):
        """
        Generate synthetic thermal image for demonstration
        
        Returns:
            numpy array: Gray16 thermal image (0-65535 range)
        """
        print("\nGenerating synthetic thermal image...")
        
        # Create base thermal pattern
        y, x = np.ogrid[:height, :width]
        
        # Multiple heat sources
        heat1 = 50000 * np.exp(-((x-150)**2 + (y-120)**2) / 5000)
        heat2 = 40000 * np.exp(-((x-500)**2 + (y-350)**2) / 8000)
        heat3 = 35000 * np.exp(-((x-350)**2 + (y-200)**2) / 6000)
        
        # Background gradient
        background = 20000 + (x * 5 + y * 3)
        
        # Combine
        thermal_img = heat1 + heat2 + heat3 + background
        thermal_img = np.clip(thermal_img, 0, 65535).astype(np.uint16)
        
        print(f"Thermal image shape: {thermal_img.shape}")
        print(f"  Pixel value range: {thermal_img.min()} to {thermal_img.max()}")
        
        return thermal_img
    
    @staticmethod
    def thermal_to_temperature(thermal_image, scale=100.0, offset=273.15):
        """
        Convert thermal image (Gray16) to temperature
        
        Formula: Temperature (°C) = (Gray16_value / scale) - offset
        
        Args:
            thermal_image: Thermal image (uint16, 0-65535)
            scale: Scaling factor (default 100)
            offset: Kelvin to Celsius offset (273.15)
        
        Returns:
            numpy array: Temperature in Celsius
        """
        print("\nConverting thermal image to temperature...")
        
        temp_kelvin = thermal_image.astype(np.float32) / scale
        temp_celsius = temp_kelvin - offset
        
        print(f"Temperature range: {temp_celsius.min():.2f}°C to {temp_celsius.max():.2f}°C")
        
        return temp_celsius
    
    @staticmethod
    def detect_hot_spots(temp_image, threshold=25, min_size=100):
        """
        Detect marine heatwave hotspots
        
        Args:
            temp_image: Temperature array
            threshold: Temperature threshold (°C)
            min_size: Minimum hotspot size (pixels)
        
        Returns:
            dict: Hotspot information
        """
        print(f"\nDetecting hotspots (threshold > {threshold}°C)...")
        
        # Binary mask
        hotspot_mask = temp_image > threshold
        
        # Label connected components
        labeled_array, num_features = ndimage.label(hotspot_mask)  # type: ignore
        
        # Get object sizes
        hotspots = []
        for i in range(1, int(num_features) + 1):
            hotspot_region = labeled_array == i
            size = np.sum(hotspot_region)
            
            if size >= min_size:
                temp_in_hotspot = temp_image[hotspot_region]
                hotspots.append({
                    'id': i,
                    'size': size,
                    'max_temp': temp_in_hotspot.max(),
                    'mean_temp': temp_in_hotspot.mean(),
                    'min_temp': temp_in_hotspot.min(),
                    'region': hotspot_region
                })
        
        print(f"Detected {len(hotspots)} significant hotspots")
        
        return {
            'hotspots': hotspots,
            'mask': hotspot_mask,
            'labeled_array': labeled_array,
            'num_features': num_features
        }
    
    @staticmethod
    def detect_cold_spots(temp_image, threshold=10, min_size=100):
        """Detect cold water anomalies"""
        print(f"\nDetecting cold spots (threshold < {threshold}°C)...")
        
        coldspot_mask = temp_image < threshold
        labeled_array, num_features = ndimage.label(coldspot_mask)  # type: ignore
        
        coldspots = []
        for i in range(1, int(num_features) + 1):
            coldspot_region = labeled_array == i
            size = np.sum(coldspot_region)
            
            if size >= min_size:
                temp_in_coldspot = temp_image[coldspot_region]
                coldspots.append({
                    'id': i,
                    'size': size,
                    'min_temp': temp_in_coldspot.min(),
                    'mean_temp': temp_in_coldspot.mean(),
                    'max_temp': temp_in_coldspot.max(),
                    'region': coldspot_region
                })
        
        print(f"Detected {len(coldspots)} significant cold spots")
        
        return {
            'coldspots': coldspots,
            'mask': coldspot_mask,
            'labeled_array': labeled_array,
            'num_features': num_features
        }
    
    @staticmethod
    def calculate_temperature_gradient(temp_image):
        """Calculate temperature gradient (fronts)"""
        print("\nCalculating temperature gradients...")
        
        grad_x = np.gradient(temp_image, axis=1)
        grad_y = np.gradient(temp_image, axis=0)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        print(f"Max gradient: {gradient_magnitude.max():.3f}°C/pixel")
        
        return {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'magnitude': gradient_magnitude,
            'direction': gradient_direction
        }


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

class SSTVisualizer:
    """Create visualizations of SST data"""
    
    @staticmethod
    def plot_temperature_map(lat, lon, sst_data, title="Sea Surface Temperature Map",
                            region=None, save_path=None):
        """
        Plot SST as a 2D map
        
        Args:
            lat: Latitude array
            lon: Longitude array
            sst_data: SST values (2D array)
            title: Plot title
            region: Geographic region to focus on
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Handle 3D data (take first time step)
        if len(sst_data.shape) == 3:
            sst_data = sst_data[0]
        
        # Create meshgrid
        LON, LAT = np.meshgrid(lon, lat)
        
        # Plot
        contourf = ax.contourf(LON, LAT, sst_data, levels=20, cmap=SSTConfig.CMAP_NAME)
        contour = ax.contour(LON, LAT, sst_data, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
        
        cbar = plt.colorbar(contourf, ax=ax, label='Temperature (°C)')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, ax
    
    @staticmethod
    def plot_thermal_image(thermal_data, title="Thermal Image", cmap='hot', 
                          save_path=None):
        """Plot thermal image"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(thermal_data, cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, label='Pixel Value (Gray16)')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax
    
    @staticmethod
    def plot_temperature_distribution(temp_image, title="Temperature Distribution",
                                     save_path=None):
        """Plot histogram of temperatures"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(temp_image.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.grid(True, alpha=0.3)

        # Statistics
        stats_text = f"""
        Mean: {temp_image.mean():.2f}°C
        Std Dev: {temp_image.std():.2f}°C
        Min: {temp_image.min():.2f}°C
        Max: {temp_image.max():.2f}°C
        Median: {np.median(temp_image):.2f}°C
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        ax2.set_title('Temperature Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, (ax1, ax2)
    
    @staticmethod
    def plot_hotspot_detection(temp_image, hotspot_data, title="Hotspot Detection",
                              save_path=None):
        """Visualize detected hotspots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original temperature
        im1 = axes[0, 0].imshow(temp_image, cmap='RdYlBu_r')
        plt.colorbar(im1, ax=axes[0, 0], label='Temp (°C)')
        axes[0, 0].set_title('Temperature Map')
        
        # Hotspot mask
        im2 = axes[0, 1].imshow(hotspot_data['mask'], cmap='Reds')
        axes[0, 1].set_title('Hotspot Mask')
        
        # Labeled hotspots
        im3 = axes[1, 0].imshow(hotspot_data['labeled_array'], cmap='tab20')
        axes[1, 0].set_title(f'Hotspots (N={len(hotspot_data["hotspots"])})')
        
        # Statistics
        stats_text = f"Detected Hotspots: {len(hotspot_data['hotspots'])}\n\n"
        for i, hs in enumerate(hotspot_data['hotspots'][:5]):
            stats_text += f"Hotspot {i+1}:\n"
            stats_text += f"  Size: {hs['size']} pixels\n"
            stats_text += f"  Max Temp: {hs['max_temp']:.2f}°C\n"
            stats_text += f"  Mean Temp: {hs['mean_temp']:.2f}°C\n"
        
        axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Hotspot Statistics')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, axes
    
    @staticmethod
    def plot_time_series(dates, mean_temps, anomalies=None, save_path=None):
        """Plot SST time series"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Mean temperature
        axes[0].plot(dates, mean_temps, marker='o', linewidth=2, markersize=4)
        axes[0].fill_between(dates, mean_temps, alpha=0.3)
        axes[0].set_ylabel('Mean SST (°C)')
        axes[0].set_title('Sea Surface Temperature Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Anomalies
        if anomalies is not None:
            colors = ['red' if x > 0 else 'blue' for x in anomalies]
            axes[1].bar(dates, anomalies, color=colors, alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[1].set_ylabel('Anomaly (°C)')
            axes[1].set_title('SST Anomaly')
            axes[1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, axes
    
    @staticmethod
    def plot_temperature_gradient(temp_image, gradient_data, save_path=None):
        """Visualize temperature gradients (fronts)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original
        im1 = axes[0, 0].imshow(temp_image, cmap='RdYlBu_r')
        plt.colorbar(im1, ax=axes[0, 0], label='Temp (°C)')
        axes[0, 0].set_title('Temperature Map')
        
        # X-gradient
        im2 = axes[0, 1].imshow(gradient_data['grad_x'], cmap='coolwarm')
        plt.colorbar(im2, ax=axes[0, 1], label='dT/dx')
        axes[0, 1].set_title('Zonal Gradient (E-W)')
        
        # Y-gradient
        im3 = axes[1, 0].imshow(gradient_data['grad_y'], cmap='coolwarm')
        plt.colorbar(im3, ax=axes[1, 0], label='dT/dy')
        axes[1, 0].set_title('Meridional Gradient (N-S)')
        
        # Magnitude (fronts)
        im4 = axes[1, 1].imshow(gradient_data['magnitude'], cmap='hot')
        plt.colorbar(im4, ax=axes[1, 1], label='|grad T|')
        axes[1, 1].set_title('Temperature Fronts (Gradient Magnitude)')
        
        plt.suptitle('Temperature Gradient Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, axes


# ============================================================================
# PART 6: ANALYSIS FUNCTIONS
# ============================================================================

class SSTAnalyzer:
    """Analyze SST data"""
    
    @staticmethod
    def calculate_anomalies(sst_data, climatology_period=10):
        """
        Calculate SST anomalies from climatology
        
        Args:
            sst_data: Time series of SST (3D: time, lat, lon)
            climatology_period: Days to average for climatology
        
        Returns:
            numpy array: SST anomalies
        """
        print(f"\nCalculating anomalies (climatology period: {climatology_period} days)...")
        
        # Calculate climatology (mean over all time steps)
        climatology = np.mean(sst_data, axis=0)
        
        # Calculate anomalies
        anomalies = sst_data - climatology
        
        print(f"Anomaly range: {anomalies.min():.2f}°C to {anomalies.max():.2f}°C")
        
        return anomalies
    
    @staticmethod
    def identify_mhw_events(sst_time_series, threshold=25, duration=5):
        """
        Identify Marine Heatwave (MHW) events
        
        Args:
            sst_time_series: Time series of mean SST
            threshold: Temperature threshold (°C)
            duration: Minimum duration (days)
        
        Returns:
            list: MHW events with dates and characteristics
        """
        print(f"\nIdentifying Marine Heatwave events (>{threshold}°C, >{duration} days)...")
        
        above_threshold = sst_time_series > threshold
        
        mhw_events = []
        in_event = False
        event_start: int | None = None
        event_temps = []
        
        for i, is_hot in enumerate(above_threshold):
            if is_hot and not in_event:
                in_event = True
                event_start = i
                event_temps = [sst_time_series[i]]
            elif is_hot and in_event:
                event_temps.append(sst_time_series[i])
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
                'end_day': len(sst_time_series),
                'duration': len(sst_time_series) - (event_start if event_start is not None else 0),
                'max_temp': max(event_temps),
                'mean_temp': np.mean(event_temps),
                'intensity': max(event_temps) - threshold
            })
        
        print(f"Detected {len(mhw_events)} MHW events")
        for i, event in enumerate(mhw_events):
            print(f"  Event {i+1}: Days {event['start_day']}-{event['end_day']}, "
                  f"Max {event['max_temp']:.1f}°C")
        
        return mhw_events
    
    @staticmethod
    def calculate_sst_statistics(sst_data, region=None):
        """Calculate comprehensive SST statistics"""
        print("\nCalculating SST statistics...")
        
        stats = {
            'mean': np.mean(sst_data),
            'std': np.std(sst_data),
            'min': np.min(sst_data),
            'max': np.max(sst_data),
            'median': np.median(sst_data),
            'q25': np.percentile(sst_data, 25),
            'q75': np.percentile(sst_data, 75)
        }
        
        print(f"Mean: {stats['mean']:.2f}°C ± {stats['std']:.2f}°C")
        print(f"  Range: {stats['min']:.2f}°C to {stats['max']:.2f}°C")
        
        return stats
    
    @staticmethod
    def calculate_sst_trend(sst_time_series, dates):
        """Calculate SST trend over time"""
        print("\nCalculating SST trend...")
        
        # Convert dates to numeric values
        days_numeric = np.arange(len(sst_time_series))
        
        # Linear regression
        result = linregress(days_numeric, sst_time_series)  # type: ignore
        slope = result[0]  # type: ignore
        intercept = result[1]  # type: ignore
        r_value = result[2]  # type: ignore
        p_value = result[3]  # type: ignore
        std_err = result[4]  # type: ignore
        
        # Trend line
        trend_line = slope * days_numeric + intercept  # type: ignore
        
        print(f"Slope: {slope:.4f}°C/day")
        print(f"  R²: {r_value**2:.4f}")  # type: ignore
        print(f"  p-value: {p_value:.4e}")
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,  # type: ignore
            'p_value': p_value,
            'trend_line': trend_line
        }


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" SEA SURFACE TEMPERATURE DETECTION PROJECT")
    print("="*70)
    
    # Setup
    SSTConfig.setup_directories()
    
    # ==================== DEMONSTRATION 1: SST DATA ====================
    print("\n\n--- DEMONSTRATION 1: SST DATA ANALYSIS ---")
    
    # Get SST data (synthetic for this demo)
    data = SSTDataHandler.generate_sample_sst_data(days=30)
    
    # Calculate statistics
    stats = SSTAnalyzer.calculate_sst_statistics(data['sst'])
    
    # Calculate anomalies
    anomalies = SSTAnalyzer.calculate_anomalies(data['sst'])
    
    # Time series analysis
    sst_mean_series = np.mean(data['sst'], axis=(1, 2))
    mhw_events = SSTAnalyzer.identify_mhw_events(sst_mean_series, threshold=25, duration=3)
    trend = SSTAnalyzer.calculate_sst_trend(sst_mean_series, data['dates'])
    
    # Visualizations
    SSTVisualizer.plot_temperature_map(
        data['latitude'], data['longitude'], data['sst'][0],
        title="Global Sea Surface Temperature",
        save_path=f"{SSTConfig.OUTPUT_DIR}/01_sst_map.png"
    )
    
    SSTVisualizer.plot_time_series(
        data['dates'], sst_mean_series,
        anomalies=sst_mean_series - np.mean(sst_mean_series),
        save_path=f"{SSTConfig.OUTPUT_DIR}/02_sst_timeseries.png"
    )
    
    # ==================== DEMONSTRATION 2: THERMAL IMAGE ====================
    print("\n\n--- DEMONSTRATION 2: THERMAL IMAGE PROCESSING ---")
    
    # Generate thermal image
    thermal_img = ThermalImageProcessor.generate_thermal_image()
    
    # Convert to temperature
    temp_image = ThermalImageProcessor.thermal_to_temperature(thermal_img)
    
    # Detect anomalies
    hotspots = ThermalImageProcessor.detect_hot_spots(temp_image, threshold=15)
    coldspots = ThermalImageProcessor.detect_cold_spots(temp_image, threshold=-250)
    
    # Calculate gradients
    gradients = ThermalImageProcessor.calculate_temperature_gradient(temp_image)
    
    # Visualizations
    SSTVisualizer.plot_thermal_image(
        thermal_img,
        title="Thermal Image (Gray16 Values)",
        save_path=f"{SSTConfig.OUTPUT_DIR}/03_thermal_image.png"
    )
    
    SSTVisualizer.plot_temperature_distribution(
        temp_image,
        title="Thermal Image Temperature",
        save_path=f"{SSTConfig.OUTPUT_DIR}/04_temperature_distribution.png"
    )
    
    SSTVisualizer.plot_hotspot_detection(
        temp_image, hotspots,
        save_path=f"{SSTConfig.OUTPUT_DIR}/05_hotspot_detection.png"
    )
    
    SSTVisualizer.plot_temperature_gradient(
        temp_image, gradients,
        save_path=f"{SSTConfig.OUTPUT_DIR}/06_temperature_gradients.png"
    )
    
    # ==================== SUMMARY REPORT ====================
    print("\n\n" + "="*70)
    print(" ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    print(f"\nSST STATISTICS:")
    print(f"  Mean Temperature: {stats['mean']:.2f}°C")
    print(f"  Std Deviation: {stats['std']:.2f}°C")
    print(f"  Range: {stats['min']:.2f}°C - {stats['max']:.2f}°C")
    
    print(f"\nTREND ANALYSIS:")
    print(f"  Trend: {trend['slope']:.6f}°C/day")
    print(f"  Total change (30 days): {trend['slope']*30:.2f}°C")
    print(f"  R^2: {trend['r_squared']:.4f}")
    
    print(f"\nMARINE HEATWAVE EVENTS:")
    print(f"  Total events detected: {len(mhw_events)}")
    
    print(f"\nTHERMAL ANOMALIES:")
    print(f"  Hotspots detected: {len(hotspots['hotspots'])}")
    if hotspots['hotspots']:
        print(f"  Largest hotspot: {max(hs['size'] for hs in hotspots['hotspots'])} pixels")
        print(f"  Highest temperature: {max(hs['max_temp'] for hs in hotspots['hotspots']):.2f}°C")
    
    print(f"\nOutput files saved to: {SSTConfig.OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE!")
    print("="*70 + "\n")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
    # Keep plots open
    plt.show()
