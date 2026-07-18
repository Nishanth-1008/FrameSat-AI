import os
import json
from datetime import datetime, timedelta
import numpy as np
# pyrefly: ignore [missing-import]
import xarray as xr

class GOES19StatsGenerator:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def generate_statistics(self, output_path, subsample_ratio=0.01):
        """
        Gathers scene records from database, computes statistics, samples NetCDFs for 
        BT histogram, and saves results to a JSON file.
        """
        scenes = self.db_manager.get_all_scenes(sorted_chronologically=True)
        if not scenes:
            return None

        total_scenes = len(scenes)
        
        # Calculate time coverage
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in scenes]
        time_coverage = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }

        # Cadence and gaps analysis
        intervals = []
        missing_timestamps = []
        cadence_counts = {}
        
        # Calculate median cadence
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                diff_min = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
                intervals.append(diff_min)
                
            median_cadence = float(np.median(intervals))
            
            # Identify missing gaps (any gap > 1.5 * median_cadence)
            for i in range(1, len(timestamps)):
                diff_min = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
                if diff_min > 1.5 * median_cadence:
                    # Estimate missing timestamps
                    num_missing = int(round(diff_min / median_cadence)) - 1
                    for step in range(1, num_missing + 1):
                        missing_time = timestamps[i-1] + timedelta(minutes=step * median_cadence)
                        missing_timestamps.append(missing_time.isoformat())
            
            # Cadence histogram bins
            for val in intervals:
                bin_val = round(val, 1)
                cadence_counts[str(bin_val)] = cadence_counts.get(str(bin_val), 0) + 1
        else:
            median_cadence = 0.0

        # BT Histogram calculation by sampling
        bt_values = []
        resolutions = set()
        
        # Determine how many files to sample (max 10 for performance)
        sample_size = min(10, len(scenes))
        sample_indices = np.linspace(0, len(scenes) - 1, sample_size, dtype=int)
        
        for idx in sample_indices:
            scene = scenes[idx]
            filepath = scene['filepath']
            if os.path.exists(filepath):
                try:
                    with xr.open_dataset(filepath, engine='h5netcdf') as ds:
                        rad = ds['Rad'].values
                        resolutions.add(f"{rad.shape[0]}x{rad.shape[1]}")
                        
                        planck_fk1 = ds['planck_fk1'].values
                        planck_fk2 = ds['planck_fk2'].values
                        planck_bc1 = ds['planck_bc1'].values
                        planck_bc2 = ds['planck_bc2'].values
                        
                        # BT conversion
                        valid_mask = rad > 0
                        bt = np.zeros_like(rad)
                        bt[valid_mask] = (planck_fk2 / np.log((planck_fk1 / rad[valid_mask]) + 1) - planck_bc1) / planck_bc2
                        
                        # Flatten and sample pixels
                        bt_flat = bt[valid_mask].flatten()
                        if len(bt_flat) > 0:
                            num_samples = max(1, int(len(bt_flat) * subsample_ratio))
                            sampled = np.random.choice(bt_flat, size=num_samples, replace=False)
                            bt_values.extend(sampled.tolist())
                except Exception as e:
                    print(f"Error sampling file {os.path.basename(filepath)} for stats: {e}")

        # Bin BT values from 180K to 320K (step of 10K)
        bt_histogram = {}
        if bt_values:
            bt_array = np.array(bt_values)
            bins = np.arange(180, 330, 10)
            hist, bin_edges = np.histogram(bt_array, bins=bins)
            for i in range(len(hist)):
                bin_range = f"{int(bin_edges[i])}K-{int(bin_edges[i+1])}K"
                bt_histogram[bin_range] = int(hist[i])

        # File sizes
        filesizes = [s['filesize'] for s in scenes]
        avg_filesize_mb = (sum(filesizes) / total_scenes) / (1024 * 1024) if total_scenes > 0 else 0
        disk_usage_mb = sum(filesizes) / (1024 * 1024)

        stats = {
            "total_scenes": total_scenes,
            "time_coverage": time_coverage,
            "expected_cadence_minutes": round(median_cadence, 2),
            "missing_timestamps": missing_timestamps,
            "cadence_histogram": cadence_counts,
            "bt_histogram": bt_histogram,
            "resolutions": list(resolutions),
            "average_resolution": list(resolutions)[0] if resolutions else "Unknown",
            "average_filesize_mb": round(avg_filesize_mb, 2),
            "disk_usage_mb": round(disk_usage_mb, 2)
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)

        return stats
