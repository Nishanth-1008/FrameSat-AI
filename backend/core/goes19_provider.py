import os
import re
import glob
import datetime
import numpy as np
import xarray as xr

class GOES19Provider:
    """
    Data provider for GOES-19 Channel 13 geostationary satellite files.
    Reads NetCDF4 (.nc) files, extracts radiance, and applies Planck conversions.
    Falls back to generating synthetic/mock scenes for local testing when no files are present.
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join("datasets", "raw", "goes19")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.scene_ids = []
        self.nc_files = []
        self.scan_times = []
        
        self._scan_directory()
        
    def _scan_directory(self):
        """Scans the directory for GOES-19 .nc files and parses timestamps."""
        pattern = os.path.join(self.data_dir, "*.nc")
        self.nc_files = sorted(glob.glob(pattern))
        
        # Parse timestamps from filenames (ABI standard format contains sYYYYJJJHHMMSS)
        for filepath in self.nc_files:
            filename = os.path.basename(filepath)
            # Find start time parameter: s\d{14}
            match = re.search(r'_s(\d{14})', filename)
            if match:
                time_str = match.group(1)[:13]
                dt = datetime.datetime.strptime(time_str, "%Y%j%H%M%S")
                self.scan_times.append(dt)
                
        # Scene IDs are named based on scan start times
        self.scene_ids = [dt.strftime("G19_ABI_%Y%m%d_%H%M") for dt in self.scan_times]
        
        # Fallback setup if no NetCDF files are present
        if not self.scene_ids:
            # Create dummy scenes for validation testing
            self.scene_ids = ["G19_ABI_20260715_1200", "G19_ABI_20260715_1215", "G19_ABI_20260715_1230"]
            print(f"[GOES19Provider] No NetCDF files found in {self.data_dir}. Falling back to 3 mock scenes.")

    def _load_nc_radiance_to_temp(self, filepath: str) -> np.ndarray:
        """
        Loads raw radiance from NetCDF and converts it to Brightness Temperature (Kelvin).
        """
        with xr.open_dataset(filepath) as ds:
            rad = ds['Rad'].values
            
            # Planck constants
            fk1 = float(ds['planck_fk1'].values) if 'planck_fk1' in ds else 3.14e5
            fk2 = float(ds['planck_fk2'].values) if 'planck_fk2' in ds else 1.34e3
            bc1 = float(ds['planck_bc1'].values) if 'planck_bc1' in ds else 0.5
            bc2 = float(ds['planck_bc2'].values) if 'planck_bc2' in ds else 1.002
            
            # Clean zero or negative values
            rad_clean = np.maximum(rad, 1e-4)
            # Convert to Kelvin: Tb = fk2 / ln(fk1/Rad + 1) - bc1
            temp_k = fk2 / np.log((fk1 / rad_clean) + 1.0) - bc1
            
        return temp_k

    def load_scene(self, scene_id: str) -> np.ndarray:
        """
        Loads a full scene array. 
        Returns shape [H, W, Frames] (e.g. 384x384x49).
        """
        if scene_id in self.scene_ids and self.nc_files:
            idx = self.scene_ids.index(scene_id)
            target_filepath = self.nc_files[idx]
            # Since a single NetCDF file represents one scan, we simulate a sequence
            # by adding duplicate/simulated frames around it to match the shape.
            single_frame = self._load_nc_radiance_to_temp(target_filepath)
            # Resize frame to target 384x384 if needed
            import cv2
            if single_frame.shape != (384, 384):
                single_frame = cv2.resize(single_frame, (384, 384), interpolation=cv2.INTER_LINEAR)
            # Expand to 49 frames sequence
            seq = np.repeat(single_frame[:, :, np.newaxis], 49, axis=2)
            return seq
        else:
            # Generate simulated scene (warm background with moving cold cloud tops)
            np.random.seed(42)
            bg = np.full((384, 384, 49), 290.0) # 290 Kelvin warm ground
            # Add synthetic moving storm circle
            for t in range(49):
                cx = 100 + t * 4
                cy = 150 + int(np.sin(t/5.0) * 20)
                y, x = np.ogrid[:384, :384]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                # Cold storm top (210 Kelvin)
                mask = dist < 60
                bg[mask, t] = 210.0 + dist[mask] * 0.5
            return bg

    def get_triplet(self, scene_id: str, t0_idx: int = 23, t1_idx: int = 24, t2_idx: int = 25) -> tuple:
        """
        Returns a frame triplet (t0, t1, t2) from the scene.
        """
        scene_data = self.load_scene(scene_id)
        return (
            scene_data[:, :, t0_idx],
            scene_data[:, :, t1_idx],
            scene_data[:, :, t2_idx]
        )

    def get_scene_metadata(self, scene_id: str) -> dict:
        """
        Returns scene metadata parameters.
        """
        scene_data = self.load_scene(scene_id)
        return {
            "scene_id": scene_id,
            "modality": "goes_c13",
            "min_raw_val": float(scene_data.min()), # Kelvin limits
            "max_raw_val": float(scene_data.max()),
            "shape": scene_data.shape,
            "frames": scene_data.shape[2]
        }
