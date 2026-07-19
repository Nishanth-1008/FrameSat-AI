import os
import json
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import boto3
# pyrefly: ignore [missing-import]
from botocore import UNSIGNED
# pyrefly: ignore [missing-import]
from botocore.client import Config
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

class GOES19Downloader:
    def __init__(self, cache_dir='datasets/cache/goes19_cache', product='ABI-L1b-RadC', channel=13):
        self.cache_dir = cache_dir
        self.product = product
        self.channel = channel
        self.bucket_name = 'noaa-goes19'
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Public S3 bucket requires unsigned requests
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def get_s3_prefix(self, dt):
        """Returns the prefix for a specific hour: Product/Year/DayOfYear/Hour/"""
        year = dt.strftime('%Y')
        doy = dt.strftime('%j')
        hour = dt.strftime('%H')
        return f"{self.product}/{year}/{doy}/{hour}/"

    def download_range(self, start_date, end_date):
        """Downloads GOES-19 NetCDF files for the given date range (hourly steps)."""
        downloaded_files = []
        current_date = start_date
        
        while current_date <= end_date:
            prefix = self.get_s3_prefix(current_date)
            try:
                # Paginate through the bucket with the prefix
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                found_any = False
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Check if it's the correct channel
                        # e.g. OR_ABI-L1b-RadC-M6C13_G19_s2024222...nc
                        if f"C{self.channel:02d}" in key and key.endswith('.nc'):
                            filename = os.path.basename(key)
                            local_path = os.path.join(self.cache_dir, filename)
                            
                            if not os.path.exists(local_path):
                                print(f"Downloading {filename}...")
                                self.s3_client.download_file(self.bucket_name, key, local_path)
                            downloaded_files.append(local_path)
                            found_any = True
                if not found_any:
                    raise RuntimeError("No files found on S3 for prefix: " + prefix)
            except Exception as e:
                # Offline or bucket access error fallback
                search_str = f"s{current_date.strftime('%Y%j%H')}"
                import glob
                local_matches = glob.glob(os.path.join(self.cache_dir, f"*{search_str}*.nc"))
                chan_str = f"C{self.channel:02d}"
                local_matches = [f for f in local_matches if chan_str in os.path.basename(f)]
                if local_matches:
                    downloaded_files.extend(local_matches)
                else:
                    print(f"Warning: Offline mode fallback could not find local files for {search_str} in cache_dir: {self.cache_dir}")
                        
            current_date += timedelta(hours=1)
            
        return sorted(list(set(downloaded_files)))


import torchvision.transforms.functional as F_t

class GOES19TripletDataset(Dataset):
    """
    PyTorch Dataset for GOES-19 NetCDF satellite data.
    Yields (t0, t1, t2) triplets for frame interpolation training.

    Args:
        metadata_dir: Directory where metadata JSON files are written.
            Resolved in this priority order:
            1. Explicit ``metadata_dir`` argument (if not None).
            2. ``METADATA_DIR`` environment variable.
            3. ``/kaggle/working/goes19_metadata`` when running on Kaggle.
            4. ``cache_dir`` (local fallback).
            This ensures we never write into a read-only source directory.
    """
    def __init__(self, start_date, end_date, cache_dir='datasets/cache/goes19_cache', product='ABI-L1b-RadC', channel=13, split='train', split_ratio=0.8, seed=42, train_resize=None, force_rebuild=False, metadata_dir=None):
        self.downloader = GOES19Downloader(cache_dir=cache_dir, product=product, channel=channel)
        self.train_resize = train_resize
        
        # Check if the cache directory is read-only or in /kaggle/input and resolve a writable tensor cache path
        is_readonly = not os.access(cache_dir, os.W_OK) or "/kaggle/input" in cache_dir
        if is_readonly:
            self.tensor_cache_dir = "/kaggle/working/goes19_tensor_cache"
        else:
            self.tensor_cache_dir = cache_dir
        os.makedirs(self.tensor_cache_dir, exist_ok=True)
        
        
        # 1. Download/Collect files
        self.files = self.downloader.download_range(start_date, end_date)
        
        if len(self.files) < 3:
            raise ValueError(f"Not enough files downloaded to form a triplet. Found {len(self.files)} files.")
            
        # Resolve metadata_dir with explicit priority chain to avoid writing into read-only dirs
        if metadata_dir is not None:
            # 1. Explicit argument takes highest priority
            resolved_metadata_dir = metadata_dir
        elif os.environ.get("METADATA_DIR"):
            # 2. Environment variable (e.g., set by Kaggle notebook before calling train.py)
            resolved_metadata_dir = os.environ["METADATA_DIR"]
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle'):
            # 3. Kaggle auto-detect: always write to /kaggle/working (never /kaggle/input)
            resolved_metadata_dir = "/kaggle/working/goes19_metadata"
        else:
            # 4. Local fallback
            resolved_metadata_dir = cache_dir
            
        os.makedirs(resolved_metadata_dir, exist_ok=True)
        metadata_path = os.path.join(resolved_metadata_dir, f"triplet_metadata_{split}.json")
        
        reused = False
        if not force_rebuild and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    stored = json.load(f)
                if isinstance(stored, dict) and stored.get("files") == self.files:
                    self.triplets = [tuple(t) for t in stored.get("triplets", [])]
                    self.triplet_indices = list(range(len(self.triplets)))
                    reused = True
                    print(f"Reusing metadata from {metadata_path}")
            except Exception as e:
                print(f"Failed to load or validate existing metadata: {e}. Rebuilding...")
                reused = False
                
        if not reused:
            # 2. Extract Triplet Metadata
            self.triplets = []
            for i in range(len(self.files) - 2):
                self.triplets.append((self.files[i], self.files[i+1], self.files[i+2]))
                
            total_triplets = len(self.triplets)
            
            # 3. Deterministic split logic
            np.random.seed(seed)
            indices = np.random.permutation(total_triplets)
            train_size = int(total_triplets * split_ratio)
            
            if split == 'train':
                self.triplet_indices = indices[:train_size]
            elif split == 'val':
                self.triplet_indices = indices[train_size:]
            else:
                raise ValueError("Split must be 'train' or 'val'")
                
            # Save metadata of used triplets for reference along with files list
            metadata = {
                "files": self.files,
                "triplets": [self.triplets[idx] for idx in self.triplet_indices]
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Generated and saved metadata to {metadata_path}")
            
    def __len__(self):
        return len(self.triplet_indices)
        
    def _process_file(self, file_path):
        """Opens NetCDF, extracts Radiance, converts to BT, and normalizes. Caches result."""
        base_name = os.path.basename(file_path)
        resize_str = f"_{self.train_resize[0]}x{self.train_resize[1]}" if self.train_resize else ""
        cache_tensor_path = os.path.join(self.tensor_cache_dir, base_name.replace('.nc', f'{resize_str}.pt'))
        
        if os.path.exists(cache_tensor_path):
            try:
                return torch.load(cache_tensor_path)
            except Exception:
                pass # Fallback to re-processing if corrupted
                
        with xr.open_dataset(file_path, engine='h5netcdf') as ds:
            rad = ds['Rad'].values
            
            # Brightness temperature conversion
            planck_fk1 = ds['planck_fk1'].values
            planck_fk2 = ds['planck_fk2'].values
            planck_bc1 = ds['planck_bc1'].values
            planck_bc2 = ds['planck_bc2'].values
            
            # Avoid divide by zero or log of negative
            valid_mask = rad > 0
            bt = np.zeros_like(rad)
            bt[valid_mask] = (planck_fk2 / np.log((planck_fk1 / rad[valid_mask]) + 1) - planck_bc1) / planck_bc2
            
            # Fill invalid pixels with a default cold temperature (e.g. 180K or max BT of scene)
            bt[~valid_mask] = np.nanmin(bt[valid_mask]) if np.any(valid_mask) else 200.0
            
            # Normalization (e.g., standard min/max normalization to [0,1] for GOES-13 typical BT range 180K-320K)
            bt_min, bt_max = 180.0, 320.0
            bt_norm = np.clip((bt - bt_min) / (bt_max - bt_min), 0, 1)
            
            # Convert to float32 tensor (1, H, W)
            tensor = torch.from_numpy(bt_norm).float().unsqueeze(0)
            
            if self.train_resize is not None:
                tensor = F_t.resize(tensor, self.train_resize, antialias=True)
                
            # Save to cache
            try:
                torch.save(tensor, cache_tensor_path)
            except Exception as e:
                print(f"Warning: Failed to cache tensor {cache_tensor_path}: {e}")
                
            return tensor
            
    def get_metadata(self, idx):
        """Extract metadata (e.g., file paths and timestamps) for a given triplet."""
        real_idx = self.triplet_indices[idx]
        files = self.triplets[real_idx]
        
        timestamps = []
        for f in files:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                # time_coverage_start is usually ISO 8601 string
                t_str = ds.attrs.get('time_coverage_start', '')
                if t_str:
                    try:
                        # e.g., '2024-10-10T21:01:17.1Z'
                        dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                        timestamps.append(dt)
                    except ValueError:
                        timestamps.append(None)
                else:
                    timestamps.append(None)
                    
        return {
            'files': files,
            'timestamps': timestamps
        }

    def __getitem__(self, idx):
        real_idx = self.triplet_indices[idx]
        f0, f1, f2 = self.triplets[real_idx]
        
        t0_tensor = self._process_file(f0)
        t1_tensor = self._process_file(f1)
        t2_tensor = self._process_file(f2)
        
        return t0_tensor, t1_tensor, t2_tensor
