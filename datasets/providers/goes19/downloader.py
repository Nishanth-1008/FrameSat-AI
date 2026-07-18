import os
import time
import math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
# pyrefly: ignore [missing-import]
import boto3
# pyrefly: ignore [missing-import]
from botocore import UNSIGNED
# pyrefly: ignore [missing-import]
from botocore.client import Config
# pyrefly: ignore [missing-import]
from tqdm import tqdm

class GOES19Downloader:
    def __init__(self, cache_dir='datasets/cache/goes19_cache', sector='CONUS', channel=13, workers=8, resume=True, overwrite=False):
        self.cache_dir = cache_dir
        self.sector = sector
        self.channel = channel
        self.workers = workers
        self.resume = resume
        self.overwrite = overwrite
        self.bucket_name = 'noaa-goes19'
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup anonymous S3 client
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        # Determine S3 prefix and filter based on sector
        self.s3_prefix_product, self.filter_string = self._resolve_sector_prefix_filter()
        
        # Stats tracking
        self.total_downloaded_bytes = 0
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.skipped_downloads = 0

    def _resolve_sector_prefix_filter(self):
        if self.sector == "CONUS":
            return "ABI-L1b-RadC", f"C{self.channel:02d}"
        elif self.sector == "Full Disk":
            return "ABI-L1b-RadF", f"C{self.channel:02d}"
        elif self.sector == "Mesoscale 1":
            return "ABI-L1b-RadM", f"RadM1-M6C{self.channel:02d}"
        elif self.sector == "Mesoscale 2":
            return "ABI-L1b-RadM", f"RadM2-M6C{self.channel:02d}"
        else:
            raise ValueError(f"Unsupported sector: {self.sector}")

    def get_s3_prefix(self, dt):
        """Get S3 prefix for the given hour: Product/Year/DayOfYear/Hour/"""
        year = dt.strftime('%Y')
        doy = dt.strftime('%j')
        hour = dt.strftime('%H')
        return f"{self.s3_prefix_product}/{year}/{doy}/{hour}/"

    def list_files_for_hour(self, dt):
        """List and filter NetCDF keys in the S3 bucket for a specific hour."""
        prefix = self.get_s3_prefix(dt)
        keys_and_sizes = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.nc') and self.filter_string in key:
                        keys_and_sizes.append((key, obj['Size']))
        except Exception as e:
            print(f"Error listing S3 objects for prefix {prefix}: {str(e)}")
            
        return keys_and_sizes

    def download_file(self, key, expected_size, retries=3):
        """Download a single S3 key to local cache with retry logic."""
        filename = os.path.basename(key)
        local_path = os.path.join(self.cache_dir, filename)
        
        # Resume / Skip check
        if self.resume and not self.overwrite and os.path.exists(local_path):
            if os.path.getsize(local_path) == expected_size:
                self.skipped_downloads += 1
                return local_path, "skipped"
                
        # Download with retry logic
        attempt = 0
        backoff = 1.0
        while attempt < retries:
            try:
                self.s3_client.download_file(self.bucket_name, key, local_path)
                # Verify downloaded size
                if os.path.getsize(local_path) == expected_size:
                    self.successful_downloads += 1
                    self.total_downloaded_bytes += expected_size
                    return local_path, "downloaded"
                else:
                    raise IOError(f"Filesize mismatch for {filename}. Expected: {expected_size}, Got: {os.path.getsize(local_path)}")
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    self.failed_downloads += 1
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except:
                            pass
                    return None, f"failed after {retries} attempts: {str(e)}"
                time.sleep(backoff)
                backoff *= 2.0  # exponential backoff

    def download_files_concurrently(self, keys_and_sizes):
        """Download a list of (key, size) tuples concurrently, displaying progress."""
        if not keys_and_sizes:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Map futures to keys
            future_to_key = {
                executor.submit(self.download_file, key, size): (key, size)
                for key, size in keys_and_sizes
            }
            
            # Setup progress bar
            total_size_bytes = sum(size for _, size in keys_and_sizes)
            pbar = tqdm(
                total=total_size_bytes, 
                unit='B', 
                unit_scale=True, 
                desc=f"Downloading {self.sector} Ch{self.channel}"
            )
            
            for future in as_completed(future_to_key):
                key, size = future_to_key[future]
                try:
                    local_path, status = future.result()
                    if local_path:
                        results.append((local_path, status, key))
                    # Update progress bar by the file size regardless of status
                    pbar.update(size)
                except Exception as exc:
                    print(f"Exception during download of {key}: {exc}")
                    pbar.update(size)
                    
            pbar.close()
            
        return results

    def get_statistics(self, duration_seconds):
        """Return download statistics summary."""
        avg_speed_mb = (self.total_downloaded_bytes / (1024 * 1024)) / duration_seconds if duration_seconds > 0 else 0
        return {
            "total_bytes_downloaded": self.total_downloaded_bytes,
            "total_files_downloaded": self.successful_downloads,
            "total_files_skipped": self.skipped_downloads,
            "total_files_failed": self.failed_downloads,
            "duration_seconds": round(duration_seconds, 2),
            "average_speed_mbps": round(avg_speed_mb, 2)
        }
