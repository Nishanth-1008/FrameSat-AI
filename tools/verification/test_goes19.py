import os
import sys
from datetime import datetime, timedelta
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset, GOES19Downloader

def test_goes19_dataset():
    # Pick a date where GOES-19 is known to have data
    # GOES-U (19) was launched June 25, 2024. Data might be available late 2024/2025.
    # Let's dynamically find a recent date that has data to avoid hardcoding a missing date.
    
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'noaa-goes19'
    product = 'ABI-L1b-RadC'
    
    # Try finding an object from the root to extract a valid year/doy/hour
    print("Finding a valid date with GOES-19 data...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{product}/", MaxKeys=100)
    
    valid_key = None
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'C13' in obj['Key'] and obj['Key'].endswith('.nc'):
                    valid_key = obj['Key']
                    break
        if valid_key:
            break
            
    if not valid_key:
        print("Could not find any C13 data in the bucket!")
        return
        
    print(f"Found sample key: {valid_key}")
    # Key format: ABI-L1b-RadC/2025/123/12/...
    parts = valid_key.split('/')
    year = int(parts[1])
    doy = int(parts[2])
    hour = int(parts[3])
    
    start_date = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour)
    # We need at least 3 files for a triplet, so let's span 3 hours just in case
    end_date = start_date + timedelta(hours=3)
    
    print(f"Testing dataset with date range: {start_date} to {end_date}")
    
    dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir='datasets/goes19_test_cache',
        product=product,
        channel=13,
        split='train',
        split_ratio=1.0 # Use all for train
    )
    
    print(f"Number of triplets formed: {len(dataset)}")
    if len(dataset) == 0:
        print("Warning: No triplets were formed.")
        return
        
    # Get the first item
    t0, t1, t2 = dataset[0]
    
    print(f"t0 shape: {t0.shape}, dtype: {t0.dtype}")
    print(f"t1 shape: {t1.shape}, dtype: {t1.dtype}")
    print(f"t2 shape: {t2.shape}, dtype: {t2.dtype}")
    
    assert t0.shape == t1.shape == t2.shape
    assert len(t0.shape) == 3 # (1, H, W)
    assert t0.dtype == torch.float32
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_goes19_dataset()
