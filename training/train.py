import os
import sys
import json
import argparse
import random
import glob
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
from training.trainer.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Practical-RIFE 4.26 on GOES-19 Satellite Data")
    parser.add_argument("--config", type=str, default="configs/train_rife426.json", help="Path to training config")
    args = parser.parse_args()
    
    config_path = os.path.join(current_dir, args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config.get("seed", 42))
    
    # 1. Automatically detect Kaggle and override paths
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')
    cache_dir = os.path.abspath(os.path.join(current_dir, "..", config.get("dataset_path", "datasets/cache/goes19_cache")))
    quarantine_dir = os.path.abspath(os.path.join(current_dir, "..", config.get("quarantine_dir", "datasets/quarantine/goes19_quarantine")))
    
    # Resolve a writable metadata_dir — env var takes precedence, then Kaggle auto-detect, then local
    metadata_dir = os.environ.get("METADATA_DIR", None)
    
    if is_kaggle:
        print("Kaggle environment detected. Auto-mapping dataset paths...")
        possible_dirs = [
            "/kaggle/input/framesat-goes19-v1/framesat-goes19-v1",
            "/kaggle/input/framesat-goes19-v1",
            "/kaggle/input/framesat-ai-training-bundle/framesat-goes19-v1",
            "/kaggle/input/framesat-ai-training-bundle"
        ]
        mapped = False
        for p_dir in possible_dirs:
            p_cache = os.path.join(p_dir, "cache")
            if os.path.exists(p_cache):
                cache_dir = p_cache
                p_quarantine = os.path.join(p_dir, "quarantine")
                if os.path.exists(p_quarantine):
                    quarantine_dir = p_quarantine
                else:
                    quarantine_dir = "/kaggle/working/datasets/quarantine/goes19_quarantine"
                mapped = True
                print(f"Mapped dataset to Kaggle directory: {p_dir}")
                break
        if not mapped:
            for root, dirs, files in os.walk("/kaggle/input"):
                if "cache" in dirs:
                    cache_dir = os.path.join(root, "cache")
                    if "quarantine" in dirs:
                        quarantine_dir = os.path.join(root, "quarantine")
                    else:
                        quarantine_dir = "/kaggle/working/datasets/quarantine/goes19_quarantine"
                    print(f"Mapped dataset via search to: {root}")
                    break
        # On Kaggle, force metadata into /kaggle/working if not already set by env var
        if metadata_dir is None:
            metadata_dir = "/kaggle/working/goes19_metadata"
    
    config["dataset_path"] = cache_dir
    config["quarantine_dir"] = quarantine_dir
    
    # 2. Phase 1 — Dataset Audit
    valid_files = sorted(glob.glob(os.path.join(cache_dir, "*.nc")))
    num_valid = len(valid_files)
    
    num_rejected = 0
    if os.path.exists(quarantine_dir):
        num_rejected = len(glob.glob(os.path.join(quarantine_dir, "*.nc")))
        
    stats_path = os.path.join(os.path.dirname(cache_dir), "dataset_statistics.json")
    if num_rejected == 0 and os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                total_scenes = stats.get("total_scenes", 0)
                if total_scenes > num_valid:
                    num_rejected = total_scenes - num_valid
        except Exception:
            pass
            
    num_scenes = num_valid + num_rejected
    num_triplets = max(0, num_valid - 2)
    channel = config.get("channel", 13)
    sector = config.get("sector", "CONUS")
    
    # Resolve original spatial resolution and check for corrupted metadata
    orig_resolution = "Unknown"
    if num_valid > 0:
        sample_file = valid_files[0]
        try:
            with xr.open_dataset(sample_file, engine='h5netcdf') as ds:
                # Corrupted metadata checks
                for coeff in ['planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2']:
                    if coeff not in ds:
                        raise ValueError(f"Missing Planck coefficient {coeff}")
                if 'Rad' not in ds:
                     raise ValueError("Missing Rad variable")
                rad_shape = ds['Rad'].shape
                orig_resolution = f"{rad_shape[0]} x {rad_shape[1]}"
        except Exception as e:
            print(f"Abort training: Corrupted metadata check failed for {sample_file}. Error: {e}")
            sys.exit(1)
            
    train_resize = config.get("train_resize", [384, 384])
    total_bytes = sum(os.path.getsize(f) for f in valid_files)
    dataset_size_gb = total_bytes / (1024 ** 3)
    
    print("==================================================")
    print("\nGOES-19 DATASET SUMMARY\n")
    print(f"Scenes:\n{num_scenes}\n")
    print(f"Valid scenes:\n{num_valid}\n")
    print(f"Rejected:\n{num_rejected}\n")
    print(f"Triplets:\n{num_triplets}\n")
    print(f"Channel:\n{channel}\n")
    print(f"Sector:\n{sector}\n")
    print(f"Resolution:\n{orig_resolution}\n")
    print(f"Training Resolution:\n{train_resize[0]} x {train_resize[1]}\n")
    print(f"Dataset Size:\n{dataset_size_gb:.2f} GB\n")
    print("==================================================")
    
    # Hard Abort Conditions
    if num_triplets == 0:
        print("Abort training: zero triplets found!")
        sys.exit(1)
        
    # Check for missing weights
    pretrained_weights = config.get("pretrained_weights", "")
    train_log_dir = os.path.abspath(os.path.join(current_dir, "..", pretrained_weights))
    weight_file = os.path.join(train_log_dir, 'flownet.pkl')
    if not os.path.exists(weight_file):
        print(f"Abort training: Pretrained weights not found at {weight_file}")
        sys.exit(1)
        
    # Load GOES-19 Dataset
    start_date = datetime.fromisoformat(config.get("start_date", "2024-10-10T21:00:00"))
    end_date = datetime.fromisoformat(config.get("end_date", "2024-10-14T09:00:00"))
    split_ratio = config.get("split_ratio", 0.8)
    seed = config.get("seed", 42)
    
    train_dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        product='ABI-L1b-RadC',
        channel=channel,
        split='train',
        split_ratio=split_ratio,
        seed=seed,
        train_resize=tuple(train_resize),
        metadata_dir=metadata_dir,
    )
    val_dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        product='ABI-L1b-RadC',
        channel=channel,
        split='val',
        split_ratio=split_ratio,
        seed=seed,
        train_resize=tuple(train_resize),
        metadata_dir=metadata_dir,
    )
    
    # Optional subset cap for lightweight first runs (max_triplets in config)
    max_triplets = config.get("max_triplets", None)
    if max_triplets is not None and max_triplets > 0:
        from torch.utils.data import Subset
        import random as _rnd
        _rnd.seed(seed)
        train_indices = _rnd.sample(range(len(train_dataset)), min(max_triplets, len(train_dataset)))
        val_indices = _rnd.sample(range(len(val_dataset)), min(max(1, max_triplets // 5), len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Subset mode active: using {len(train_dataset)} train / {len(val_dataset)} val triplets.")
    
    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Train Dataset: {len(train_dataset)} triplets")
    print(f"Val Dataset: {len(val_dataset)} triplets")
    
    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()

if __name__ == "__main__":
    main()
