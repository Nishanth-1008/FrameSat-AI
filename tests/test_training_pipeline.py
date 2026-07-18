import os
import json
import pytest
import numpy as np
import torch
from unittest.mock import patch
import xarray as xr
from datetime import datetime

# Adjust path to find backend and training
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset, GOES19Downloader
from training.trainer.trainer import Trainer
from torch.utils.data import DataLoader

class DummyIFNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input is imgs (which is t0_p and t2_p concatenated along dim=1 -> 3 + 3 = 6 channels)
        self.conv = torch.nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, imgs, timestep, scale_list):
        out = torch.sigmoid(self.conv(imgs))
        # Returns flow, mask, merged (we only use merged[-1] in Trainer.train_epoch)
        return None, None, [out]

@pytest.fixture
def dummy_dataset_path(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Create 3 dummy NetCDF files to form exactly 1 triplet
    files = []
    for i in range(3):
        ds = xr.Dataset(
            {
                "Rad": (("y", "x"), np.full((128, 128), 100.0, dtype=np.float32))
            },
            attrs={"time_coverage_start": f"2024-10-10T21:{i:02d}:00Z"}
        )
        ds["planck_fk1"] = np.float32(100.0)
        ds["planck_fk2"] = np.float32(173.286)
        ds["planck_bc1"] = np.float32(0.0)
        ds["planck_bc2"] = np.float32(1.0)
        
        file_path = cache_dir / f"dummy_scene_{i}.nc"
        ds.to_netcdf(file_path, engine="h5netcdf")
        files.append(str(file_path))
        
    return str(cache_dir), files

@patch("datasets.providers.goes19.goes19_builder.GOES19Downloader.download_range")
def test_complete_training_pipeline(mock_download_range, dummy_dataset_path, tmp_path):
    cache_dir, dummy_files = dummy_dataset_path
    
    # Mock the downloader to return our dummy files instead of S3 calls
    mock_download_range.return_value = dummy_files
    
    # 1. Test Dataset Creation and Metadata Caching
    start_date = datetime(2024, 10, 10, 21, 0, 0)
    end_date = datetime(2024, 10, 10, 21, 3, 0)
    
    # Force rebuild to generate metadata.json; explicitly set metadata_dir so
    # the test is immune to KAGGLE_KERNEL_RUN_TYPE being set in the environment
    dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        product='ABI-L1b-RadC',
        channel=13,
        split='train',
        split_ratio=1.0,
        train_resize=(64, 64),
        force_rebuild=True,
        metadata_dir=cache_dir,
    )
    
    # Verify metadata is cached in the expected location
    metadata_path = os.path.join(cache_dir, "triplet_metadata_train.json")
    assert os.path.exists(metadata_path), "Metadata cache was not generated."
    
    # Verify dataset yields correctly
    assert len(dataset) == 1, "Dataset should contain exactly 1 triplet."
    t0, t1, t2 = dataset[0]
    assert t0.shape == (1, 64, 64), "Tensor shape should match train_resize with 1 channel."
    
    # 2. Test DataLoader instantiation
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Setup Trainer configuration
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    
    config = {
        "epochs": 1,
        "learning_rate": 1e-4,
        "loss_alpha": 0.5,
        "pretrained_weights": "dummy_path", # Will be mocked out
        "dataset_path": cache_dir,
        "use_amp": False,
        "log_interval": 1,
        "train_resize": [64, 64]
    }
    
    with patch("training.trainer.trainer.Trainer._load_model") as mock_load_model, \
         patch("training.trainer.trainer.Trainer._evaluate_best_checkpoint") as mock_evaluate, \
         patch("training.trainer.trainer.Trainer._generate_training_report") as mock_report, \
         patch("training.trainer.trainer.current_dir", str(runs_dir)): # Override current_dir so runs go to tmp_path
         
        # Return our dummy model
        mock_model = DummyIFNet()
        mock_load_model.return_value = mock_model
        
        # Instantiate Trainer
        trainer = Trainer(config, train_loader, val_loader)
        trainer.output_dir = str(runs_dir / "Experiment_001")
        os.makedirs(trainer.output_dir, exist_ok=True)
        
        # 4. Test Single Optimization Step
        # Ensure we are in training mode
        mock_model.train()
        
        # Run a single epoch
        train_loss = trainer.train_epoch(0)
        assert train_loss > 0, "Train loss should be a positive value."
        
        # 5. Test validation step
        val_metrics = trainer.validate(0)
        assert "psnr" in val_metrics
        assert "val_loss" in val_metrics
        
        # 6. Test Checkpointing
        trainer._save_checkpoint(is_best=True, epoch=0)
        
        latest_path = os.path.join(trainer.output_dir, "latest.pth")
        best_path = os.path.join(trainer.output_dir, "best.pth")
        
        assert os.path.exists(latest_path), "latest.pth checkpoint was not saved."
        assert os.path.exists(best_path), "best.pth checkpoint was not saved."
        
        # Verify checkpoint contents
        checkpoint = torch.load(latest_path, map_location='cpu')
        assert checkpoint['epoch'] == 0
        assert 'state_dict' in checkpoint
        assert 'optimizer' in checkpoint
