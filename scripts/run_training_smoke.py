import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import platform
import logging
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader

# Setup paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
from training.losses.loss import CombinedLoss

class PipelineTracker:
    def __init__(self, logger):
        self.timings = {}
        self.logger = logger
        
    def print_timings(self):
        print("\nStage Timings\n")
        total = 0.0
        for name, elapsed in self.timings.items():
            dots = "." * (26 - len(name))
            print(f"{name} {dots} {elapsed:.2f} s")
            total += elapsed
        
        dots = "." * (26 - len("Total"))
        print(f"\nTotal {dots} {total:.2f} s\n")

class StageTimer:
    def __init__(self, name, tracker):
        self.name = name
        self.tracker = tracker
        self.start = None

    def __enter__(self):
        self.start = time.time()
        dots = "." * (27 - len(self.name))
        print(f"{self.name} {dots} ", end="", flush=True)
        self.tracker.logger.info(f"Starting stage: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        self.tracker.timings[self.name] = elapsed
        if exc_type is None:
            print("✓")
            self.tracker.logger.info(f"Completed stage: {self.name} in {elapsed:.2f}s")
        else:
            print("✗")
            self.tracker.logger.error(f"Failed stage: {self.name} in {elapsed:.2f}s")
            self.tracker.logger.error("Exception:", exc_info=(exc_type, exc_val, exc_tb))

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_smoke.log"
    
    logger = logging.getLogger("TrainingSmoke")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(str(log_path), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # We only log to file, console is strictly the checklist
    logger.addHandler(fh)
    return logger, log_path

def validate_environment(cache_dir: Path, checkpoint_dir: Path, log_dir: Path, logger):
    logger.info("Starting Environment Validation")
    
    print(f"Python Version           {platform.python_version()}")
    print(f"PyTorch Version          {torch.__version__}")
    print(f"CUDA Available           {torch.cuda.is_available()}")
    print(f"CUDA Version             {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Name                 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Selected Device          {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Working Directory        {Path.cwd()}")
    print(f"Dataset Root             {cache_dir.parent}")
    print(f"Metadata Cache Directory {cache_dir}")
    print(f"Checkpoint Directory     {checkpoint_dir}")
    print(f"Log Directory            {log_dir}\n")
    
    logger.info(f"Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    
    for d in [cache_dir, checkpoint_dir, log_dir]:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {d}: {e}")
                print(f"ERROR: Cannot create directory {d}")
                sys.exit(1)
                
        if not os.access(str(d), os.W_OK):
            logger.error(f"Directory not writable: {d}")
            print(f"ERROR: Directory not writable: {d}")
            sys.exit(1)
            
    logger.info("Environment validation successful")

def create_dummy_netcdfs(cache_dir: Path):
    # Only create if they don't exist to allow testing cache reuse
    files = []
    for i in range(3):
        file_path = cache_dir / f"dummy_scene_{i}.nc"
        if not file_path.exists():
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
            ds.to_netcdf(str(file_path), engine="h5netcdf")
        files.append(str(file_path))
    return files

def get_model(logger):
    rife_src_dir = PROJECT_ROOT / "evaluation" / "models" / "rife_src"
    train_log_dir = PROJECT_ROOT / "evaluation" / "weights" / "rife_426" / "train_log"
    
    if str(rife_src_dir) not in sys.path:
        sys.path.insert(0, str(rife_src_dir))
    if str(train_log_dir) not in sys.path:
        sys.path.insert(0, str(train_log_dir))
        
    try:
        from train_log.IFNet_HDv3 import IFNet
        model = IFNet()
        weight_file = train_log_dir / 'flownet.pkl'
        if weight_file.exists():
            logger.info(f"Loading pretrained weights from {weight_file}")
            state_dict = torch.load(str(weight_file), map_location='cpu')
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state, strict=False)
        else:
            logger.warning(f"Pretrained weights not found at {weight_file}. Initializing randomly for smoke test.")
        return model
    except Exception as e:
        logger.error("Failed to load IFNet architecture", exc_info=True)
        raise

def run_smoke_test():
    # Setup configurable paths via environment or fallback
    log_dir = Path(os.getenv("SMOKE_LOG_DIR", str(PROJECT_ROOT / "logs")))
    logger, log_path = setup_logger(log_dir)
    
    smoke_cache = Path(os.getenv("SMOKE_CACHE_DIR", str(PROJECT_ROOT / "backend" / "datasets" / "smoke_cache")))
    smoke_checkpoints = Path(os.getenv("SMOKE_CKPT_DIR", str(PROJECT_ROOT / "backend" / "checkpoints")))
    
    validate_environment(smoke_cache, smoke_checkpoints, log_dir, logger)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tracker = PipelineTracker(logger)
    
    print("================ TRAINING SMOKE TEST ================\n")

    try:
        with StageTimer("Environment", tracker):
            pass

        with StageTimer("Kaggle Path Check", tracker):
            # Simulate Kaggle environment to verify metadata_dir never resolves to a read-only path
            import unittest.mock
            original_environ = os.environ.copy()
            try:
                os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "TEST"
                os.environ.pop("METADATA_DIR", None)  # ensure auto-detect is tested
                # Re-import to get fresh resolution
                import importlib
                import datasets.providers.goes19.goes19_builder as _builder_mod
                importlib.reload(_builder_mod)
                
                # Instantiate dataset with mocked downloader to check path resolution only
                with unittest.mock.patch.object(_builder_mod.GOES19Downloader, "download_range",
                                               return_value=[str(smoke_cache / f"dummy_scene_{i}.nc") for i in range(3)]):
                    ds_test = _builder_mod.GOES19TripletDataset(
                        start_date=datetime(2024, 10, 10, 21, 0, 0),
                        end_date=datetime(2024, 10, 10, 21, 3, 0),
                        cache_dir=str(smoke_cache),
                        split='train',
                        split_ratio=1.0,
                        train_resize=(64, 64),
                        force_rebuild=False,
                    )
                # Verify metadata was written to /kaggle/working, not inside cache_dir
                import glob as _glob
                kaggle_meta = _glob.glob("/kaggle/working/goes19_metadata/*.json")
                # We can't check actual /kaggle path locally; just ensure no write into input-style source
                # The real assertion: METADATA_DIR env var should override if set
                os.environ["METADATA_DIR"] = str(smoke_cache / "kaggle_sim_metadata")
                importlib.reload(_builder_mod)
                with unittest.mock.patch.object(_builder_mod.GOES19Downloader, "download_range",
                                               return_value=[str(smoke_cache / f"dummy_scene_{i}.nc") for i in range(3)]):
                    ds_test2 = _builder_mod.GOES19TripletDataset(
                        start_date=datetime(2024, 10, 10, 21, 0, 0),
                        end_date=datetime(2024, 10, 10, 21, 3, 0),
                        cache_dir=str(smoke_cache),
                        split='train',
                        split_ratio=1.0,
                        train_resize=(64, 64),
                        force_rebuild=True,
                    )
                # Verify metadata landed in METADATA_DIR, not cache_dir
                sim_meta = smoke_cache / "kaggle_sim_metadata" / "triplet_metadata_train.json"
                assert sim_meta.exists(), f"METADATA_DIR env var was not respected: {sim_meta}"
                logger.info("Kaggle path resolution check passed")
            finally:
                # Restore original environment and reload module
                os.environ.clear()
                os.environ.update(original_environ)
                importlib.reload(_builder_mod)

        with StageTimer("Dataset Build", tracker):
            logger.info("Mocking S3 downloads with local dummy NetCDF generation")
            create_dummy_netcdfs(smoke_cache)
            import unittest.mock
            with unittest.mock.patch("datasets.providers.goes19.goes19_builder.GOES19Downloader.download_range") as mock_dl:
                mock_dl.return_value = [str(smoke_cache / f"dummy_scene_{i}.nc") for i in range(3)]
                
                # Check if metadata exists to test reuse instead of force rebuilding always
                metadata_path = smoke_cache / "triplet_metadata_train.json"
                force_rebuild = not metadata_path.exists()
                
                start_date = datetime(2024, 10, 10, 21, 0, 0)
                end_date = datetime(2024, 10, 10, 21, 3, 0)
                
                dataset = GOES19TripletDataset(
                    start_date=start_date,
                    end_date=end_date,
                    cache_dir=str(smoke_cache),
                    product='ABI-L1b-RadC',
                    channel=13,
                    split='train',
                    split_ratio=1.0,
                    train_resize=(128, 128),
                    force_rebuild=force_rebuild
                )
                assert len(dataset) > 0, "Dataset length must be > 0"
                logger.info(f"Dataset built. Length: {len(dataset)}")

        with StageTimer("Metadata Cache", tracker):
            assert metadata_path.exists(), "Metadata cache file was not created"
            logger.info(f"Verified metadata cache exists at {metadata_path}")

        with StageTimer("DataLoader", tracker):
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            logger.info("DataLoader initialized")

        with StageTimer("Batch Loaded", tracker):
            batch = next(iter(dataloader))
            t0, t1, t2 = batch
            assert t0.shape == (1, 1, 128, 128)
            logger.info(f"Batch loaded. Shapes - t0: {t0.shape}, t1: {t1.shape}, t2: {t2.shape}")
            
            t0 = torch.cat([t0, t0, t0], dim=1).to(device)
            t1 = torch.cat([t1, t1, t1], dim=1).to(device)
            t2 = torch.cat([t2, t2, t2], dim=1).to(device)
            imgs = torch.cat((t0, t2), dim=1)
            logger.info(f"Formatted batch for model. imgs shape: {imgs.shape}")

        with StageTimer("Model Init", tracker):
            model = get_model(logger).to(device)
            model.train()
            criterion = CombinedLoss(alpha=0.5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with StageTimer("Forward Pass", tracker):
            from torch.amp import autocast
            use_amp = torch.cuda.is_available()
            with autocast('cuda', enabled=use_amp):
                flow, mask, merged = model(imgs, timestep=0.5, scale_list=[16, 8, 4, 2, 1])
                pred = merged[-1]
                assert pred.shape == t1.shape, "Output shape mismatch"
            logger.info(f"Forward pass complete. Pred shape: {pred.shape}")

        with StageTimer("Loss Computed", tracker):
            loss, loss_l1, loss_ssim = criterion(pred, t1)
            assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"
            logger.info(f"Loss computed: {loss.item():.4f}")

        with StageTimer("Backward Pass", tracker):
            loss.backward()

        with StageTimer("Optimizer Step", tracker):
            optimizer.step()

        with StageTimer("Checkpoint Saved", tracker):
            checkpoint_path = smoke_checkpoints / "smoke_checkpoint.pth"
            state = {
                'epoch': 0,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, str(checkpoint_path))
            assert checkpoint_path.exists(), "Checkpoint was not saved"
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        with StageTimer("Checkpoint Reloaded", tracker):
            loaded_state = torch.load(str(checkpoint_path), map_location='cpu')
            model.load_state_dict(loaded_state['state_dict'])
            assert 'epoch' in loaded_state, "Epoch not in checkpoint"
            logger.info("Checkpoint successfully reloaded")
            
        # Clean up temporary checkpoint unless requested otherwise
        try:
            checkpoint_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temporary checkpoint: {e}")

        print("\n================ SUCCESS =================")
        tracker.print_timings()

    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        print("\n================ FAILED ==================")
        tracker.print_timings()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
