import os
import sys
import json
import torch
import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.trainer.trainer import Trainer
from datasets.providers.goes19.goes19_builder import GOES19TripletDataset

def main():
    config_path = "training/configs/train_rife426.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    print("Setting up Trainer for post-training test...")
    # Use minimal dataset settings for fast initialization
    start_date = datetime.datetime(2024, 10, 10, 21, 0, 0)
    end_date = start_date + datetime.timedelta(hours=1)
    
    dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir=config.get("dataset_path", "datasets/cache/goes19_cache"),
        product='ABI-L1b-RadC',
        channel=config.get("channel", 13),
        split='train',
        split_ratio=1.0,
        train_resize=(384, 384)
    )
    
    loader = DataLoader(dataset, batch_size=1)
    
    # Initialize trainer
    trainer = Trainer(config, loader, loader)
    
    print(f"Trainer initialized in: {trainer.output_dir}")
    print("Faking a completed training state...")
    
    # Fake best.pth using the currently loaded model (which is the pre-trained weights)
    state = {
        'epoch': 0,
        'state_dict': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'best_psnr': 17.5,
        'config': config
    }
    
    best_path = os.path.join(trainer.output_dir, "best.pth")
    torch.save(state, best_path)
    trainer.best_psnr = 17.5
    trainer.best_ssim = 0.65
    trainer.start_epoch = 0
    trainer.total_training_time = 120.0 # fake 2 minutes
    
    # Add a fake history entry so the plot curves work
    trainer.history.append({
        "epoch": 0,
        "train_loss": 0.25,
        "val_loss": 0.24,
        "psnr": 17.5,
        "ssim": 0.65,
        "fsim": 0.70,
        "mse": 0.02,
        "mae": 0.08
    })
    
    print("Running post-training methods...")
    
    trainer._save_environment_info(trainer.total_training_time)
    trainer._evaluate_best_checkpoint()
    trainer._generate_training_report()
    trainer._generate_readme()
    trainer._update_experiment_registry()
    
    print("Done! Post-training evaluation and report generation completed successfully.")

if __name__ == "__main__":
    main()
