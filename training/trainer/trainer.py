import os
import sys
import json
import csv
import time
import datetime
import subprocess
import random
import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from evaluation.metrics.metrics import compute_psnr, compute_ssim, compute_fsim, compute_mse, compute_mae
from training.losses.loss import CombinedLoss

class TeeLogger:
    def __init__(self, filename, terminal):
        self.terminal = terminal
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if '\r' in message:
            return
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_git_commit_hash():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).strip().decode('utf-8')
        return commit
    except Exception:
        return "unknown"

class Trainer:
    def __init__(self, config, train_loader, val_loader, train_sampler=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        self.epochs = config.get("epochs", 10)
        self.lr = config.get("learning_rate", 1e-4)
        self.alpha = config.get("loss_alpha", 0.5)
        self.weights_path = config.get("pretrained_weights", "")
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.lr_scheduler_type = config.get("lr_scheduler", "cosine")
        
        # Resolve Experiment ID and output folder
        config_out = self.config.get("output_dir")
        if config_out:
            runs_dir = os.path.join(config_out, "runs")
        else:
            is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')
            if is_kaggle:
                runs_dir = "/kaggle/working/artifacts/training/runs"
            else:
                runs_dir = os.path.abspath(os.path.join(current_dir, "runs"))
        os.makedirs(runs_dir, exist_ok=True)
        
        if config.get("resume", False):
            # If resume is True, find the highest existing Experiment_XXX folder to continue
            existing_exps = []
            for d in os.listdir(runs_dir):
                if d.startswith("Experiment_") and os.path.isdir(os.path.join(runs_dir, d)):
                    try:
                        idx = int(d.split("_")[1])
                        existing_exps.append(idx)
                    except ValueError:
                        pass
            best_idx = max(existing_exps) if existing_exps else 1
            self.run_id = f"Experiment_{best_idx:03d}"
        else:
            # Create a new Experiment_XXX folder
            existing_exps = []
            for d in os.listdir(runs_dir):
                if d.startswith("Experiment_") and os.path.isdir(os.path.join(runs_dir, d)):
                    try:
                        idx = int(d.split("_")[1])
                        existing_exps.append(idx)
                    except ValueError:
                        pass
            next_idx = max(existing_exps) + 1 if existing_exps else 1
            self.run_id = f"Experiment_{next_idx:03d}"
            
        self.output_dir = os.path.join(runs_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup directories
        self.sample_predictions_dir = os.path.join(self.output_dir, "sample_predictions")
        os.makedirs(self.sample_predictions_dir, exist_ok=True)
        
        is_rank0 = not self.config.get("is_distributed", False) or self.config.get("rank", 0) == 0
        
        # Setup Tee Logger (Only on Rank 0)
        if is_rank0:
            log_file = os.path.join(self.output_dir, "training.log")
            sys.stdout = TeeLogger(log_file, sys.stdout)
            sys.stderr = TeeLogger(log_file, sys.stderr)
            
        self.git_commit = get_git_commit_hash()
        self.config["git_commit"] = self.git_commit
        
        if is_rank0:
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(self.config, f, indent=4)
            self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        else:
            class MockSummaryWriter:
                def __init__(self, *args, **kwargs): pass
                def add_scalar(self, *args, **kwargs): pass
                def add_image(self, *args, **kwargs): pass
                def close(self): pass
            self.writer = MockSummaryWriter()
        
        self.model = self._load_model(self.weights_path)
        self.model.to(self.device)
        
        if self.device.type == "cuda":
            assert next(self.model.parameters()).is_cuda, "Model failed to move to CUDA! Silent CPU fallback detected."
            if self.config.get("is_distributed", False):
                from torch.nn.parallel import DistributedDataParallel as DDP
                local_rank = self.config.get("local_rank", 0)
                print(f"Wrapping model in DistributedDataParallel (local_rank: {local_rank})")
                self.model = DDP(self.model, device_ids=[local_rank])
            elif torch.cuda.device_count() > 1:
                print(f"Pushing limits: Wrapping model in DataParallel for {torch.cuda.device_count()} GPUs!")
                self.model = torch.nn.DataParallel(self.model)

        
        self.criterion = CombinedLoss(alpha=self.alpha)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        if self.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif self.lr_scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
        else:
            self.scheduler = None
        
        self.use_amp = (self.device.type == "cuda") and config.get("use_amp", True)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        
        self.start_epoch = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.patience_counter = 0
        
        # Select 10 random triplet indices for validation visual tracking
        random.seed(config.get("seed", 42))
        num_val_triplets = len(val_loader.dataset)
        if num_val_triplets > 0:
            self.val_vis_indices = set(random.sample(range(num_val_triplets), min(10, num_val_triplets)))
        else:
            self.val_vis_indices = set()
            
        # Automatic recovery / Resume checkpoint load
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(self.config["dataset_path"]), "metadata.db"))
        self._load_checkpoint()
            
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")
        self.json_path = os.path.join(self.output_dir, "metrics.json")
        self.history = []
        
    def _load_model(self, path):
        for mod in list(sys.modules.keys()):
            if mod.startswith("IFNet") or mod.startswith("train_log") or mod.startswith("model"):
                del sys.modules[mod]

        # `path` is already the final, absolute train_log directory — no re-joining.
        train_log_dir = os.path.abspath(path)
        rife_src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "evaluation", "models", "rife_src"))

        print(f"Resolved pretrained weight directory: {train_log_dir}")
        weight_file = os.path.join(train_log_dir, "flownet.pkl")
        print(f"Resolved weight file: {weight_file}")

        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                "Pretrained weights not found.\n"
                f"  Configured pretrained_weights : {path}\n"
                f"  Resolved directory            : {train_log_dir}\n"
                f"  Expected weight file          : {weight_file}\n"
                f"  Directory exists              : {os.path.isdir(train_log_dir)}\n"
                f"  Weight file exists            : False"
            )

        sys.path.insert(0, rife_src_dir)
        sys.path.insert(0, train_log_dir)

        try:
            from train_log.IFNet_HDv3 import IFNet
            model = IFNet()
            state_dict = torch.load(weight_file, map_location="cpu")
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state, strict=False)
            print(f"Loaded pretrained weights from {weight_file}")
            return model
        finally:
            if train_log_dir in sys.path:
                sys.path.remove(train_log_dir)
            if rife_src_dir in sys.path:
                sys.path.remove(rife_src_dir)


    def _pad(self, tensor):
        _, _, h, w = tensor.shape
        tmp_h = max(32, int(32 * np.ceil(h / 32.0)))
        pad_h = tmp_h - h
        tmp_w = max(32, int(32 * np.ceil(w / 32.0)))
        pad_w = tmp_w - w
        if pad_h > 0 or pad_w > 0:
            return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h)), pad_h, pad_w
        return tensor, 0, 0
        
    def train_epoch(self, epoch):
        if self.config.get("is_distributed", False) and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
            
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)
        log_interval = self.config.get("log_interval", 10)
        
        use_tqdm = self.config.get("use_tqdm", False) and sys.stdout.isatty()
        
        if use_tqdm:
            pbar = tqdm(
                self.train_loader,
                desc=f"Train Epoch {epoch}",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                ascii=" ░▒▓█",
                leave=False
            )
            loader = pbar
        else:
            loader = self.train_loader
            print(f"\n--- Epoch {epoch}/{self.epochs} Training ---")
        
        accumulation_steps = self.config.get("accumulation_steps", 1)
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (t0, t1, t2) in enumerate(loader):
            t0 = torch.cat([t0, t0, t0], dim=1).to(self.device, non_blocking=True)
            t1 = torch.cat([t1, t1, t1], dim=1).to(self.device, non_blocking=True)
            t2 = torch.cat([t2, t2, t2], dim=1).to(self.device, non_blocking=True)
            
            t0_p, pad_h, pad_w = self._pad(t0)
            t1_p, _, _ = self._pad(t1)
            t2_p, _, _ = self._pad(t2)
            
            imgs = torch.cat((t0_p, t2_p), dim=1)
            scale_list = [16, 8, 4, 2, 1]
            
            with autocast("cuda", enabled=self.use_amp):
                flow, mask, merged = self.model(imgs, timestep=0.5, scale_list=scale_list)
                pred = merged[-1]
                loss, loss_l1, loss_ssim = self.criterion(pred, t1_p)
                # Scale loss to support gradient accumulation
                loss = loss / accumulation_steps
                
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                
            self.scaler.scale(loss).backward()
            
            # Step only after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get("grad_clip", 1.0))
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += (loss.item() * accumulation_steps)
            
            if use_tqdm:
                if batch_idx % log_interval == 0:
                    postfix = {"Loss": f"{loss.item():.4f}", "L1": f"{loss_l1.item():.4f}", "SSIM": f"{loss_ssim.item():.4f}"}
                    if torch.cuda.is_available():
                        postfix["VRAM"] = f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}MB"
                    pbar.set_postfix(postfix)
            else:
                if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
                    pct = (batch_idx + 1) / total_batches * 100
                    bar_len = 15
                    filled_len = int(bar_len * (batch_idx + 1) // total_batches)
                    bar = '█' * filled_len + '░' * (bar_len - filled_len)
                    vram_str = f" | VRAM: {torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}MB" if torch.cuda.is_available() else ""
                    print(f"Batch {batch_idx+1:03d}/{total_batches:03d} [{bar}] {pct:3.0f}% | Loss: {loss.item():.4f} (L1: {loss_l1.item():.4f}, SSIM: {loss_ssim.item():.4f}){vram_str}")
                    
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        return avg_loss
        
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        sample_count = 0
        total_batches = len(self.val_loader)
        log_interval = self.config.get("log_interval", 10)
        
        all_psnr, all_ssim, all_fsim, all_mse, all_mae = [], [], [], [], []
        
        use_tqdm = self.config.get("use_tqdm", False) and sys.stdout.isatty()
        
        if use_tqdm:
            val_pbar = tqdm(
                self.val_loader,
                desc=f"Val Epoch {epoch}",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                ascii=" ░▒▓█",
                leave=False
            )
            loader = val_pbar
        else:
            loader = self.val_loader
            print(f"--- Epoch {epoch}/{self.epochs} Validation ---")
            
        with torch.no_grad():
            for batch_idx, (t0, t1, t2) in enumerate(loader):
                t0_tensor = torch.cat([t0, t0, t0], dim=1).to(self.device, non_blocking=True)
                t1_tensor = torch.cat([t1, t1, t1], dim=1).to(self.device, non_blocking=True)
                t2_tensor = torch.cat([t2, t2, t2], dim=1).to(self.device, non_blocking=True)
                
                t0_p, pad_h, pad_w = self._pad(t0_tensor)
                t1_p, _, _ = self._pad(t1_tensor)
                t2_p, _, _ = self._pad(t2_tensor)
                
                imgs = torch.cat((t0_p, t2_p), dim=1)
                
                with autocast("cuda", enabled=self.use_amp):
                    flow, mask, merged = self.model(imgs, timestep=0.5, scale_list=[16, 8, 4, 2, 1])
                    pred_p = merged[-1]
                    loss, _, _ = self.criterion(pred_p, t1_p)
                    
                total_loss += loss.item()
                
                if pad_h > 0 or pad_w > 0:
                    pred = pred_p[:, :, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]
                else:
                    pred = pred_p
                    
                # Iterate through batch to record metrics and visualizations
                bs = t0.size(0)
                for b in range(bs):
                    global_idx = sample_count + b
                    
                    pred_single = pred[b]
                    pred_gray = pred_single.mean(dim=0).cpu().numpy()
                    pred_gray = np.clip(pred_gray, 0.0, 1.0)
                    
                    gt_gray = t1[b, 0].cpu().numpy()
                    
                    all_psnr.append(compute_psnr(gt_gray, pred_gray))
                    all_ssim.append(compute_ssim(gt_gray, pred_gray))
                    all_fsim.append(compute_fsim(gt_gray, pred_gray))
                    all_mse.append(compute_mse(gt_gray, pred_gray))
                    all_mae.append(compute_mae(gt_gray, pred_gray))
                    
                    # Phase 4 - Save 10 random triplets validation visualizations
                    if global_idx in self.val_vis_indices:
                        gt_save_path = os.path.join(self.sample_predictions_dir, f"epoch_{epoch:03d}_triplet_{global_idx:02d}_gt.png")
                        pred_save_path = os.path.join(self.sample_predictions_dir, f"epoch_{epoch:03d}_triplet_{global_idx:02d}_pred.png")
                        diff_save_path = os.path.join(self.sample_predictions_dir, f"epoch_{epoch:03d}_triplet_{global_idx:02d}_diff.png")
                        
                        plt.imsave(gt_save_path, gt_gray, cmap='gray', vmin=0.0, vmax=1.0)
                        plt.imsave(pred_save_path, pred_gray, cmap='gray', vmin=0.0, vmax=1.0)
                        diff = np.abs(gt_gray - pred_gray)
                        plt.imsave(diff_save_path, diff, cmap='hot', vmin=0.0, vmax=0.2)
                        
                sample_count += bs
                if use_tqdm:
                    val_pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}", 
                        "Avg PSNR": f"{np.mean(all_psnr):.2f}dB" if all_psnr else "N/A"
                    })
                else:
                    if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
                        pct = (batch_idx + 1) / total_batches * 100
                        bar_len = 15
                        filled_len = int(bar_len * (batch_idx + 1) // total_batches)
                        bar = '█' * filled_len + '░' * (bar_len - filled_len)
                        print(f"Val Batch {batch_idx+1:03d}/{total_batches:03d} [{bar}] {pct:3.0f}% | Loss: {loss.item():.4f} | Avg PSNR: {np.mean(all_psnr):.2f}dB")
                    
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
            "fsim": float(np.mean(all_fsim)),
            "mse": float(np.mean(all_mse)),
            "mae": float(np.mean(all_mae))
        }
        
        self.writer.add_scalar('Loss/val', metrics['val_loss'], epoch)
        self.writer.add_scalar('Metrics/PSNR', metrics['psnr'], epoch)
        self.writer.add_scalar('Metrics/SSIM', metrics['ssim'], epoch)
        
        return metrics
        
    def _save_checkpoint(self, is_best, epoch):
        if self.config.get("is_distributed", False) and self.config.get("rank", 0) != 0:
            return
            
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        latest_path = os.path.join(self.output_dir, "latest.pth")
        torch.save(state, latest_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best.pth")
            torch.save(state, best_path)
            
    def _load_checkpoint(self):
        target_path = os.path.join(self.output_dir, "latest.pth")
        resume_checkpoint = self.config.get("resume_checkpoint", "")
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            target_path = resume_checkpoint
            
        if os.path.exists(target_path):
            state = torch.load(target_path, map_location=self.device)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state['state_dict'])
            else:
                self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.start_epoch = state['epoch'] + 1
            self.best_psnr = state['best_psnr']
            print(f"Resumed from {target_path} at epoch {state['epoch']}")
        else:
            if self.config.get("resume", False):
                print(f"Warning: Could not find checkpoint at {target_path} to resume from.")
            
    def _log_metrics(self, epoch, train_loss, val_metrics):
        if self.config.get("is_distributed", False) and self.config.get("rank", 0) != 0:
            return
            
        metric_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics
        }
        self.history.append(metric_dict)
        
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=4)
            
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metric_dict)
            
    def _update_experiment_registry(self):
        if self.config.get("is_distributed", False) and self.config.get("rank", 0) != 0:
            return
            
        is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')
        if is_kaggle:
            registry_path = "/kaggle/working/experiments.csv"
        else:
            registry_path = os.path.join(current_dir, "experiments.csv")
        file_exists = os.path.isfile(registry_path)
        
        row = {
            "Experiment ID": self.run_id,
            "Model": "Practical-RIFE 4.26",
            "Dataset": os.path.basename(self.config.get("dataset_path", "")),
            "Epochs": self.epochs,
            "Learning Rate": self.lr,
            "Loss Function": f"alpha={self.alpha} L1+SSIM",
            "Best PSNR": f"{self.best_psnr:.4f}",
            "Best SSIM": f"{self.best_ssim:.4f}",
            "Best Checkpoint Path": os.path.join(self.output_dir, "best.pth")
        }
        
        with open(registry_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    def _save_environment_info(self, training_time_sec=None):
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        torch_version = torch.__version__
        
        # Try to read dataset version from dataset_statistics.json
        dataset_version = "GOES-19 v1"
        stats_path = os.path.join(os.path.dirname(self.config.get("dataset_path", "")), "dataset_statistics.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    dataset_version = f"GOES-19 v1 ({stats.get('total_scenes', 1003)} scenes)"
            except Exception:
                pass
                
        git_commit = get_git_commit_hash()
        
        env_info = {
            "gpu_model": gpu_model,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "training_time_seconds": training_time_sec if training_time_sec is not None else 0.0,
            "training_time_formatted": str(datetime.timedelta(seconds=int(training_time_sec))) if training_time_sec is not None else "00:00:00"
        }
        
        with open(os.path.join(self.output_dir, "environment.json"), "w") as f:
            json.dump(env_info, f, indent=4)

    def _plot_curves(self):
        if not self.history:
            return
        epochs = [h["epoch"] for h in self.history]
        train_losses = [h["train_loss"] for h in self.history]
        val_losses = [h["val_loss"] for h in self.history]
        psnrs = [h["psnr"] for h in self.history]
        ssims = [h["ssim"] for h in self.history]
        
        # Plot Loss Curves
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Train Loss", color="blue")
        plt.plot(epochs, val_losses, label="Val Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        
        # Plot Validation Curves (PSNR & SSIM)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("PSNR (dB)", color="blue")
        ax1.plot(epochs, psnrs, color="blue", label="PSNR")
        ax1.tick_params(axis='y', labelcolor="blue")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("SSIM", color="red")
        ax2.plot(epochs, ssims, color="red", label="SSIM")
        ax2.tick_params(axis='y', labelcolor="red")
        
        plt.title("Validation Metrics (PSNR & SSIM)")
        fig.tight_layout()
        plt.grid(True)
        val_curve_path = os.path.join(self.output_dir, "validation_curves.png")
        plt.savefig(val_curve_path)
        plt.close()

    def _evaluate_best_checkpoint(self):
        print("Running Final Evaluation on best checkpoint...")
        best_path = os.path.join(self.output_dir, "best.pth")
        if not os.path.exists(best_path):
            print(f"Warning: Best checkpoint not found at {best_path}. Using latest.pth instead.")
            best_path = os.path.join(self.output_dir, "latest.pth")
            
        if not os.path.exists(best_path):
            print("Error: No checkpoints found for final evaluation!")
            return
            
        temp_dir = os.path.join(self.output_dir, "temp_eval_weights")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Source .py model files from the same pretrained_weights directory used for training.
            # This is already an absolute path — no joining required.
            src_weights_dir = os.path.abspath(self.config.get("pretrained_weights", ""))

            import shutil
            if os.path.exists(src_weights_dir):
                for item in os.listdir(src_weights_dir):
                    if item.endswith(".py"):
                        shutil.copy(os.path.join(src_weights_dir, item), os.path.join(temp_dir, item))
            else:
                print(f"Warning: Pretrained weights source directory not found at {src_weights_dir}")
                
            checkpoint = torch.load(best_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            torch.save(clean_state, os.path.join(temp_dir, "flownet.pkl"))
            
            from models.rife.interpolator import RIFEInterpolator
            model = RIFEInterpolator()
            model.load_weights(temp_dir)
            
            from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
            start_date = datetime.datetime(2024, 10, 10, 21, 0, 0)
            end_date = start_date + datetime.timedelta(hours=3)
            cache_dir = self.config.get("dataset_path", "datasets/cache/goes19_cache")
            
            eval_dataset = GOES19TripletDataset(
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                product='ABI-L1b-RadC',
                channel=self.config.get("channel", 13),
                split='train',
                split_ratio=1.0,
                train_resize=(512, 512)
            )
            
            total = len(eval_dataset)
            print(f"Evaluating {total} triplets at 512x512 resolution...")
            
            eval_results = []
            for idx in range(total):
                t0, t1, t2 = eval_dataset[idx]
                t0_np = t0.squeeze(0).numpy()
                t1_np = t1.squeeze(0).numpy()
                t2_np = t2.squeeze(0).numpy()
                
                start_time = time.time()
                pred_np = model.interpolate(t0_np, t2_np)
                runtime_ms = (time.time() - start_time) * 1000.0
                
                psnr = compute_psnr(t1_np, pred_np)
                ssim = compute_ssim(t1_np, pred_np)
                fsim = compute_fsim(t1_np, pred_np)
                mse_val = compute_mse(t1_np, pred_np)
                mae_val = compute_mae(t1_np, pred_np)
                
                eval_results.append({
                    "psnr": psnr,
                    "ssim": ssim,
                    "fsim": fsim,
                    "mse": mse_val,
                    "mae": mae_val,
                    "runtime_ms": runtime_ms
                })
                
            mean_psnr = float(np.mean([r["psnr"] for r in eval_results]))
            mean_ssim = float(np.mean([r["ssim"] for r in eval_results]))
            mean_fsim = float(np.mean([r["fsim"] for r in eval_results]))
            mean_mse = float(np.mean([r["mse"] for r in eval_results]))
            mean_mae = float(np.mean([r["mae"] for r in eval_results]))
            mean_runtime = float(np.mean([r["runtime_ms"] for r in eval_results]))
            
            self.final_metrics = {
                "psnr": mean_psnr,
                "ssim": mean_ssim,
                "fsim": mean_fsim,
                "mse": mean_mse,
                "mae": mean_mae,
                "runtime_ms": mean_runtime
            }
            
            baseline_json_path = os.path.abspath(os.path.join(current_dir, "..", "evaluation", "reports", "baseline_benchmark", "baseline_report.json"))
            baseline = {
                "psnr": 17.1855,
                "ssim": 0.2195,
                "fsim": 0.6103,
                "mse": 0.0192,
                "mae": 0.0989,
                "runtime_ms": 663.4272
            }
            if os.path.exists(baseline_json_path):
                try:
                    with open(baseline_json_path, 'r') as f:
                        b_data = json.load(f)
                        bm = b_data["metrics"]
                        baseline = {
                            "psnr": bm["psnr"]["mean"],
                            "ssim": bm["ssim"]["mean"],
                            "fsim": bm["fsim"]["mean"],
                            "mse": bm["mse"]["mean"],
                            "mae": bm["mae"]["mean"],
                            "runtime_ms": bm["runtime_ms"]["mean"]
                        }
                except Exception as e:
                    print(f"Error reading baseline report: {e}")
                    
            def format_impr(metric, b_val, f_val):
                diff = f_val - b_val
                if metric in ["psnr", "ssim", "fsim"]:
                    sign = "+" if diff >= 0 else ""
                    return f"{sign}{diff:.3f}" if metric != "psnr" else f"{sign}{diff:.2f}"
                else:
                    sign = "-" if diff < 0 else "+"
                    return f"{diff:.6f}" if metric == "mse" else (f"{diff:.4f}" if metric == "mae" else f"{diff:.1f}")
            
            md_content = f"""# Baseline vs. Fine-tuned (Practical-RIFE 4.26)

This report compares the performance of the baseline Practical-RIFE 4.26 model (pretrained on general datasets) against the fine-tuned model (trained on GOES-19 ABI Channel 13 satellite imagery).

| Metric | Baseline | Fine-tuned | Improvement |
| :--- | :--- | :--- | :--- |
| PSNR | {baseline['psnr']:.2f} | {self.final_metrics['psnr']:.2f} | {format_impr('psnr', baseline['psnr'], self.final_metrics['psnr'])} |
| SSIM | {baseline['ssim']:.3f} | {self.final_metrics['ssim']:.3f} | {format_impr('ssim', baseline['ssim'], self.final_metrics['ssim'])} |
| FSIM | {baseline['fsim']:.3f} | {self.final_metrics['fsim']:.3f} | {format_impr('fsim', baseline['fsim'], self.final_metrics['fsim'])} |
| MSE | {baseline['mse']:.6f} | {self.final_metrics['mse']:.6f} | {format_impr('mse', baseline['mse'], self.final_metrics['mse'])} |
| MAE | {baseline['mae']:.4f} | {self.final_metrics['mae']:.4f} | {format_impr('mae', baseline['mae'], self.final_metrics['mae'])} |
| Runtime | {baseline['runtime_ms']:.1f} ms | {self.final_metrics['runtime_ms']:.1f} ms | {format_impr('runtime_ms', baseline['runtime_ms'], self.final_metrics['runtime_ms'])} ms |
"""
            with open(os.path.join(self.output_dir, "baseline_vs_finetuned.md"), "w", encoding="utf-8") as f:
                f.write(md_content)
            root_md_path = os.path.abspath(os.path.join(current_dir, "..", "baseline_vs_finetuned.md"))
            with open(root_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
            print(f"Generated comparison report at {root_md_path}")
            
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _generate_readme(self):
        readme_content = f"""# Experiment {self.run_id} — Practical-RIFE 4.26 Fine-tuning

This folder contains all artifacts from **Experiment {self.run_id}**: fine-tuning Practical-RIFE 4.26 on GOES-19 satellite imagery.

## Contents
- `config.json`: The training configuration.
- `environment.json`: GPU, PyTorch, CUDA, and dataset environment metadata.
- `metrics.csv` & `metrics.json`: Epoch-by-epoch training and validation metrics.
- `tensorboard/`: TensorBoard logs.
- `best.pth` & `latest.pth`: Model checkpoints (best and latest).
- `sample_predictions/`: Ground truth, prediction, and difference visualizations for 10 random validation triplets.
- `training.log`: Standard output and error logs from the run.
- `baseline_vs_finetuned.md`: Performance comparison table.
- `training_report.md`: Detailed experiment summary report.

## Summary Results
- **Best Validation PSNR:** {self.best_psnr:.4f} dB
- **Best Validation SSIM:** {self.best_ssim:.4f}
"""
        with open(os.path.join(self.output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

    def _generate_training_report(self):
        print("Generating training report...")
        self._plot_curves()
        
        best_epoch = self.start_epoch
        best_psnr_val = 0.0
        for h in self.history:
            if h["psnr"] > best_psnr_val:
                best_psnr_val = h["psnr"]
                best_epoch = h["epoch"]
                
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        max_mem_str = "N/A"
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            max_mem_str = f"{max_mem_mb:.2f} MB"
            
        duration_str = str(datetime.timedelta(seconds=int(self.total_training_time)))
        
        comparison_md = ""
        comp_path = os.path.join(self.output_dir, "baseline_vs_finetuned.md")
        if os.path.exists(comp_path):
            with open(comp_path, 'r', encoding='utf-8') as f:
                comparison_md = f.read()
                if "# Baseline vs. Fine-tuned" in comparison_md:
                    comparison_md = comparison_md.split("\n\n", 1)[1]
                    
        cache_dir = self.config.get("dataset_path", "datasets/cache/goes19_cache")
        quarantine_dir = self.config.get("quarantine_dir", "datasets/quarantine/goes19_quarantine")
        valid_files = glob.glob(os.path.join(cache_dir, "*.nc"))
        num_valid = len(valid_files)
        num_rejected = len(glob.glob(os.path.join(quarantine_dir, "*.nc"))) if os.path.exists(quarantine_dir) else 0
        
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
        total_bytes = sum(os.path.getsize(f) for f in valid_files)
        dataset_size_gb = total_bytes / (1024 ** 3)
        
        vis_images_md = ""
        if self.val_vis_indices:
            vis_idx = sorted(list(self.val_vis_indices))[0]
            gt_img = f"sample_predictions/epoch_{best_epoch:03d}_triplet_{vis_idx:02d}_gt.png"
            pred_img = f"sample_predictions/epoch_{best_epoch:03d}_triplet_{vis_idx:02d}_pred.png"
            diff_img = f"sample_predictions/epoch_{best_epoch:03d}_triplet_{vis_idx:02d}_diff.png"
            if os.path.exists(os.path.join(self.output_dir, gt_img)):
                vis_images_md = f"""### Sample Prediction (Triplet #{vis_idx} - Best Epoch {best_epoch})

| Ground Truth | Prediction | Difference Heatmap |
|:---:|:---:|:---:|
| ![{gt_img}]({gt_img}) | ![{pred_img}]({pred_img}) | ![{diff_img}]({diff_img}) |
"""

        report_content = f"""# Experiment 001: Practical-RIFE 4.26 GOES-19 Fine-Tuning Report

## Executive Summary
This report summarizes the results of fine-tuning Practical-RIFE 4.26 on GOES-19 ABI Channel 13 satellite imagery for frame interpolation.

## Dataset Summary
- **Total Scenes:** {num_scenes}
- **Valid Scenes:** {num_valid}
- **Rejected/Quarantined:** {num_rejected}
- **Triplets Formed:** {num_triplets}
- **Channel:** {self.config.get("channel", 13)}
- **Sector:** {self.config.get("sector", "CONUS")}
- **Dataset Size:** {dataset_size_gb:.2f} GB

## Training Details
- **Training Duration:** {duration_str}
- **Best Epoch:** Epoch {best_epoch}
- **GPU Model:** {gpu_model}
- **Max GPU Memory Allocated:** {max_mem_str}

## Loss & Validation Curves

### Training & Validation Loss
![Loss Curve](loss_curve.png)

### Validation Metrics
![Validation Curves](validation_curves.png)

## Evaluation & Metrics Comparison
{comparison_md}

{vis_images_md}

## Next Recommendations
1. **Extend Epochs:** The validation metrics show steady improvement; training for additional epochs could yield higher PSNR/SSIM.
2. **Learning Rate Tuning:** Perform a grid search on learning rates (e.g. 5e-5, 2e-4) to optimize convergence.
3. **Loss Weight Optimization:** Tune the alpha loss weight (L1 vs SSIM loss weight) to emphasize structural alignment in cloud motions.
4. **Data Augmentation:** Apply random rotations and flips to prevent spatial overfitting and improve model generalization.
"""
        report_path = os.path.join(self.output_dir, "training_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"Training report successfully saved to {report_path}")

    def train(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}... Directory: {self.output_dir}")
        self.start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start_time = time.time()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                train_loss = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)
                
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['psnr'])
                    else:
                        self.scheduler.step()
                
                is_best = val_metrics["psnr"] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics["psnr"]
                    self.best_ssim = val_metrics["ssim"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                self._save_checkpoint(is_best, epoch)
                self._log_metrics(epoch, train_loss, val_metrics)
                
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch} Summary ({epoch_time:.1f}s): Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.patience_counter} epochs without improvement.")
                    break
                    
            self.total_training_time = time.time() - self.start_time
            self._save_environment_info(self.total_training_time)
            
            # Post-training Final Evaluation
            self._evaluate_best_checkpoint()
            
            # Generate Report
            self._generate_training_report()
            
            # Generate README
            self._generate_readme()
            
            print("Training complete. Updating experiment registry.")
            self._update_experiment_registry()
            self.writer.close()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted via Ctrl+C. Saving latest checkpoint gracefully...")
            epoch_val = epoch if 'epoch' in locals() else self.start_epoch
            self._save_checkpoint(is_best=False, epoch=epoch_val)
            print("Checkpoint saved. Exiting training.")
            sys.exit(0)
