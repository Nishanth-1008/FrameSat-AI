import os
import time
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from .datasets import SatelliteDataset
from .metrics import compute_psnr, compute_ssim, compute_mae, compute_mse, compute_fsim
from .visualizer import save_comparison_figure
from .preprocessing import get_preprocessor_from_config

class Evaluator:
    """
    Represents the evaluation executor.
    Sets up run directories, runs inference, tracks metrics and latency,
    applies preprocessing steps, and outputs results.
    """
    
    def __init__(self, model, dataset: SatelliteDataset, config: dict):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.experiment_name = config.get("experiment_name", "experiment")
        self.save_predictions = config.get("save_predictions", False)
        self.num_events = config.get("events", 5)
        
        # Setup preprocessor
        self.preprocessor = get_preprocessor_from_config(config)
        
        # Setup output folders
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            "evaluation", "experiments", f"run_{self.timestamp}_{self.experiment_name}"
        )
        
        self.visual_dir = os.path.join(self.run_dir, "visualizations")
        self.pred_dir = os.path.join(self.run_dir, "predictions")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)
        if self.save_predictions:
            os.makedirs(self.pred_dir, exist_ok=True)
            
        # Save run config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
    def run(self) -> dict:
        """
        Executes the evaluation run.
        """
        self.dataset.load()
        events_to_run = min(self.num_events, self.dataset.num_events)
        
        results = []
        log_lines = []
        
        def log_info(msg):
            print(msg)
            log_lines.append(f"[{datetime.now().isoformat()}] {msg}")
            
        log_info(f"Starting experiment: {self.experiment_name}")
        log_info(f"Modality: {self.dataset.modality.upper()} | Events: {events_to_run}")
        log_info(f"Preprocessor: {self.preprocessor.__class__.__name__}")
        
        for idx in range(events_to_run):
            log_info(f"\n--- Event {idx+1}/{events_to_run} ---")
            
            # 1. Fetch raw frames (t0, gt, t2)
            t0, gt, t2 = self.dataset.get_event_triplet(idx)
            
            # 2. Apply preprocessing (CLAHE, resize, etc.)
            t0_proc = self.preprocessor.process(t0)
            gt_proc = self.preprocessor.process(gt)
            t2_proc = self.preprocessor.process(t2)
            
            # 3. Run model inference with timing
            start_time = time.time()
            pred_proc = self.model.interpolate(t0_proc, t2_proc)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # 4. Compute metrics
            psnr = compute_psnr(gt_proc, pred_proc)
            ssim = compute_ssim(gt_proc, pred_proc)
            fsim = compute_fsim(gt_proc, pred_proc)
            mse = compute_mse(gt_proc, pred_proc)
            mae = compute_mae(gt_proc, pred_proc)
            
            log_info(f"  Metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, FSIM: {fsim:.4f}, MSE: {mse:.6f}, MAE: {mae:.4f}")
            log_info(f"  Latency - Time: {inference_time:.1f} ms")
            
            event_result = {
                "event_index": idx,
                "psnr": psnr,
                "ssim": ssim,
                "fsim": fsim,
                "mse": mse,
                "mae": mae,
                "inference_time_ms": inference_time
            }
            results.append(event_result)
            
            # 5. Save visualizations
            out_path = os.path.join(self.visual_dir, f"event_{idx}_vis.png")
            title = f"Event {idx} ({self.dataset.modality.upper()}) | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
            save_comparison_figure(t0_proc, gt_proc, pred_proc, out_path, title)
            
            # Save individual component images as requested
            plt.imsave(os.path.join(self.visual_dir, f"event_{idx}_gt.png"), gt_proc, cmap='gray')
            plt.imsave(os.path.join(self.visual_dir, f"event_{idx}_pred.png"), pred_proc, cmap='gray')
            diff = np.abs(gt_proc - pred_proc)
            plt.imsave(os.path.join(self.visual_dir, f"event_{idx}_diff.png"), diff, cmap='hot', vmin=0.0, vmax=0.2)
            
            # 6. Save raw predictions (numpy arrays) if configured
            if self.save_predictions:
                npz_path = os.path.join(self.pred_dir, f"event_{idx}_pred.npz")
                np.savez_compressed(npz_path, t0=t0_proc, gt=gt_proc, pred=pred_proc)
                
        # Aggregate stats
        avg_psnr = sum(r['psnr'] for r in results) / events_to_run
        avg_ssim = sum(r['ssim'] for r in results) / events_to_run
        avg_fsim = sum(r['fsim'] for r in results) / events_to_run
        avg_mse = sum(r['mse'] for r in results) / events_to_run
        avg_mae = sum(r['mae'] for r in results) / events_to_run
        avg_time = sum(r['inference_time_ms'] for r in results) / events_to_run
        
        summary = {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_fsim": avg_fsim,
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "avg_inference_time_ms": avg_time
        }
        
        report = {
            "timestamp": self.timestamp,
            "experiment_name": self.experiment_name,
            "modality": self.dataset.modality,
            "model_type": self.model.__class__.__name__,
            "num_events": events_to_run,
            "config": self.config,
            "summary": summary,
            "details": results
        }
        
        # Save metrics file
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(report, f, indent=4)
            
        # Write logs
        with open(os.path.join(self.run_dir, "run.log"), "w") as f:
            f.write("\n".join(log_lines))
            
        # Append to main CSV
        csv_path = os.path.join("evaluation", "experiments", "experiments_summary.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Experiment", "Model", "Modality", "Events", "Avg_PSNR", "Avg_SSIM", "Avg_FSIM", "Avg_MSE", "Avg_MAE", "Avg_Time_ms"])
            writer.writerow([
                self.timestamp, self.experiment_name, self.model.__class__.__name__, self.dataset.modality, 
                events_to_run, f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_fsim:.4f}", f"{avg_mse:.6f}", f"{avg_mae:.4f}", f"{avg_time:.1f}"
            ])
            
        log_info("\n================ RUN FINISHED ================")
        log_info(f"Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f} | Avg FSIM: {avg_fsim:.4f} | Avg MSE: {avg_mse:.6f} | Avg MAE: {avg_mae:.4f}")
        log_info(f"Saved run report to: {self.run_dir}")
        log_info("==============================================")
        
        return report
