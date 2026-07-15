import os
import time
import json
import csv
from datetime import datetime
from .datasets import SEVIRDataset
from .metrics import compute_psnr, compute_ssim, compute_mae
from .visualizer import save_comparison_figure

class Evaluator:
    """
    Orchestrates the evaluation pipeline for a given model and dataset.
    Generates summary reports and visual outputs.
    """
    
    def __init__(self, model, dataset: SEVIRDataset, output_dir: str = "outputs", experiment_dir: str = "experiments"):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.experiment_dir = experiment_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def run(self, num_events: int = 5, save_visuals: bool = True):
        """
        Run the evaluation over the specified number of events.
        """
        self.dataset.load()
        num_events = min(num_events, self.dataset.num_events)
        
        results = []
        
        for idx in range(num_events):
            print(f"\nEvaluating Event {idx+1}/{num_events}...")
            
            # Fetch triplet (using default frames 23, 24, 25 for testing dynamics)
            t0, gt, t2 = self.dataset.get_event_triplet(idx)
            
            # Run inference and time it
            start_time = time.time()
            pred = self.model.interpolate(t0, t2)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Compute metrics
            psnr = compute_psnr(gt, pred)
            ssim = compute_ssim(gt, pred)
            mae = compute_mae(gt, pred)
            
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MAE: {mae:.4f}, Time: {inference_time:.1f} ms")
            
            event_result = {
                "event_index": idx,
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
                "inference_time_ms": inference_time
            }
            results.append(event_result)
            
            # Save visual
            if save_visuals:
                out_path = os.path.join(self.output_dir, f"{self.dataset.modality}_event_{idx}.png")
                title = f"Event {idx} ({self.dataset.modality.upper()}) | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
                save_comparison_figure(t0, gt, pred, out_path, title)
                
        # Aggregate and save report
        self._generate_report(results, num_events)
        
    def _generate_report(self, results: list, num_events: int):
        avg_psnr = sum(r['psnr'] for r in results) / num_events
        avg_ssim = sum(r['ssim'] for r in results) / num_events
        avg_mae = sum(r['mae'] for r in results) / num_events
        avg_time = sum(r['inference_time_ms'] for r in results) / num_events
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "modality": self.dataset.modality,
            "model_type": self.model.__class__.__name__,
            "num_events": num_events,
            "summary": {
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "avg_mae": avg_mae,
                "avg_inference_time_ms": avg_time
            },
            "details": results
        }
        
        # Save JSON
        json_path = os.path.join(self.experiment_dir, f"report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        # Append to master CSV
        csv_path = os.path.join(self.experiment_dir, "experiments_summary.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Modality", "Events", "Avg_PSNR", "Avg_SSIM", "Avg_MAE", "Avg_Time_ms"])
            writer.writerow([
                timestamp, self.model.__class__.__name__, self.dataset.modality, 
                num_events, f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_mae:.4f}", f"{avg_time:.1f}"
            ])
            
        print("\n================ EVALUATION SUMMARY ================")
        print(f"Modality: {self.dataset.modality.upper()} | Model: {self.model.__class__.__name__}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average MAE:  {avg_mae:.4f}")
        print(f"Average Time: {avg_time:.1f} ms/frame")
        print(f"Report saved to: {json_path}")
        print("====================================================")
