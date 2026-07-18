import os
import sys
import glob
import json
from datetime import datetime

# Add root directory to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from evaluation.datasets import SatelliteDataset
from models.rife.interpolator import RIFEInterpolator
from evaluation.evaluator import Evaluator

def run_config(config_path: str) -> dict:
    print(f"\n====================================================")
    print(f"Batch Runner: Loading config {os.path.basename(config_path)}...")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model_type = config.get("model", "rife").lower()
    modality = config.get("modality", "vis").lower()
    
    # Initialize model
    if model_type == "rife":
        model = RIFEInterpolator()
        w_path = config.get("weights")
        if w_path:
            if not os.path.isabs(w_path):
                weights_path = os.path.abspath(os.path.join(current_dir, "..", w_path))
            else:
                weights_path = w_path
        else:
            weights_path = os.path.join(current_dir, "models", "rife_src", "train_log")
        model.load_weights(weights_path)
    else:
        raise ValueError(f"Model {model_type} is not supported.")
        
    dataset = SatelliteDataset(
        modality=modality,
        download_dir=os.path.join(current_dir, "datasets")
    )
    
    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        config=config
    )
    
    try:
        report = evaluator.run()
        # Save run path info in report for link generation
        report["run_dir"] = evaluator.run_dir
        return report
    finally:
        dataset.close()

def generate_comparison_report(reports: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(current_dir, "experiments", "comparison_report.md")
    
    # Sort models by PSNR, SSIM, FSIM, MSE, MAE, and Time to find rankings
    reports_sorted_psnr = sorted(reports, key=lambda x: x["summary"]["avg_psnr"], reverse=True)
    reports_sorted_ssim = sorted(reports, key=lambda x: x["summary"]["avg_ssim"], reverse=True)
    reports_sorted_fsim = sorted(reports, key=lambda x: x["summary"]["avg_fsim"], reverse=True)
    reports_sorted_mse = sorted(reports, key=lambda x: x["summary"]["avg_mse"])
    reports_sorted_mae = sorted(reports, key=lambda x: x["summary"]["avg_mae"])
    reports_sorted_time = sorted(reports, key=lambda x: x["summary"]["avg_inference_time_ms"])

    lines = [
        "# Practical-RIFE Checkpoint Comparison Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report evaluates and ranks the performance metrics across all configurations in this batch run.",
        "",
        "## Quantitative Performance Summary",
        "",
        "| Experiment Name | Model Type | modality | Avg PSNR (dB) | Avg SSIM | Avg FSIM | Avg MSE | Avg MAE | Avg Time (ms) | Run Directory |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in reports:
        summary = r["summary"]
        run_dir_rel = os.path.relpath(r["run_dir"], current_dir).replace("\\", "/")
        lines.append(
            f"| {r['experiment_name']} | {r['model_type']} | {r['modality'].upper()} | "
            f"{summary['avg_psnr']:.4f} | {summary['avg_ssim']:.4f} | {summary['avg_fsim']:.4f} | "
            f"{summary['avg_mse']:.6f} | {summary['avg_mae']:.4f} | {summary['avg_inference_time_ms']:.1f} | "
            f"[Link](./{run_dir_rel}) |"
        )
        
    lines.append("")
    lines.append("## Checkpoint Rankings")
    lines.append("")
    
    lines.append("### 1. Ranking by Structural Quality (PSNR)")
    for i, r in enumerate(reports_sorted_psnr):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_psnr']:.4f} dB")
    lines.append("")
    
    lines.append("### 2. Ranking by Structural Similarity (SSIM)")
    for i, r in enumerate(reports_sorted_ssim):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_ssim']:.4f}")
    lines.append("")
    
    lines.append("### 3. Ranking by Feature Similarity (FSIM)")
    for i, r in enumerate(reports_sorted_fsim):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_fsim']:.4f}")
    lines.append("")
    
    lines.append("### 4. Ranking by Mean Squared Error (MSE - lower is better)")
    for i, r in enumerate(reports_sorted_mse):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_mse']:.6f}")
    lines.append("")

    lines.append("### 5. Ranking by Mean Absolute Error (MAE - lower is better)")
    for i, r in enumerate(reports_sorted_mae):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_mae']:.4f}")
    lines.append("")

    lines.append("### 6. Ranking by Inference Latency (Runtime - lower is better)")
    for i, r in enumerate(reports_sorted_time):
        lines.append(f"{i+1}. **{r['experiment_name']}**: {r['summary']['avg_inference_time_ms']:.1f} ms")
    lines.append("")
    
    lines.append("## Insights and Observations")
    lines.append("- Refer to individual run subdirectories linked above to inspect Ground Truth (`event_X_gt.png`), Prediction (`event_X_pred.png`), and Difference Heatmap (`event_X_diff.png`) images.")
    lines.append("- Visual quality can be cross-referenced with the absolute error heatmap, where warmer colors (white/yellow) indicate higher interpolation errors.")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"\nComparative report successfully saved to: {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch execute satellite evaluations.")
    parser.add_argument("--configs-dir", type=str, default="evaluation/configs", help="Directory containing config JSONs")
    args = parser.parse_args()
    
    config_pattern = os.path.join(args.configs_dir, "*.json")
    config_files = glob.glob(config_pattern)
    
    if not config_files:
        print(f"No JSON configuration files found in {args.configs_dir}")
        return
        
    print(f"Found {len(config_files)} configurations to run.")
    
    reports = []
    for f in config_files:
        try:
            report = run_config(f)
            reports.append(report)
        except Exception as e:
            print(f"Error running config {f}: {e}")
            
    if reports:
        generate_comparison_report(reports)

if __name__ == "__main__":
    main()
