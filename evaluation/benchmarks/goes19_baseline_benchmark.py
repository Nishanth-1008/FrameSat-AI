"""
Phase 4: Baseline Benchmark of pretrained Practical-RIFE 4.26 on GOES-19 triplets.

Computes:
- PSNR, SSIM, FSIM, MSE, MAE, and runtime per triplet.
- Saves predictions, ground truth, and difference heatmaps.
- Produces a comprehensive baseline report for future comparison.
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
from models.rife.interpolator import RIFEInterpolator
from evaluation.metrics.metrics import compute_psnr, compute_ssim, compute_fsim, compute_mse, compute_mae


def run_baseline_benchmark(cache_dir, report_dir, weights_path=None):
    """Main entry point for Phase 4."""
    print("\n" + "=" * 60)
    print("PHASE 4: Baseline Benchmark (Pretrained RIFE 4.26)")
    print("=" * 60)

    benchmark_dir = os.path.join(report_dir, "baseline_benchmark")
    visual_dir = os.path.join(benchmark_dir, "visualizations")
    os.makedirs(visual_dir, exist_ok=True)

    # Default weights path
    if weights_path is None:
        weights_path = os.path.join(current_dir, "weights", "rife_426", "train_log")

    # Load model
    print("Loading pretrained RIFE 4.26 model...")
    model = RIFEInterpolator()
    model.load_weights(weights_path)
    print("  Model loaded successfully.")

    # Load dataset (512x512 for tractable benchmark)
    start_date = datetime(2024, 10, 10, 21, 0, 0)
    end_date = start_date + timedelta(hours=3)

    dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        product='ABI-L1b-RadC',
        channel=13,
        split='train',
        split_ratio=1.0,
        train_resize=(512, 512)
    )

    total = len(dataset)
    print(f"Running benchmark on {total} triplets at 512x512...")

    results = []

    for idx in range(total):
        t0, t1, t2 = dataset[idx]

        # Convert to numpy (H, W) for the RIFE interpolator
        t0_np = t0.squeeze(0).numpy()
        t1_np = t1.squeeze(0).numpy()  # Ground truth
        t2_np = t2.squeeze(0).numpy()

        # Inference with timing
        start_time = time.time()
        pred_np = model.interpolate(t0_np, t2_np)
        runtime_ms = (time.time() - start_time) * 1000.0

        # Compute metrics
        psnr = compute_psnr(t1_np, pred_np)
        ssim = compute_ssim(t1_np, pred_np)
        fsim = compute_fsim(t1_np, pred_np)
        mse_val = compute_mse(t1_np, pred_np)
        mae_val = compute_mae(t1_np, pred_np)

        result = {
            "index": idx,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "fsim": round(fsim, 4),
            "mse": round(mse_val, 6),
            "mae": round(mae_val, 4),
            "runtime_ms": round(runtime_ms, 2)
        }
        results.append(result)

        # Save visuals
        plt.imsave(os.path.join(visual_dir, f"triplet_{idx:04d}_gt.png"), t1_np, cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(visual_dir, f"triplet_{idx:04d}_pred.png"), pred_np, cmap='gray', vmin=0, vmax=1)
        diff = np.abs(t1_np - pred_np)
        plt.imsave(os.path.join(visual_dir, f"triplet_{idx:04d}_diff.png"), diff, cmap='hot', vmin=0, vmax=0.2)

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{total} — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Runtime: {runtime_ms:.0f}ms")

    # Aggregate statistics
    psnr_vals = [r["psnr"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    fsim_vals = [r["fsim"] for r in results]
    mse_vals = [r["mse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    runtime_vals = [r["runtime_ms"] for r in results]

    def stat_summary(vals):
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "median": round(float(np.median(vals)), 4),
        }

    # Best / median / worst by PSNR
    sorted_results = sorted(results, key=lambda x: x["psnr"])
    worst = sorted_results[0]
    median_result = sorted_results[len(sorted_results) // 2]
    best = sorted_results[-1]

    summary = {
        "model": "Practical-RIFE 4.26 (pretrained, no fine-tuning)",
        "dataset": "GOES-19 ABI-L1b-RadC Channel 13",
        "resolution": "512x512",
        "num_triplets": total,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "psnr": stat_summary(psnr_vals),
            "ssim": stat_summary(ssim_vals),
            "fsim": stat_summary(fsim_vals),
            "mse": stat_summary(mse_vals),
            "mae": stat_summary(mae_vals),
            "runtime_ms": stat_summary(runtime_vals),
        },
        "case_studies": {
            "best": best,
            "median": median_result,
            "worst": worst,
        },
        "details": results,
    }

    # Save JSON
    json_path = os.path.join(benchmark_dir, "baseline_report.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Generate Markdown report
    md_path = os.path.join(benchmark_dir, "baseline_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# GOES-19 Baseline Benchmark Report\n\n")
        f.write(f"**Model:** Practical-RIFE 4.26 (pretrained, no fine-tuning)\n")
        f.write(f"**Dataset:** GOES-19 ABI-L1b-RadC Channel 13 (CONUS)\n")
        f.write(f"**Resolution:** 512×512 (resized from 1500×2500)\n")
        f.write(f"**Triplets evaluated:** {total}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Mean | Std | Min | Max | Median |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        for metric_name in ["psnr", "ssim", "fsim", "mse", "mae", "runtime_ms"]:
            s = summary["metrics"][metric_name]
            unit = " dB" if metric_name == "psnr" else (" ms" if metric_name == "runtime_ms" else "")
            f.write(f"| {metric_name.upper()} | {s['mean']}{unit} | {s['std']} | {s['min']} | {s['max']} | {s['median']} |\n")
        f.write("\n")

        f.write("## Case Studies\n\n")
        for case_name, case in [("Best", best), ("Median", median_result), ("Worst", worst)]:
            idx = case["index"]
            f.write(f"### {case_name} Case (Triplet #{idx})\n\n")
            f.write(f"- **PSNR:** {case['psnr']} dB\n")
            f.write(f"- **SSIM:** {case['ssim']}\n")
            f.write(f"- **FSIM:** {case['fsim']}\n")
            f.write(f"- **MSE:** {case['mse']}\n")
            f.write(f"- **MAE:** {case['mae']}\n")
            f.write(f"- **Runtime:** {case['runtime_ms']} ms\n\n")
            f.write(f"| Ground Truth | Prediction | Difference Heatmap |\n")
            f.write(f"|:---:|:---:|:---:|\n")
            f.write(f"| ![GT](visualizations/triplet_{idx:04d}_gt.png) | ![Pred](visualizations/triplet_{idx:04d}_pred.png) | ![Diff](visualizations/triplet_{idx:04d}_diff.png) |\n\n")

        f.write("## Purpose\n\n")
        f.write("This report establishes the **pre-fine-tuning baseline**. After training on GOES-19 data, ")
        f.write("the fine-tuned model's metrics should be compared against these values to quantify improvement.\n")

    print(f"\n  Baseline Report: {md_path}")
    print(f"  JSON Data: {json_path}")
    print(f"  Visualizations: {visual_dir}")

    # Print summary
    ms = summary["metrics"]
    print(f"\n  === BASELINE RESULTS ===")
    print(f"  PSNR:  {ms['psnr']['mean']:.2f} ± {ms['psnr']['std']:.2f} dB")
    print(f"  SSIM:  {ms['ssim']['mean']:.4f} ± {ms['ssim']['std']:.4f}")
    print(f"  FSIM:  {ms['fsim']['mean']:.4f} ± {ms['fsim']['std']:.4f}")
    print(f"  MSE:   {ms['mse']['mean']:.6f}")
    print(f"  MAE:   {ms['mae']['mean']:.4f}")
    print(f"  Runtime: {ms['runtime_ms']['mean']:.0f} ms/triplet")

    return summary


if __name__ == "__main__":
    cache_dir = os.path.join(os.path.dirname(current_dir), "datasets", "goes19_test_cache")
    report_dir = os.path.join(current_dir, "reports")
    run_baseline_benchmark(cache_dir, report_dir)
