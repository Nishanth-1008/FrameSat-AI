"""
GOES-19 Pre-Training Verification Suite — Master Runner

Orchestrates all 4 phases sequentially:
  Phase 1: Scientific Verification
  Phase 2: Dataset Integrity
  Phase 3: Visual QA
  Phase 4: Baseline Benchmark
"""
import os
import sys
import time

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from evaluation.goes19_scientific_verification import run_scientific_verification
from evaluation.goes19_integrity_check import run_integrity_check
from evaluation.goes19_visual_qa import run_visual_qa
from evaluation.goes19_baseline_benchmark import run_baseline_benchmark


def main():
    print("=" * 70)
    print("  GOES-19 PRE-TRAINING VERIFICATION SUITE")
    print("=" * 70)

    cache_dir = os.path.join(project_root, "datasets", "goes19_test_cache")
    report_dir = os.path.join(project_root, "evaluation", "reports")
    os.makedirs(report_dir, exist_ok=True)

    overall_start = time.time()
    phase_results = {}

    # ─── Phase 1: Scientific Verification ──────────────────────────────
    t0 = time.time()
    sci_result = run_scientific_verification(cache_dir, report_dir)
    phase_results["phase1_scientific"] = {
        "passed": sci_result.get("overall_pass", False) if sci_result else False,
        "runtime_s": round(time.time() - t0, 2)
    }

    # ─── Phase 2: Dataset Integrity ────────────────────────────────────
    t0 = time.time()
    integrity_result = run_integrity_check(cache_dir, report_dir)
    phase_results["phase2_integrity"] = {
        "accepted": integrity_result.get("accepted_count", 0) if integrity_result else 0,
        "rejected": integrity_result.get("rejected_count", 0) if integrity_result else 0,
        "runtime_s": round(time.time() - t0, 2)
    }

    # ─── Phase 3: Visual QA ────────────────────────────────────────────
    t0 = time.time()
    visual_result = run_visual_qa(cache_dir, report_dir)
    phase_results["phase3_visual_qa"] = {
        "samples_generated": visual_result.get("sample_count", 0) if visual_result else 0,
        "gallery_path": visual_result.get("gallery_path", "") if visual_result else "",
        "runtime_s": round(time.time() - t0, 2)
    }

    # ─── Phase 4: Baseline Benchmark ───────────────────────────────────
    t0 = time.time()
    benchmark_result = run_baseline_benchmark(cache_dir, report_dir)
    if benchmark_result:
        ms = benchmark_result["metrics"]
        phase_results["phase4_baseline"] = {
            "psnr_mean": ms["psnr"]["mean"],
            "ssim_mean": ms["ssim"]["mean"],
            "fsim_mean": ms["fsim"]["mean"],
            "runtime_s": round(time.time() - t0, 2)
        }
    else:
        phase_results["phase4_baseline"] = {"error": "Benchmark failed", "runtime_s": round(time.time() - t0, 2)}

    # ─── Final Summary ─────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("  VERIFICATION SUITE COMPLETE")
    print("=" * 70)
    print(f"  Total Runtime: {total_time:.1f}s\n")

    p1 = phase_results["phase1_scientific"]
    print(f"  Phase 1 (Scientific):  {'PASSED' if p1['passed'] else 'FAILED'}  ({p1['runtime_s']}s)")

    p2 = phase_results["phase2_integrity"]
    print(f"  Phase 2 (Integrity):   {p2['accepted']} accepted / {p2['rejected']} rejected  ({p2['runtime_s']}s)")

    p3 = phase_results["phase3_visual_qa"]
    print(f"  Phase 3 (Visual QA):   {p3['samples_generated']} samples generated  ({p3['runtime_s']}s)")

    p4 = phase_results["phase4_baseline"]
    if "psnr_mean" in p4:
        print(f"  Phase 4 (Baseline):    PSNR={p4['psnr_mean']:.2f} dB | SSIM={p4['ssim_mean']:.4f} | FSIM={p4['fsim_mean']:.4f}  ({p4['runtime_s']}s)")
    else:
        print(f"  Phase 4 (Baseline):    {p4.get('error', 'Unknown error')}  ({p4['runtime_s']}s)")

    print(f"\n  Reports directory: {report_dir}")
    print(f"    |-- scientific_verification.md")
    print(f"    |-- integrity_report.md")
    print(f"    |-- visual_qa/visual_qa_gallery.html")
    print(f"    +-- baseline_benchmark/baseline_report.md")


if __name__ == "__main__":
    main()
