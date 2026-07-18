"""
Phase 2: Dataset Integrity Check for GOES-19 triplets.

Validates:
- All triplets are strictly chronological (t0 < t1 < t2).
- No missing timestamps.
- No duplicate scenes.
- Time interval distribution analysis.
- Accepted vs. rejected triplet reporting.
"""
import os
import sys
import json
import glob
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))


def extract_timestamps(nc_files):
    """Extract timestamps from all NetCDF files."""
    file_timestamps = []
    errors = []

    for f in nc_files:
        try:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                t_str = ds.attrs.get('time_coverage_start', '')
                if t_str:
                    dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                    file_timestamps.append((f, dt))
                else:
                    errors.append({"file": os.path.basename(f), "error": "Missing time_coverage_start attribute"})
        except Exception as e:
            errors.append({"file": os.path.basename(f), "error": f"Failed to open: {str(e)}"})

    # Sort by timestamp
    file_timestamps.sort(key=lambda x: x[1])
    return file_timestamps, errors


def check_duplicates(file_timestamps):
    """Detect duplicate timestamps."""
    ts_counter = Counter(dt.isoformat() for _, dt in file_timestamps)
    duplicates = {ts: count for ts, count in ts_counter.items() if count > 1}
    return duplicates


def compute_intervals(file_timestamps):
    """Compute time intervals between consecutive scenes in minutes."""
    intervals = []
    for i in range(1, len(file_timestamps)):
        delta = (file_timestamps[i][1] - file_timestamps[i - 1][1]).total_seconds() / 60.0
        intervals.append(delta)
    return intervals


def validate_triplets(file_timestamps, max_gap_minutes=15.0):
    """
    Generate triplets and classify as accepted/rejected.
    A triplet is rejected if:
    - Any timestamp gap exceeds max_gap_minutes.
    - Timestamps are not strictly chronological.
    """
    accepted = []
    rejected = []

    for i in range(len(file_timestamps) - 2):
        f0, t0 = file_timestamps[i]
        f1, t1 = file_timestamps[i + 1]
        f2, t2 = file_timestamps[i + 2]

        triplet = {
            "index": i,
            "files": [os.path.basename(f0), os.path.basename(f1), os.path.basename(f2)],
            "timestamps": [t0.isoformat(), t1.isoformat(), t2.isoformat()],
        }

        reasons = []

        # Chronological check
        if not (t0 < t1 < t2):
            reasons.append("Non-chronological timestamps")

        # Gap check
        gap1 = (t1 - t0).total_seconds() / 60.0
        gap2 = (t2 - t1).total_seconds() / 60.0
        triplet["gap1_min"] = round(gap1, 2)
        triplet["gap2_min"] = round(gap2, 2)

        if gap1 > max_gap_minutes:
            reasons.append(f"Gap t0→t1 = {gap1:.1f}m exceeds {max_gap_minutes}m")
        if gap2 > max_gap_minutes:
            reasons.append(f"Gap t1→t2 = {gap2:.1f}m exceeds {max_gap_minutes}m")

        if reasons:
            triplet["rejection_reasons"] = reasons
            rejected.append(triplet)
        else:
            accepted.append(triplet)

    return accepted, rejected


def generate_histogram(intervals, report_dir):
    """Generate and save the time interval histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(intervals, bins=50, color='#4A90D9', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Time Interval (minutes)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('GOES-19 Scene Time Interval Distribution', fontsize=14)
    ax.axvline(x=5.0, color='red', linestyle='--', label='Expected 5-min cadence')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(report_dir, "interval_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(results, report_dir):
    """Generate Markdown and JSON reports."""
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, "integrity_report.json")
    # Convert for JSON serialization
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    md_path = os.path.join(report_dir, "integrity_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# GOES-19 Dataset Integrity Report\n\n")

        # Summary
        f.write("## 1. Scene Summary\n\n")
        f.write(f"- **Total NetCDF files scanned:** {results['total_files']}\n")
        f.write(f"- **Files with valid timestamps:** {results['valid_timestamps']}\n")
        f.write(f"- **Files with errors:** {results['file_errors_count']}\n\n")

        # Duplicates
        f.write("## 2. Duplicate Timestamp Detection\n\n")
        dupes = results.get("duplicates", {})
        if not dupes:
            f.write("> [!TIP]\n> No duplicate timestamps detected.\n\n")
        else:
            f.write("> [!WARNING]\n> Duplicate timestamps found!\n\n")
            for ts, count in dupes.items():
                f.write(f"- `{ts}` appears **{count}** times\n")
            f.write("\n")

        # Triplet validation
        f.write("## 3. Triplet Validation\n\n")
        f.write(f"- **Total possible triplets:** {results['total_triplets']}\n")
        f.write(f"- **Accepted triplets:** {results['accepted_count']} ✅\n")
        f.write(f"- **Rejected triplets:** {results['rejected_count']} ❌\n")
        f.write(f"- **Acceptance rate:** {results['acceptance_rate_pct']:.1f}%\n\n")

        if results['rejected_count'] > 0:
            f.write("### Rejected Triplets\n\n")
            f.write("| Index | Gap1 (min) | Gap2 (min) | Reason |\n")
            f.write("|:---|:---|:---|:---|\n")
            for r in results.get("rejected_triplets", [])[:20]:  # Show up to 20
                reasons = "; ".join(r.get("rejection_reasons", []))
                f.write(f"| {r['index']} | {r['gap1_min']} | {r['gap2_min']} | {reasons} |\n")
            f.write("\n")

        # Interval statistics
        f.write("## 4. Time Interval Distribution\n\n")
        istats = results.get("interval_stats", {})
        if istats:
            f.write(f"- **Mean interval:** {istats['mean_min']:.2f} minutes\n")
            f.write(f"- **Std deviation:** {istats['std_min']:.2f} minutes\n")
            f.write(f"- **Min interval:** {istats['min_min']:.2f} minutes\n")
            f.write(f"- **Max interval:** {istats['max_min']:.2f} minutes\n")
            f.write(f"- **Median interval:** {istats['median_min']:.2f} minutes\n\n")
        f.write("![Interval Histogram](interval_histogram.png)\n\n")

        # File errors
        if results['file_errors_count'] > 0:
            f.write("## 5. File Errors\n\n")
            for err in results.get("file_errors", []):
                f.write(f"- `{err['file']}`: {err['error']}\n")

    return md_path, json_path


def run_integrity_check(cache_dir, report_dir, max_gap_minutes=15.0):
    """Main entry point for Phase 2."""
    print("\n" + "=" * 60)
    print("PHASE 2: Dataset Integrity Check")
    print("=" * 60)

    nc_files = sorted(glob.glob(os.path.join(cache_dir, "*.nc")))
    if not nc_files:
        print(f"ERROR: No NetCDF files found in {cache_dir}")
        return None

    print(f"Scanning {len(nc_files)} files...")

    file_timestamps, file_errors = extract_timestamps(nc_files)
    print(f"  Valid timestamps extracted: {len(file_timestamps)}")
    print(f"  File errors: {len(file_errors)}")

    duplicates = check_duplicates(file_timestamps)
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate timestamps found")
    else:
        print("  No duplicate timestamps")

    intervals = compute_intervals(file_timestamps)
    accepted, rejected = validate_triplets(file_timestamps, max_gap_minutes)
    total_triplets = len(accepted) + len(rejected)
    acceptance_rate = (len(accepted) / total_triplets * 100) if total_triplets > 0 else 0

    print(f"  Triplets: {len(accepted)} accepted / {len(rejected)} rejected ({acceptance_rate:.1f}% acceptance rate)")

    interval_stats = {}
    if intervals:
        interval_stats = {
            "mean_min": round(float(np.mean(intervals)), 2),
            "std_min": round(float(np.std(intervals)), 2),
            "min_min": round(float(np.min(intervals)), 2),
            "max_min": round(float(np.max(intervals)), 2),
            "median_min": round(float(np.median(intervals)), 2),
        }

    results = {
        "total_files": len(nc_files),
        "valid_timestamps": len(file_timestamps),
        "file_errors_count": len(file_errors),
        "file_errors": file_errors,
        "duplicates": duplicates,
        "total_triplets": total_triplets,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate_pct": round(acceptance_rate, 2),
        "accepted_triplets": accepted,
        "rejected_triplets": rejected,
        "interval_stats": interval_stats,
    }

    os.makedirs(report_dir, exist_ok=True)
    hist_path = generate_histogram(intervals, report_dir)
    print(f"  Histogram: {hist_path}")

    md_path, json_path = generate_report(results, report_dir)
    print(f"  Report: {md_path}")

    return results


if __name__ == "__main__":
    cache_dir = os.path.join(os.path.dirname(current_dir), "datasets", "goes19_test_cache")
    report_dir = os.path.join(current_dir, "reports")
    run_integrity_check(cache_dir, report_dir)
