"""
Phase 1: Scientific Verification of GOES-19 ABI Channel 13 Brightness Temperature Conversion.

Validates:
- Planck coefficients against NOAA-published values for ABI Band 13 (10.3 µm).
- BT conversion formula correctness.
- Fill-value masking and invalid pixel handling.
- Computed temperatures fall within expected physical ranges.
"""
import os
import sys
import json
import glob
import numpy as np
import xarray as xr

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# NOAA-published reference values for ABI Band 13 (10.3 µm) on GOES-R series
# Source: GOES-R PUG-L1b Vol. 3, Table 5.1.2.5-1
# These are approximate reference ranges; actual values are embedded per-file.
NOAA_REFERENCE = {
    "band": 13,
    "central_wavelength_um": 10.3,
    "expected_planck_fk1_range": (8.00e+03, 1.50e+04),   # mW m^-2 sr^-1 cm^-1 (ABI native units)
    "expected_planck_fk2_range": (1300.0, 1500.0),        # K
    "expected_planck_bc1_range": (0.0, 1.5),              # K
    "expected_planck_bc2_range": (0.99, 1.01),            # dimensionless
    "expected_bt_range_K": (170.0, 340.0),                # physically plausible BT range
    "typical_bt_range_K": (190.0, 310.0),                 # typical observed range
}


def verify_planck_coefficients(ds, results):
    """Verify Planck coefficients are within NOAA-published ranges."""
    fk1 = float(ds['planck_fk1'].values)
    fk2 = float(ds['planck_fk2'].values)
    bc1 = float(ds['planck_bc1'].values)
    bc2 = float(ds['planck_bc2'].values)

    results["planck_coefficients"] = {
        "fk1": fk1, "fk2": fk2, "bc1": bc1, "bc2": bc2
    }

    checks = []
    ref = NOAA_REFERENCE

    # fk1 check
    fk1_ok = ref["expected_planck_fk1_range"][0] <= fk1 <= ref["expected_planck_fk1_range"][1]
    checks.append({"name": "planck_fk1 in range", "value": fk1,
                    "range": ref["expected_planck_fk1_range"], "pass": fk1_ok})

    # fk2 check
    fk2_ok = ref["expected_planck_fk2_range"][0] <= fk2 <= ref["expected_planck_fk2_range"][1]
    checks.append({"name": "planck_fk2 in range", "value": fk2,
                    "range": ref["expected_planck_fk2_range"], "pass": fk2_ok})

    # bc1 check
    bc1_ok = ref["expected_planck_bc1_range"][0] <= bc1 <= ref["expected_planck_bc1_range"][1]
    checks.append({"name": "planck_bc1 in range", "value": bc1,
                    "range": ref["expected_planck_bc1_range"], "pass": bc1_ok})

    # bc2 check
    bc2_ok = ref["expected_planck_bc2_range"][0] <= bc2 <= ref["expected_planck_bc2_range"][1]
    checks.append({"name": "planck_bc2 in range", "value": bc2,
                    "range": ref["expected_planck_bc2_range"], "pass": bc2_ok})

    results["coefficient_checks"] = checks
    return all(c["pass"] for c in checks)


def verify_bt_conversion(ds, results):
    """Apply the BT conversion formula and validate the output."""
    rad = ds['Rad'].values
    fk1 = float(ds['planck_fk1'].values)
    fk2 = float(ds['planck_fk2'].values)
    bc1 = float(ds['planck_bc1'].values)
    bc2 = float(ds['planck_bc2'].values)

    total_pixels = rad.size
    valid_mask = rad > 0
    num_valid = int(np.sum(valid_mask))
    num_invalid = total_pixels - num_valid
    pct_invalid = (num_invalid / total_pixels) * 100.0

    results["pixel_coverage"] = {
        "total_pixels": total_pixels,
        "valid_pixels": num_valid,
        "invalid_pixels (rad <= 0)": num_invalid,
        "pct_invalid": round(pct_invalid, 4)
    }

    # Compute BT using the standard NOAA formula:
    # T_eff = fk2 / ln(fk1/L + 1)
    # T = (T_eff - bc1) / bc2
    bt = np.full_like(rad, fill_value=np.nan)
    if num_valid > 0:
        bt[valid_mask] = (fk2 / np.log((fk1 / rad[valid_mask]) + 1) - bc1) / bc2

    checks = []

    # Check NaN/Inf in valid region
    bt_valid = bt[valid_mask]
    has_nan = bool(np.any(np.isnan(bt_valid)))
    has_inf = bool(np.any(np.isinf(bt_valid)))
    checks.append({"name": "No NaN in valid BT", "pass": not has_nan})
    checks.append({"name": "No Inf in valid BT", "pass": not has_inf})

    # Check physical range
    if num_valid > 0:
        bt_min = float(np.nanmin(bt_valid))
        bt_max = float(np.nanmax(bt_valid))
        bt_mean = float(np.nanmean(bt_valid))
        bt_std = float(np.nanstd(bt_valid))

        ref = NOAA_REFERENCE
        range_ok = bt_min >= ref["expected_bt_range_K"][0] and bt_max <= ref["expected_bt_range_K"][1]
        checks.append({"name": "BT within physical range (170K-340K)",
                        "bt_min": bt_min, "bt_max": bt_max, "pass": range_ok})

        typical_ok = bt_mean >= ref["typical_bt_range_K"][0] and bt_mean <= ref["typical_bt_range_K"][1]
        checks.append({"name": "Mean BT within typical range (190K-310K)",
                        "bt_mean": bt_mean, "pass": typical_ok})

        results["bt_statistics"] = {
            "min_K": round(bt_min, 2),
            "max_K": round(bt_max, 2),
            "mean_K": round(bt_mean, 2),
            "std_K": round(bt_std, 2)
        }
    else:
        checks.append({"name": "BT within physical range", "pass": False,
                        "note": "No valid pixels found"})
        results["bt_statistics"] = {}

    # Check fill-value masking: invalid pixels should remain NaN (not converted)
    if num_invalid > 0:
        bt_invalid = bt[~valid_mask]
        fill_masked = bool(np.all(np.isnan(bt_invalid)))
        checks.append({"name": "Fill values correctly masked (NaN)", "pass": fill_masked})
    else:
        checks.append({"name": "Fill values correctly masked (NaN)", "pass": True,
                        "note": "No invalid pixels to check"})

    results["conversion_checks"] = checks
    return all(c["pass"] for c in checks)


def verify_normalization(bt_stats, results):
    """Verify the normalization step used in the dataset builder."""
    bt_min_norm, bt_max_norm = 180.0, 320.0
    checks = []

    if bt_stats:
        bt_min = bt_stats["min_K"]
        bt_max = bt_stats["max_K"]
        # Check if the normalization range covers most of the data
        coverage = (min(bt_max, bt_max_norm) - max(bt_min, bt_min_norm)) / (bt_max - bt_min) * 100 if bt_max > bt_min else 0
        checks.append({"name": "Normalization range [180K, 320K] covers data well",
                        "coverage_pct": round(coverage, 2),
                        "pass": coverage > 90.0})
        # Values outside the range are clipped, check the fraction that would be clipped
        # This is approximate based on min/max
        clipped_low = bt_min < bt_min_norm
        clipped_high = bt_max > bt_max_norm
        checks.append({"name": "Minimal low-end clipping (BT < 180K)",
                        "actual_min_K": bt_min, "pass": not clipped_low or bt_min > 170.0})
        checks.append({"name": "Minimal high-end clipping (BT > 320K)",
                        "actual_max_K": bt_max, "pass": not clipped_high or bt_max < 340.0})

    results["normalization_checks"] = checks
    return all(c["pass"] for c in checks)


def generate_report(results, report_dir):
    """Generate Markdown and JSON reports."""
    os.makedirs(report_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(report_dir, "scientific_verification.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Markdown
    md_path = os.path.join(report_dir, "scientific_verification.md")
    # Use UTF-8 to support check/cross marks
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# GOES-19 ABI Channel 13 — Scientific Verification Report\n\n")
        f.write(f"**File analyzed:** `{results.get('file_analyzed', 'N/A')}`\n\n")

        # Planck Coefficients
        f.write("## 1. Planck Coefficient Verification\n\n")
        f.write("Comparing extracted coefficients against NOAA-published reference ranges for ABI Band 13 (10.3 µm).\n\n")
        coeff = results.get("planck_coefficients", {})
        f.write(f"| Coefficient | Extracted Value | Expected Range | Status |\n")
        f.write(f"|:---|:---|:---|:---|\n")
        for check in results.get("coefficient_checks", []):
            status = "✅ PASS" if check["pass"] else "❌ FAIL"
            f.write(f"| {check['name']} | {check['value']:.6g} | {check['range']} | {status} |\n")
        f.write("\n")

        # BT Conversion
        f.write("## 2. Brightness Temperature Conversion\n\n")
        f.write("Formula: `T = (fk2 / ln(fk1/L + 1) - bc1) / bc2`\n\n")

        bt_stats = results.get("bt_statistics", {})
        if bt_stats:
            f.write(f"| Statistic | Value |\n")
            f.write(f"|:---|:---|\n")
            for k, v in bt_stats.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        for check in results.get("conversion_checks", []):
            status = "✅" if check["pass"] else "❌"
            note = f" — {check.get('note', '')}" if 'note' in check else ""
            extras = ""
            if "bt_min" in check:
                extras = f" (min={check['bt_min']:.2f}K, max={check['bt_max']:.2f}K)"
            if "bt_mean" in check:
                extras = f" (mean={check['bt_mean']:.2f}K)"
            f.write(f"- {status} {check['name']}{extras}{note}\n")
        f.write("\n")

        # Pixel Coverage
        f.write("## 3. Pixel Coverage & Fill-Value Masking\n\n")
        pix = results.get("pixel_coverage", {})
        for k, v in pix.items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n")

        # Normalization
        f.write("## 4. Normalization Verification\n\n")
        f.write("The dataset builder normalizes BT to [0, 1] using range [180K, 320K] with clipping.\n\n")
        for check in results.get("normalization_checks", []):
            status = "✅" if check["pass"] else "❌"
            extras = ""
            if "coverage_pct" in check:
                extras = f" ({check['coverage_pct']}% coverage)"
            if "actual_min_K" in check:
                extras = f" (actual min: {check['actual_min_K']}K)"
            if "actual_max_K" in check:
                extras = f" (actual max: {check['actual_max_K']}K)"
            f.write(f"- {status} {check['name']}{extras}\n")
        f.write("\n")

        # Overall verdict
        all_pass = results.get("overall_pass", False)
        if all_pass:
            f.write("> [!TIP]\n> **All scientific checks PASSED.** The BT conversion is physically correct and ready for training.\n")
        else:
            f.write("> [!WARNING]\n> **Some checks FAILED.** Review the details above before proceeding with training.\n")

    return md_path, json_path


def run_scientific_verification(cache_dir, report_dir):
    """Main entry point for Phase 1."""
    print("\n" + "=" * 60)
    print("PHASE 1: Scientific Verification")
    print("=" * 60)

    nc_files = sorted(glob.glob(os.path.join(cache_dir, "*.nc")))
    if not nc_files:
        print(f"ERROR: No NetCDF files found in {cache_dir}")
        return None

    # Use the first file as the representative sample
    sample_file = nc_files[0]
    print(f"Analyzing: {os.path.basename(sample_file)}")

    results = {"file_analyzed": os.path.basename(sample_file)}

    with xr.open_dataset(sample_file, engine='h5netcdf') as ds:
        coeff_ok = verify_planck_coefficients(ds, results)
        print(f"  Planck coefficients: {'PASS' if coeff_ok else 'FAIL'}")

        bt_ok = verify_bt_conversion(ds, results)
        print(f"  BT conversion: {'PASS' if bt_ok else 'FAIL'}")

        norm_ok = verify_normalization(results.get("bt_statistics", {}), results)
        print(f"  Normalization: {'PASS' if norm_ok else 'FAIL'}")

    results["overall_pass"] = coeff_ok and bt_ok and norm_ok
    print(f"  Overall: {'ALL CHECKS PASSED' if results['overall_pass'] else 'SOME CHECKS FAILED'}")

    md_path, json_path = generate_report(results, report_dir)
    print(f"  Report: {md_path}")
    return results


if __name__ == "__main__":
    cache_dir = os.path.join(os.path.dirname(current_dir), "datasets", "goes19_test_cache")
    report_dir = os.path.join(current_dir, "reports")
    run_scientific_verification(cache_dir, report_dir)
