import os
import sys
import glob
import zipfile
import json
import shutil

# Insert root folder to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def generate_readme(output_path):
    readme_content = """# FrameSat GOES-19 ABI Channel 13 Dataset

This dataset was automatically acquired, validated, and packaged using the FrameSat-AI acquisition pipeline. It is optimized for large-scale training and fine-tuning of Frame Interpolation models (such as Practical-RIFE 4.26).

## Dataset Structure

- **cache/**: Contains validated, uncorrupted GOES-19 NetCDF Level-1b Radiance files for Channel 13.
- **metadata.db**: SQLite database indexing all scenes (including filenames, file sizes, MD5 checksums, timestamps, sectors, and scan modes) for high-performance dataloader batch querying.
- **dataset_statistics.json**: Aggregate statistics including time coverage, average file sizes, resolution summaries, and CADENCE/BT temperature distributions.
- **reports/**: Pre-training QA reports:
  - Scientific verification of Brightness Temperature conversion calculations.
  - Timeline chronology and cadence integrity checks.
  - Visual QA galleries including side-by-side plots and animated GIFs.

## Setup & Preprocessing

The NetCDF files contain raw radiance values. For training, load the files and compute the Brightness Temperature (BT) using the Planck formula:
```
T = (planck_fk2 / ln(planck_fk1/Rad + 1) - planck_bc1) / planck_bc2
```
Normalise the temperature range using `[180K, 320K]` to `[0.0, 1.0]` before passing into the model.

---
*Created by the FrameSat GOES-19 Acquisition Pipeline.*
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    zip_filename = "framesat-goes19-v1.zip"
    print(f"Creating Kaggle export dataset zip: {zip_filename}...")
    
    root_arc = "framesat-goes19-v1"
    
    # Paths in workspace
    cache_dir = "datasets/cache/goes19_cache"
    db_path = "datasets/metadata/metadata.db"
    stats_path = "datasets/statistics/dataset_statistics.json"
    reports_dir = "artifacts/reports/qa"
    readme_temp = "datasets/README_export.md"
    
    # Check if critical files exist
    if not os.path.exists(db_path):
        print(f"Error: Database index not found at {db_path}. Please run download_goes.py first.")
        sys.exit(1)
        
    generate_readme(readme_temp)
    
    # Write to zip file directly to save memory and intermediate disk space
    count_files = 0
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Add README.md
            zip_file.write(readme_temp, arcname=os.path.join(root_arc, "README.md"))
            print("  Added README.md")
            
            # 2. Add metadata.db
            zip_file.write(db_path, arcname=os.path.join(root_arc, "metadata.db"))
            print("  Added metadata.db")
            
            # 3. Add dataset_statistics.json
            if os.path.exists(stats_path):
                zip_file.write(stats_path, arcname=os.path.join(root_arc, "dataset_statistics.json"))
                print("  Added dataset_statistics.json")
                
            # 4. Add reports/ folder recursively
            if os.path.exists(reports_dir):
                for root, dirs, files in os.walk(reports_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        rel_path = os.path.relpath(filepath, reports_dir)
                        arcname = os.path.join(root_arc, "reports", rel_path)
                        zip_file.write(filepath, arcname=arcname)
                print("  Added reports directory")
                
            # 5. Add cache/ NetCDF files
            if os.path.exists(cache_dir):
                nc_files = glob.glob(os.path.join(cache_dir, "*.nc"))
                print(f"  Packaging {len(nc_files)} NetCDF files...")
                for filepath in nc_files:
                    arcname = os.path.join(root_arc, "cache", os.path.basename(filepath))
                    zip_file.write(filepath, arcname=arcname)
                    count_files += 1
                    
        print(f"Successfully exported {count_files} scenes to {zip_filename}!")
        
    except Exception as e:
        print(f"Error zipping dataset: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temp file
        if os.path.exists(readme_temp):
            os.remove(readme_temp)

if __name__ == "__main__":
    main()
