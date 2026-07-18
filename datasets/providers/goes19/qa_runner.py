import os
import sys
import json
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# pyrefly: ignore [missing-import]
import xarray as xr
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..', '..', '..')))

from evaluation.qa.data_qa.visualizer import TripletVisualizer
from evaluation.qa.data_qa.validator import TripletValidator
from evaluation.qa.data_qa.statistics import DatasetStatistics
from evaluation.qa.data_qa.report import QAReport

class GOES19QARunner:
    def __init__(self, db_manager, reports_dir):
        self.db_manager = db_manager
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.visual_qa_dir = os.path.join(self.reports_dir, "visual_qa")
        os.makedirs(self.visual_qa_dir, exist_ok=True)

    def run_scientific_verification(self, sample_scene):
        """Phase 1: scientific verification on a single representative scene."""
        print("  Running Scientific Verification...")
        filepath = sample_scene['filepath']
        results = {"file_analyzed": os.path.basename(filepath)}
        
        try:
            with xr.open_dataset(filepath, engine='h5netcdf') as ds:
                fk1 = float(ds['planck_fk1'].values)
                fk2 = float(ds['planck_fk2'].values)
                bc1 = float(ds['planck_bc1'].values)
                bc2 = float(ds['planck_bc2'].values)
                
                results["planck_coefficients"] = {"fk1": fk1, "fk2": fk2, "bc1": bc1, "bc2": bc2}
                
                # Check coefficients ranges
                checks = [
                    {"name": "planck_fk1 in range", "pass": 8000.0 <= fk1 <= 15000.0, "value": fk1},
                    {"name": "planck_fk2 in range", "pass": 1300.0 <= fk2 <= 1500.0, "value": fk2},
                    {"name": "planck_bc1 in range", "pass": 0.0 <= bc1 <= 1.5, "value": bc1},
                    {"name": "planck_bc2 in range", "pass": 0.99 <= bc2 <= 1.01, "value": bc2}
                ]
                results["coefficient_checks"] = checks
                
                # Verify BT formula
                rad = ds['Rad'].values
                valid_mask = rad > 0
                bt = np.zeros_like(rad)
                bt[valid_mask] = (fk2 / np.log((fk1 / rad[valid_mask]) + 1) - bc1) / bc2
                
                bt_valid = bt[valid_mask]
                if len(bt_valid) > 0:
                    bt_min, bt_max = float(bt_valid.min()), float(bt_valid.max())
                    bt_mean = float(bt_valid.mean())
                    results["bt_statistics"] = {"min_K": round(bt_min, 2), "max_K": round(bt_max, 2), "mean_K": round(bt_mean, 2)}
                    
                    bt_checks = [
                        {"name": "No NaN in valid BT", "pass": not np.any(np.isnan(bt_valid))},
                        {"name": "No Inf in valid BT", "pass": not np.any(np.isinf(bt_valid))},
                        {"name": "BT within physical range (170K-340K)", "pass": 170.0 <= bt_min and bt_max <= 340.0},
                        {"name": "Mean BT within typical range (190K-310K)", "pass": 190.0 <= bt_mean <= 310.0}
                    ]
                    results["conversion_checks"] = bt_checks
                else:
                    results["conversion_checks"] = [{"name": "Valid pixels check", "pass": False}]
                    
        except Exception as e:
            results["error"] = str(e)
            results["overall_pass"] = False
            return results

        results["overall_pass"] = all(c["pass"] for c in results.get("coefficient_checks", [])) and all(c["pass"] for c in results.get("conversion_checks", []))
        
        # Write report
        md_path = os.path.join(self.reports_dir, "scientific_verification.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# GOES-19 ABI Channel 13 — Scientific Verification Report\n\n")
            f.write(f"- **File analyzed:** `{results['file_analyzed']}`\n")
            f.write(f"- **Overall status:** {'PASSED' if results['overall_pass'] else 'FAILED'}\n\n")
            f.write("### Planck Coefficients\n")
            for c in results.get("coefficient_checks", []):
                f.write(f"- {'✅' if c['pass'] else '❌'} {c['name']}: {c['value']:.4f}\n")
            f.write("\n### Temperature Verification\n")
            if "bt_statistics" in results:
                stats = results["bt_statistics"]
                f.write(f"- Min: {stats['min_K']}K, Max: {stats['max_K']}K, Mean: {stats['mean_K']}K\n")
            for c in results.get("conversion_checks", []):
                f.write(f"- {'✅' if c['pass'] else '❌'} {c['name']}\n")
                
        return results

    def _process_to_tensor(self, filepath):
        """Open NetCDF, calculate BT and return normalized tensor."""
        with xr.open_dataset(filepath, engine='h5netcdf') as ds:
            rad = ds['Rad'].values
            planck_fk1 = ds['planck_fk1'].values
            planck_fk2 = ds['planck_fk2'].values
            planck_bc1 = ds['planck_bc1'].values
            planck_bc2 = ds['planck_bc2'].values
            
            valid_mask = rad > 0
            bt = np.zeros_like(rad)
            bt[valid_mask] = (planck_fk2 / np.log((planck_fk1 / rad[valid_mask]) + 1) - planck_bc1) / planck_bc2
            bt[~valid_mask] = np.nanmin(bt[valid_mask]) if np.any(valid_mask) else 200.0
            
            bt_min, bt_max = 180.0, 320.0
            bt_norm = np.clip((bt - bt_min) / (bt_max - bt_min), 0, 1)
            return torch.from_numpy(bt_norm).float().unsqueeze(0)

    def make_animated_gif(self, t0, t1, t2, timestamps, save_path):
        """Generate animated GIF for the triplet."""
        tensors = [t0, t1, t2]
        labels = ['t0', 't1 (GT)', 't2']
        frames = []
        for i in range(3):
            img = tensors[i].squeeze(0).numpy()
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L').convert('RGB')
            draw = ImageDraw.Draw(pil_img)
            ts_str = timestamps[i].strftime('%H:%M:%S') if timestamps[i] else "N/A"
            draw.text((10, 10), f"{labels[i]} - {ts_str}", fill=(255, 255, 0))
            frames.append(pil_img)
            
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=True)

    def run_qa_pipeline(self, sample_size=50):
        """Run the full triplet validation, statistics aggregation, and visual gallery QA."""
        print("  Running QA Pipeline...")
        scenes = self.db_manager.get_all_scenes(sorted_chronologically=True)
        if len(scenes) < 3:
            print("  Warning: Not enough scenes to form triplets for QA.")
            return
            
        # Form triplets
        triplets = []
        for i in range(len(scenes) - 2):
            triplets.append((scenes[i], scenes[i+1], scenes[i+2]))
            
        # Run scientific verification on the first scene
        self.run_scientific_verification(scenes[0])
        
        # Setup reports and statistics
        visualizer = TripletVisualizer(output_dir=self.visual_qa_dir)
        validator = TripletValidator(min_bt=180.0, max_bt=320.0, max_time_gap_minutes=120)
        stats = DatasetStatistics()
        report = QAReport(output_dir=self.reports_dir)
        stats.set_num_scenes(len(scenes))
        
        # Sampling indices for Visual QA
        sample_count = min(sample_size, len(triplets))
        random.seed(42)
        visual_indices = set(random.sample(range(len(triplets)), sample_count))
        
        gallery_entries = []
        
        for i, triplet in enumerate(triplets):
            try:
                s0, s1, s2 = triplet
                t0 = self._process_to_tensor(s0['filepath'])
                t1 = self._process_to_tensor(s1['filepath'])
                t2 = self._process_to_tensor(s2['filepath'])
                
                timestamps = [datetime.fromisoformat(s['timestamp']) for s in triplet]
                
                # Run validation
                err_t0 = validator.validate_tensor(t0)
                err_t1 = validator.validate_tensor(t1)
                err_t2 = validator.validate_tensor(t2)
                err_trip = validator.validate_triplet(t0, t1, t2, timestamps)
                
                all_errs = err_t0 + err_t1 + err_t2 + err_trip
                for e in all_errs:
                    report.add_error(i, e)
                    
                if not all_errs:
                    stats.add_triplet(t0, t1, t2, timestamps)
                    
                # Visual QA if sampled
                if i in visual_indices:
                    png_name = f"triplet_{i:04d}.png"
                    gif_name = f"triplet_{i:04d}.gif"
                    
                    # Generate side-by-side PNG
                    png_path = visualizer.visualize(t0, t1, t2, timestamps, png_name)
                    report.add_visualization(png_path)
                    
                    # Generate GIF
                    gif_path = os.path.join(self.visual_qa_dir, gif_name)
                    self.make_animated_gif(t0, t1, t2, timestamps, gif_path)
                    
                    deltas = [
                        (timestamps[1] - timestamps[0]).total_seconds() / 60.0,
                        (timestamps[2] - timestamps[1]).total_seconds() / 60.0
                    ]
                    gallery_entries.append({
                        "index": i,
                        "timestamps": timestamps,
                        "deltas": deltas,
                        "png_name": png_name,
                        "gif_name": gif_name
                    })
                    
            except Exception as e:
                report.add_error(i, f"Error processing triplet: {str(e)}")
                
        # Export general report
        report.set_statistics(stats.get_summary())
        report.export()
        
        # Export HTML gallery
        self._generate_html_gallery(gallery_entries)
        print(f"  QA Pipeline complete. Reports exported to {self.reports_dir}")

    def _generate_html_gallery(self, entries):
        html_path = os.path.join(self.visual_qa_dir, "visual_qa_gallery.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>GOES-19 Visual QA Gallery</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { color: #e94560; text-align: center; }
h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 6px; display: inline-block; }
.triplet { background: #16213e; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
.triplet img { max-width: 100%; border-radius: 8px; margin: 8px 0; }
.meta { color: #a8a8a8; font-size: 0.9em; margin-top: 6px; }
.row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.row .col { flex: 1; min-width: 300px; }
</style>
</head>
<body>
<h1>GOES-19 Visual QA Gallery</h1>
""")
            for entry in entries:
                f.write(f"""<div class='triplet'>
<h2>Triplet #{entry['index']}</h2>
<p class='meta'>t0: {entry['timestamps'][0].isoformat()} | t1: {entry['timestamps'][1].isoformat()} | t2: {entry['timestamps'][2].isoformat()}</p>
<p class='meta'>&Delta;(t0&rarr;t1): {entry['deltas'][0]:.1f}m | &Delta;(t1&rarr;t2): {entry['deltas'][1]:.1f}m</p>
<div class='row'>
<div class='col'><h3>Side-by-Side</h3><img src='{entry['png_name']}' alt='Triplet PNG'></div>
<div class='col'><h3>Animation</h3><img src='{entry['gif_name']}' alt='Triplet GIF'></div>
</div>
</div>
""")
            f.write("</body>\n</html>")
