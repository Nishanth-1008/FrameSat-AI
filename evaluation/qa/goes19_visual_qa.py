"""
Phase 3: Visual QA for GOES-19 triplets.

Generates:
- Side-by-side PNGs with timestamp overlays.
- Animated GIFs cycling t0 → t1 → t2.
- An HTML gallery for manual review.
"""
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import xarray as xr

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset


def make_side_by_side_png(t0, t1, t2, timestamps, save_path, triplet_idx):
    """Generate a side-by-side PNG with timestamps and deltas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tensors = [t0, t1, t2]
    labels = ['t0 (Input)', 't1 (Ground Truth)', 't2 (Input)']

    for i in range(3):
        img = tensors[i].squeeze(0).numpy()
        axes[i].imshow(img, cmap='inferno', vmin=0, vmax=1)

        ts_str = timestamps[i].strftime('%Y-%m-%d %H:%M:%S UTC') if timestamps[i] else "Unknown"

        if i > 0 and timestamps[i] and timestamps[0]:
            delta = (timestamps[i] - timestamps[0]).total_seconds() / 60.0
            delta_str = f"\nΔ = +{delta:.1f} min"
        else:
            delta_str = ""

        axes[i].set_title(f"{labels[i]}\n{ts_str}{delta_str}", fontsize=10, fontweight='bold')
        axes[i].axis('off')

    fig.suptitle(f"Triplet #{triplet_idx}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def make_animated_gif(t0, t1, t2, timestamps, save_path):
    """Generate an animated GIF cycling through t0 → t1 → t2."""
    tensors = [t0, t1, t2]
    labels = ['t0', 't1 (GT)', 't2']
    frames = []

    for i in range(3):
        img = tensors[i].squeeze(0).numpy()
        # Convert to uint8 for PIL
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L').convert('RGB')

        # Add text overlay
        draw = ImageDraw.Draw(pil_img)
        ts_str = timestamps[i].strftime('%H:%M:%S') if timestamps[i] else "N/A"
        text = f"{labels[i]} — {ts_str}"
        # Use default font
        draw.text((10, 10), text, fill=(255, 255, 0))
        frames.append(pil_img)

    # Save as GIF (loop, 500ms per frame)
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=500, loop=0, optimize=True)


def generate_html_gallery(entries, output_dir):
    """Generate an HTML gallery page linking all PNGs and GIFs."""
    html_path = os.path.join(output_dir, "visual_qa_gallery.html")

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n")
        f.write("<meta charset='UTF-8'>\n")
        f.write("<title>GOES-19 Visual QA Gallery</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }\n")
        f.write("h1 { color: #e94560; text-align: center; }\n")
        f.write("h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 6px; display: inline-block; }\n")
        f.write(".triplet { background: #16213e; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }\n")
        f.write(".triplet img { max-width: 100%; border-radius: 8px; margin: 8px 0; }\n")
        f.write(".meta { color: #a8a8a8; font-size: 0.9em; margin-top: 6px; }\n")
        f.write(".row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }\n")
        f.write(".row .col { flex: 1; min-width: 300px; }\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write(f"<h1>GOES-19 Visual QA Gallery ({len(entries)} Triplets)</h1>\n")
        f.write(f"<p style='text-align:center;'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")

        for entry in entries:
            f.write(f"<div class='triplet'>\n")
            f.write(f"<h2>Triplet #{entry['index']}</h2>\n")

            ts_strs = [t.strftime('%Y-%m-%d %H:%M:%S UTC') if t else 'Unknown' for t in entry['timestamps']]
            f.write(f"<p class='meta'>t0: {ts_strs[0]} | t1: {ts_strs[1]} | t2: {ts_strs[2]}</p>\n")

            if entry.get('deltas'):
                f.write(f"<p class='meta'>&Delta;(t0&rarr;t1): {entry['deltas'][0]:.1f}m | &Delta;(t1&rarr;t2): {entry['deltas'][1]:.1f}m</p>\n")

            f.write("<div class='row'>\n")
            f.write(f"<div class='col'><h3>Side-by-Side</h3><img src='{entry['png_name']}' alt='Triplet PNG'></div>\n")
            f.write(f"<div class='col'><h3>Animation</h3><img src='{entry['gif_name']}' alt='Triplet GIF'></div>\n")
            f.write("</div>\n")
            f.write("</div>\n")

        f.write("</body>\n</html>")

    return html_path


def run_visual_qa(cache_dir, report_dir, sample_size=50):
    """Main entry point for Phase 3."""
    print("\n" + "=" * 60)
    print("PHASE 3: Visual QA")
    print("=" * 60)

    visual_dir = os.path.join(report_dir, "visual_qa")
    os.makedirs(visual_dir, exist_ok=True)

    # Load dataset with resize to keep GIFs/PNGs manageable
    from datetime import timedelta
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
    sample_count = min(sample_size, total)
    random.seed(42)
    sample_indices = sorted(random.sample(range(total), sample_count))

    print(f"Sampling {sample_count} triplets from {total} total...")

    entries = []
    for idx_i, idx in enumerate(sample_indices):
        t0, t1, t2 = dataset[idx]
        meta = dataset.get_metadata(idx)
        timestamps = meta['timestamps']

        png_name = f"triplet_{idx:04d}.png"
        gif_name = f"triplet_{idx:04d}.gif"

        png_path = os.path.join(visual_dir, png_name)
        gif_path = os.path.join(visual_dir, gif_name)

        make_side_by_side_png(t0, t1, t2, timestamps, png_path, idx)
        make_animated_gif(t0, t1, t2, timestamps, gif_path)

        deltas = []
        if timestamps[0] and timestamps[1]:
            deltas.append((timestamps[1] - timestamps[0]).total_seconds() / 60.0)
        if timestamps[1] and timestamps[2]:
            deltas.append((timestamps[2] - timestamps[1]).total_seconds() / 60.0)

        entries.append({
            'index': idx,
            'timestamps': timestamps,
            'deltas': deltas,
            'png_name': png_name,
            'gif_name': gif_name,
        })

        if (idx_i + 1) % 10 == 0:
            print(f"  Progress: {idx_i + 1}/{sample_count}")

    html_path = generate_html_gallery(entries, visual_dir)
    print(f"  Gallery: {html_path}")
    print(f"  Generated {len(entries)} PNGs and GIFs in {visual_dir}")

    return {"sample_count": sample_count, "gallery_path": html_path, "output_dir": visual_dir}


if __name__ == "__main__":
    cache_dir = os.path.join(os.path.dirname(current_dir), "datasets", "goes19_test_cache")
    report_dir = os.path.join(current_dir, "reports")
    run_visual_qa(cache_dir, report_dir)
