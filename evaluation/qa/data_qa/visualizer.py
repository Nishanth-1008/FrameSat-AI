import os
import matplotlib.pyplot as plt

class TripletVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def visualize(self, t0, t1, t2, timestamps, save_name):
        """
        Visualizes a triplet of GOES-19 imagery.
        t0, t1, t2: (1, H, W) PyTorch tensors (normalized BT).
        timestamps: list of 3 datetime objects.
        save_name: name of the output PNG file.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        tensors = [t0, t1, t2]
        labels = ['t0', 't1', 't2']
        
        for i in range(3):
            # Convert (1, H, W) to (H, W) for plotting
            img = tensors[i].squeeze(0).numpy()
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            ts_str = timestamps[i].strftime('%Y-%m-%d %H:%M:%S') if timestamps[i] else "Unknown"
            
            # Compute time delta from t0
            if i > 0 and timestamps[i] and timestamps[0]:
                delta = timestamps[i] - timestamps[0]
                delta_str = f" (+{delta.total_seconds() / 60:.1f}m)"
            else:
                delta_str = ""
                
            axes[i].set_title(f"{labels[i]} - {ts_str}{delta_str}", fontsize=10)
            axes[i].axis('off')
            
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return out_path
