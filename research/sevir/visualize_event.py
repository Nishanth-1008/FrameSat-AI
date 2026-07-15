import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    os.makedirs("research/sevir/outputs", exist_ok=True)
    
    file_path = "evaluation/datasets/SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5"
    if not os.path.exists(file_path):
        print(f"Dataset file not found at: {file_path}")
        return

    with h5py.File(file_path, "r") as f:
        # Load event IDs if available, else use dataset keys
        event_ids = f['id'][:] if 'id' in f else None
        vis_dataset = f['vis']
        
        event_idx = 0
        event_data = vis_dataset[event_idx]
        event_id = event_ids[event_idx].decode('utf-8') if event_ids is not None else f"Event_{event_idx}"
        
        num_frames = event_data.shape[2]
        frame_shape = event_data.shape[:2]
        
        # SEVIR standard time interval is 5 minutes for VIS/IR
        time_interval_min = 5 
        
        print("================ EVENT EXPLORATION ================")
        print(f"Selected Event Index: {event_idx}")
        print(f"Event ID: {event_id}")
        print(f"Number of Frames: {num_frames}")
        print(f"Time Interval between frames: {time_interval_min} minutes")
        print(f"Shape of each frame: {frame_shape[0]}x{frame_shape[1]}")
        print(f"Data range: Min={event_data.min()}, Max={event_data.max()}")
        print("===================================================")
        
        # Normalize event for plotting
        norm_factor = float(event_data.max() - event_data.min())
        norm_data = (event_data - event_data.min()) / (norm_factor if norm_factor > 0 else 255.0)
        
        # Visualize frames 0, 1, and 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i in range(3):
            axes[i].imshow(norm_data[:, :, i], cmap='gray')
            axes[i].set_title(f"Frame {i} (t = {i * time_interval_min} mins)")
            axes[i].axis('off')
            
        plt.suptitle(f"SEVIR Event: {event_id} (Consecutive Frames)", fontsize=14)
        out_path = "research/sevir/outputs/event_0_visualization.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
