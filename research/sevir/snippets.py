"""
SEVIR Dataset Access Snippets
Demonstrates common data access patterns using the h5py library.
"""

import h5py
import numpy as np

# File path to local SEVIR HDF5 VIS sample
FILE_PATH = "evaluation/datasets/SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5"

def snippet_load_event(event_idx=0):
    """
    1. Load one event
    """
    print("\n--- Snippet 1: Load one event ---")
    with h5py.File(FILE_PATH, "r") as f:
        # Load event index from 'vis' dataset
        # Dimension shape is: [Events, Height, Width, Frames]
        event_data = f['vis'][event_idx]
        print(f"Loaded event data at index {event_idx}. Shape: {event_data.shape}")
    return event_data

def snippet_load_frame(event_idx=0, frame_idx=0):
    """
    2. Load one frame
    """
    print("\n--- Snippet 2: Load one frame ---")
    with h5py.File(FILE_PATH, "r") as f:
        # Load a single frame directly using slice indices to save memory
        frame = f['vis'][event_idx, :, :, frame_idx]
        print(f"Loaded event {event_idx}, frame {frame_idx}. Shape: {frame.shape}")
    return frame

def snippet_load_triplet(event_idx=0, t0_idx=23, t1_idx=24, t2_idx=25):
    """
    3. Load a triplet (t0, t1, t2)
    """
    print("\n--- Snippet 3: Load a triplet ---")
    with h5py.File(FILE_PATH, "r") as f:
        # Extract individual frames for interpolation input and ground truth
        t0 = f['vis'][event_idx, :, :, t0_idx]
        t1 = f['vis'][event_idx, :, :, t1_idx] # target frame
        t2 = f['vis'][event_idx, :, :, t2_idx]
        print(f"Loaded triplet from event {event_idx}:")
        print(f"  t0 (Frame {t0_idx}): Shape={t0.shape}")
        print(f"  t1 (Frame {t1_idx}): Shape={t1.shape} (Ground Truth)")
        print(f"  t2 (Frame {t2_idx}): Shape={t2.shape}")
    return t0, t1, t2

def snippet_iterate_event(event_idx=0):
    """
    4. Iterate through an event frame-by-frame
    """
    print("\n--- Snippet 4: Iterate through an event ---")
    with h5py.File(FILE_PATH, "r") as f:
        event_dataset = f['vis']
        num_frames = event_dataset.shape[3]
        print(f"Iterating through {num_frames} frames of event index {event_idx}...")
        
        # Access each frame slice by slice (efficient memory usage)
        for frame_idx in range(num_frames):
            frame = event_dataset[event_idx, :, :, frame_idx]
            # Perform processing on the frame
            if frame_idx % 10 == 0:
                print(f"  Processed frame {frame_idx}/{num_frames}. Mean pixel val: {frame.mean():.2f}")

if __name__ == "__main__":
    snippet_load_event()
    snippet_load_frame()
    snippet_load_triplet()
    snippet_iterate_event()
