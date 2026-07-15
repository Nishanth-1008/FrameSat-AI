import os
import h5py
import numpy as np

def main():
    # Make sure output directories exist
    os.makedirs("research/sevir/outputs", exist_ok=True)
    
    file_path = "evaluation/datasets/SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5"
    if not os.path.exists(file_path):
        print(f"Dataset file not found at: {file_path}")
        print("Please copy or download the VIS dataset first.")
        return

    print("================ SEVIR DATASET DISCOVERY ================")
    print(f"Format: HDF5 (File: {os.path.basename(file_path)})")
    
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        print(f"Available Keys (Variables): {keys}")
        
        for k in keys:
            ds = f[k]
            print(f"\n--- Variable: '{k}' ---")
            print(f"  Shape (Dimensions): {ds.shape}")
            print(f"  Data Type: {ds.dtype}")
            
            # Numeric check
            if np.issubdtype(ds.dtype, np.number):
                sample_event = ds[0]
                val_min = sample_event.min()
                val_max = sample_event.max()
                print(f"  Pixel Value Range (Sample Event 0): Min={val_min}, Max={val_max}")
                print(f"  Resolution (Height x Width): {ds.shape[1]}x{ds.shape[2]}")
                print(f"  Number of Events: {ds.shape[0]}")
                print(f"  Time Axis (Frames per Event): {ds.shape[3]}")
            else:
                print(f"  Values (First 3): {[val.decode('utf-8') if isinstance(val, bytes) else val for val in ds[:3]]}")
            
            # Attributes/Metadata
            if ds.attrs:
                print("  Metadata Attributes:")
                for attr_name, attr_val in ds.attrs.items():
                    print(f"    {attr_name}: {attr_val}")
                    
        # Check overall file structure and global attributes
        if f.attrs:
            print("\n--- Global Attributes ---")
            for attr_name, attr_val in f.attrs.items():
                print(f"  {attr_name}: {attr_val}")
                
    print("=========================================================")

if __name__ == "__main__":
    main()
