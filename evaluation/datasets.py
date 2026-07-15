import os
import urllib.request
import h5py
import numpy as np

class SEVIRDataset:
    """
    Handles downloading and loading SEVIR HDF5 datasets for evaluation.
    """
    
    # Pre-defined URLs for testing small sample files
    KNOWN_URLS = {
        'vis': "https://sevir.s3.amazonaws.com/data/vis/2018/SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5",
        'vil': "https://sevir.s3.amazonaws.com/data/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5",
        'ir107': "https://sevir.s3.amazonaws.com/data/ir107/2018/SEVIR_IR107_STORMEVENTS_2018_0101_0630.h5"
    }

    def __init__(self, modality: str, download_dir: str = "datasets"):
        self.modality = modality.lower()
        self.download_dir = download_dir
        
        if self.modality not in self.KNOWN_URLS:
            raise ValueError(f"Unknown modality '{modality}'. Supported: {list(self.KNOWN_URLS.keys())}")
            
        self.url = self.KNOWN_URLS[self.modality]
        self.filename = self.url.split('/')[-1]
        self.filepath = os.path.join(self.download_dir, self.filename)
        
        # Will be populated on load
        self.h5_file = None
        self.dataset = None
        self.num_events = 0

    def download_if_needed(self):
        """Downloads the SEVIR HDF5 file if it doesn't exist locally."""
        os.makedirs(self.download_dir, exist_ok=True)
        if os.path.exists(self.filepath):
            print(f"[{self.modality.upper()}] Dataset already exists at {self.filepath}")
            return

        print(f"[{self.modality.upper()}] Downloading from {self.url}...")
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.filepath, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 1024 * 4  # 4MB blocks
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Downloaded: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end='')
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nFailed to download SEVIR sample: {e}")
            raise

    def load(self):
        """Opens the HDF5 file and assigns the target dataset."""
        self.download_if_needed()
        self.h5_file = h5py.File(self.filepath, 'r')
        
        keys = list(self.h5_file.keys())
        # Typically the dataset key matches the modality name
        dataset_key = self.modality if self.modality in keys else keys[0]
        
        self.dataset = self.h5_file[dataset_key]
        self.num_events = self.dataset.shape[0]
        print(f"[{self.modality.upper()}] Loaded '{dataset_key}' dataset with shape {self.dataset.shape}")

    def get_event_triplet(self, event_idx: int, t0_idx: int = 23, t1_idx: int = 24, t2_idx: int = 25):
        """
        Retrieves a normalized frame triplet (t0, t1, t2) from a specific event.
        
        Args:
            event_idx: Index of the event in the HDF5 file.
            t0_idx: Time index for the first frame.
            t1_idx: Time index for the target (ground truth) frame.
            t2_idx: Time index for the second frame.
            
        Returns:
            Tuple of (t0, t1, t2) as normalized numpy arrays in range [0, 1].
        """
        if self.dataset is None:
            self.load()
            
        if event_idx < 0 or event_idx >= self.num_events:
            raise IndexError(f"Event index {event_idx} out of range (0 to {self.num_events-1})")
            
        event_data = self.dataset[event_idx]
        
        d_min = event_data.min()
        d_max = event_data.max()
        
        norm_factor = float(d_max - d_min) if d_max > d_min else 255.0
        event_norm = (event_data - d_min) / norm_factor
        
        t0 = event_norm[:, :, t0_idx]
        t1 = event_norm[:, :, t1_idx]
        t2 = event_norm[:, :, t2_idx]
        
        return t0, t1, t2

    def close(self):
        """Close the HDF5 file handle."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
            self.dataset = None
