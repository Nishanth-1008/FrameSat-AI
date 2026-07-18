import numpy as np

class DatasetStatistics:
    def __init__(self):
        self.num_scenes = 0
        self.num_triplets = 0
        self.time_intervals = []
        self.bt_values = []
        self.resolutions = set()
        
    def add_triplet(self, t0, t1, t2, timestamps):
        """
        Updates statistics with a new valid triplet.
        """
        self.num_triplets += 1
        
        # Track spatial resolutions
        res = f"{t0.shape[1]}x{t0.shape[2]}"
        self.resolutions.add(res)
        
        # Track time intervals
        if None not in timestamps:
            delta1 = (timestamps[1] - timestamps[0]).total_seconds() / 60.0
            delta2 = (timestamps[2] - timestamps[1]).total_seconds() / 60.0
            self.time_intervals.extend([delta1, delta2])
            
        # Sample BT values (1% of pixels to keep memory manageable) for histogram
        for tensor in [t0, t1, t2]:
            arr = tensor.numpy().flatten()
            sample = np.random.choice(arr, size=max(1, len(arr) // 100), replace=False)
            self.bt_values.extend(sample.tolist())

    def set_num_scenes(self, num):
        self.num_scenes = num

    def get_summary(self):
        """Returns a dict of aggregated statistics."""
        return {
            "num_scenes": self.num_scenes,
            "num_triplets": self.num_triplets,
            "resolutions": list(self.resolutions),
            "time_intervals_mean_min": np.mean(self.time_intervals) if self.time_intervals else 0,
            "time_intervals_std_min": np.std(self.time_intervals) if self.time_intervals else 0,
            "bt_mean_normalized": np.mean(self.bt_values) if self.bt_values else 0,
            "bt_std_normalized": np.std(self.bt_values) if self.bt_values else 0
        }
