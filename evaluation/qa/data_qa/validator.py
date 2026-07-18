import numpy as np

class TripletValidator:
    def __init__(self, min_bt=180.0, max_bt=320.0, max_time_gap_minutes=120):
        self.min_bt = min_bt
        self.max_bt = max_bt
        self.max_time_gap_minutes = max_time_gap_minutes
        
    def validate_tensor(self, tensor):
        """
        Validates a single normalized BT tensor.
        Checks for NaN, Inf, and range issues (assuming tensor is in [0, 1]).
        Returns a list of error strings (empty if valid).
        """
        errors = []
        if np.isnan(tensor.numpy()).any():
            errors.append("NaN values detected in tensor.")
        if np.isinf(tensor.numpy()).any():
            errors.append("Inf values detected in tensor.")
            
        # Since the tensor is normalized to [0,1], values outside [0,1] suggest abnormal source BT
        if tensor.min() < 0.0 or tensor.max() > 1.0:
            errors.append(f"Abnormal normalized BT range detected: min={tensor.min():.2f}, max={tensor.max():.2f}")
            
        return errors
        
    def validate_triplet(self, t0, t1, t2, timestamps):
        """
        Validates a triplet sequence for time gaps and duplication.
        Returns a list of error strings.
        """
        errors = []
        
        # Check missing timestamps
        if None in timestamps:
            errors.append("Missing timestamp in triplet metadata.")
            return errors # Cannot compute time gaps
            
        # Check time gaps
        delta_t1 = (timestamps[1] - timestamps[0]).total_seconds() / 60.0
        delta_t2 = (timestamps[2] - timestamps[1]).total_seconds() / 60.0
        
        if delta_t1 > self.max_time_gap_minutes or delta_t2 > self.max_time_gap_minutes:
            errors.append(f"Time gap exceeds {self.max_time_gap_minutes}m: delta1={delta_t1:.1f}m, delta2={delta_t2:.1f}m.")
            
        if delta_t1 <= 0 or delta_t2 <= 0:
            errors.append("Non-chronological or duplicate timestamps detected.")
            
        # Duplicate scene detection via exact tensor match
        if np.array_equal(t0.numpy(), t1.numpy()) or np.array_equal(t1.numpy(), t2.numpy()):
            errors.append("Duplicate scene content detected in sequential frames.")
            
        return errors
