import numpy as np

def simulate_spacing(num_frames: int):
    # This simulates the logic from InterpolationService.interpolate_sequence
    depth = 1
    while (2**depth - 1) < num_frames:
        depth += 1
        
    full_seq_len = 2**depth - 1
    
    # Let's map full_seq_len elements to their relative time offsets in (0, 1)
    # e.g., if full_seq_len is 3, relative offsets are 0.25, 0.5, 0.75
    step = 1.0 / (full_seq_len + 1)
    full_seq_offsets = [(i + 1) * step for i in range(full_seq_len)]
    
    # Downsample
    indices = np.round(np.linspace(0, full_seq_len - 1, num_frames)).astype(int)
    selected_offsets = [full_seq_offsets[idx] for idx in indices]
    
    return selected_offsets

def main():
    print("================ VERIFYING TEMPORAL SPACING ================")
    
    test_cases = [1, 2, 3, 7, 15]
    
    for k in test_cases:
        offsets = simulate_spacing(k)
        print(f"Num frames: {k:2d} | Relative time offsets: {[round(o, 4) for o in offsets]}")
        
        # Verify symmetry
        for i in range(len(offsets)):
            opp_idx = len(offsets) - 1 - i
            # The sum of symmetric offsets should be 1.0
            assert abs((offsets[i] + offsets[opp_idx]) - 1.0) < 1e-5, f"Asymmetric spacing found for k={k} at index {i}"
            
        # Verify uniform gaps
        if len(offsets) > 1:
            gaps = np.diff(offsets)
            max_gap_diff = np.max(gaps) - np.min(gaps)
            print(f"            | Gap variation: {max_gap_diff:.6f}")
            assert max_gap_diff < 0.1, f"Uniformity check failed for k={k} with gap variation {max_gap_diff:.6f}"
            
    print("================ SPACING VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()
