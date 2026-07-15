# SEVIR Dataset Exploration Report

This report documents the exploration of the **SEVIR (Storm Event Imagery Dataset)** dataset, analyzes data access patterns, and proposes the architecture of the production-ready data layer for **FrameSat AI**.

---

## Part 1 — Dataset Discovery Summary

Based on our programmatic exploration of the SEVIR sample dataset (`SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5`):
- **Dataset Format**: HDF5 (Hierarchical Data Format v5).
- **Modality Variables**: 
  - `vis`: Raw satellite imagery dataset of shape `(6, 768, 768, 49)`.
  - `id`: Event identifier byte-strings of shape `(6,)` (e.g., `S727819`).
- **Data Type**: `int16` (signed 16-bit integer).
- **Resolution**: `768x768` pixels per frame.
- **Pixel Value Range**: Min=`-31`, Max=`9360` (Raw values before normalization, containing offset values).
- **Time Axis**: 49 frames per event (equivalent to 4 hours at 5-minute intervals).

---

## Part 2 — Answers to Exploration Questions

### 1. How is SEVIR opened?
SEVIR is stored as a hierarchical HDF5 file and is opened using the python `h5py` library:
```python
import h5py
f = h5py.File('path_to_sevir.h5', 'r')
```
For higher-level analysis, `xarray` can open it when combined with custom dimension mappings, but the standard access pattern relies directly on the raw HDF5 dataset slices.

### 2. What variables exist?
The main variables in a SEVIR HDF5 file correspond to:
- **`vis` / `ir107` / `vil` / `ir069`**: The multi-dimensional arrays containing the frame sequences for the specific modality.
- **`id`**: String identifiers for each event sequence (matching the index of the image tensor).

### 3. What are the dataset dimensions?
The image tensor has the dimensions:
`[Events, Height, Width, Frames/Time]`
For example, a `vis` shape of `(6, 768, 768, 49)` represents:
- `6` separate storm events.
- `768x768` spatial resolution.
- `49` chronological frames.

### 4. How are events represented?
An event is represented as a contiguous sequence of 49 frames of size $H \times W$ capturing a storm event. The event sequence is indexed along the first dimension (`axis=0`) of the dataset, and its unique ID is stored at the matching index in the `id` dataset.

### 5. How are timestamps stored?
Timestamps are stored in a companion catalog CSV file (`GP_Event_Catalog.csv` or similar catalogs supplied with SEVIR) which maps the `id` to the absolute start time, end time, and geographic coordinates of the event. Within the HDF5 file itself, time is represented implicitly via **relative frame indices** (0 to 48) along the last dimension.

### 6. How do we extract one frame?
To extract a single frame efficiently without loading the entire event into RAM:
```python
with h5py.File('sevir.h5', 'r') as f:
    frame = f['vis'][event_idx, :, :, frame_idx]
```

### 7. How do we extract consecutive frames?
To extract a contiguous range of frames (e.g., frames 10 to 15):
```python
with h5py.File('sevir.h5', 'r') as f:
    frames = f['vis'][event_idx, :, :, start_idx:end_idx]
```

### 8. What preprocessing is required before RIFE?
To feed SEVIR data into RIFE:
1. **Dynamic Range Handling**: Remove negative fill/offset values by clipping or subtracting minimum bounds.
2. **Normalization**: Rescale values from `int16` range to `[0.0, 1.0]` float32:
   $$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
3. **Channel Conversion**: Duplicate the single-channel grayscale array to 3 channels (RGB) to match RIFE's expected input shape: `(1, 3, H, W)`.
4. **Padding (Resize)**: Pad the spatial dimensions using `torch.nn.functional.pad` to ensure $H$ and $W$ are multiples of 32 (cropped back after inference).

### 9. Which modality should FrameSat AI use first?
We recommend starting with **VIS (Visible)**. VIS imagery has the highest spatial resolution (`768x768` compared to `192x192` for VIL/IR), providing fine-grained texture detail of cloud structures. This makes structural flaws in interpolation easily visible, providing a rigorous testbed.

### 10. Can SEVIR be streamed from AWS vs Local Cache?
- **AWS S3 Streaming**: Possible using virtual file wrappers (e.g., `s3fs` + `h5py` virtual file systems), but HDF5's block layout requires multiple small network requests for metadata, leading to high latency.
- **Local Caching**: Highly recommended. 
- **Cost Estimation**:
  - Streaming 1 TB of SEVIR data over S3 standard GET requests ($0.005/1000 requests) repeatedly during epochs can quickly rack up hundreds of dollars in API call costs, plus egress fees if training outside AWS.
  - A one-time download of 1 TB costs ~$90 in egress (if outside AWS us-east-1) and $0 thereafter.

---

## Part 3 — Access Formats Comparison

We compared three data access methods:

| Access Method | Read Speed (IOPS) | Latency | Network Cost | Setup Complexity | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AWS S3 (Streaming)** | Very Low | High (~100-300ms) | High (per GET request) | Medium | Quick exploration on small samples |
| **Local HDF5 Cache** | Extremely High | Sub-millisecond | $0 (One-time egress) | Low | Standard local/VM training nodes |
| **Zarr/xarray** | High | Low | Low (Cloud-optimized) | High (needs preprocessing) | Scale-out distributed cloud training |

### Format Recommendation
For development and production, FrameSat AI should standardize on **Local HDF5 Caching** for the MVP stage because it matches the official dataset distribution formats, requires zero format conversion preprocessing, and delivers maximum I/O throughput on NVMe-equipped GPUs.

---

## Part 4 — DataLoader Recommendation

We propose the following production API design for `SevirDataLoader`:

```python
class SevirDataLoader:
    def __init__(self, dataset_path: str, modality: str = "vis"):
        self.dataset_path = dataset_path
        self.modality = modality
        
    def load_event(self, event_idx: int) -> np.ndarray:
        """Loads a full normalized event tensor: [H, W, Frames]"""
        pass

    def get_triplet(self, event_idx: int, center_frame: int = 24) -> tuple:
        """Returns normalized (t0, t1, t2) frames for RIFE input"""
        pass

    def metadata(self, event_idx: int) -> dict:
        """Returns metadata attributes (offset, scale, bounds) for the event"""
        pass
```

### Rationale
This interface is appropriate because:
- It encapsulates the complex normalization, clipping, and scale/offset operations inside the loader, preventing leakage of preprocessing details into model code.
- Slice-based indexing ensures only necessary frames are read, optimizing memory overhead.
