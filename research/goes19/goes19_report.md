# GOES-19 Channel 13 Dataset Investigation Report

This report documents the research findings on integrating GOES-19 geostationary satellite data into **FrameSat AI**, specifically targeting **ABI Channel 13** (Clean Infrared Window band, 10.3 µm) for temporal cloud interpolation.

---

## 1. GOES-19 ABI Channel 13 Structure

- **Physical Parameter**: Brightness Temperature / Radiance at 10.3 µm.
- **Purpose**: Detects cloud top temperatures, storm structures, and land surface features. It is the "clean" infrared window because atmospheric absorption (specifically water vapor) is minimal, rendering clean cloud edges.
- **Resolution**: 2 km spatial resolution at nadir.
- **Sampling Frequency**:
  - CONUS (Continental US) sectors: Every 5 minutes.
  - Full Disk sectors: Every 10 minutes.

---

## 2. File Format & Metadata

- **Format**: NetCDF4 (`.nc` files).
- **Time Indexing**:
  - NetCDF files from GOES ABI contain a single scan per file.
  - Time details are encoded directly in the filenames:
    `OR_ABI-L1b-RadF-M6C13_G19_sYYYYJJJHHMMSSn_eYYYYJJJHHMMSSn_cYYYYJJJHHMMSSn.nc`
    - `s`: Scan start timestamp (Year, Julian Day, Hour, Minute, Second).
    - `e`: Scan end timestamp.
    - `c`: File creation timestamp.
  - Inside the file, time coordinates (`time`, `time_bounds`) contain absolute epochs since 2000-01-01.

- **Primary Variables**:
  - `Rad`: 2D array of raw radiances.
  - `planck_fk1`, `planck_fk2`, `planck_bc1`, `planck_bc2`: Planck calibration constants used to convert raw radiance values into physical Brightness Temperature (Kelvin).
  - `goes_imager_projection`: Coordinate Reference System metadata (Geostationary projection).

---

## 3. Data Loading & Extraction Strategy

### Loading NetCDF Slices
We use `xarray` to open the NetCDF files due to its high-level abstraction and out-of-the-box support for coordinate projection variables:
```python
import xarray as xr
ds = xr.open_dataset('GOES_file.nc')
raw_radiance = ds['Rad'].values
```

### Planck Conversion to Brightness Temperature ($T_b$)
Raw radiance (`Rad`) is transformed to Brightness Temperature using:
$$T_b = \frac{planck\_fk2}{\ln\left(\frac{planck\_fk1}{Rad} + 1\right)} - planck\_bc1$$

### Normalization
Grayscale normalization converts Kelvin temperatures to `[0.0, 1.0]` bounds. Clean IR window temperatures generally range from $180\text{ K}$ (extremely cold, high-altitude convective storm tops) to $330\text{ K}$ (warm land surface).
$$T_{norm} = \frac{T_b - 180}{330 - 180}$$
Values are clipped to `[0.0, 1.0]` and cast to `float32`.

### Triplet Extraction
Because each file represents a single timestamp scan, triplet extraction involves:
1. Scanning a directory containing GOES-19 `.nc` files.
2. Parsing timestamps from the filenames and sorting them chronologically.
3. Grouping files into consecutive triplets `(file_t0, file_t1, file_t2)` (e.g. scans taken at 12:00, 12:05, and 12:10).
4. Loading and preprocessing frames `t0` and `t2` as input, and reserving `t1` as ground truth validation.
