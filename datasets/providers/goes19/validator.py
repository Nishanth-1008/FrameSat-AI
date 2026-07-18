import os
import shutil
import hashlib
from datetime import datetime
# pyrefly: ignore [missing-import]
import xarray as xr

class GOES19DatasetValidator:
    def __init__(self, quarantine_dir='datasets/quarantine/goes19_quarantine'):
        self.quarantine_dir = quarantine_dir
        os.makedirs(self.quarantine_dir, exist_ok=True)

    @staticmethod
    def calculate_md5(filepath):
        """Calculate local MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_file(self, filepath):
        """
        Validates the NetCDF file for readability, variables, coefficients, and timestamps.
        Returns a tuple: (is_valid, error_message, metadata_dict)
        """
        if not os.path.exists(filepath):
            return False, "File does not exist", None

        # Check filesize
        filesize = os.path.getsize(filepath)
        if filesize == 0:
            return False, "File is empty (0 bytes)", None

        # Calculate checksum
        try:
            checksum = self.calculate_md5(filepath)
        except Exception as e:
            return False, f"Failed to compute MD5 checksum: {str(e)}", None

        # Open NetCDF
        try:
            with xr.open_dataset(filepath, engine='h5netcdf') as ds:
                # Check Rad variable
                if 'Rad' not in ds:
                    return False, "Missing 'Rad' variable in dataset", None
                
                # Check Planck coefficients
                missing_coeffs = []
                for coeff in ['planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2']:
                    if coeff not in ds:
                        missing_coeffs.append(coeff)
                if missing_coeffs:
                    return False, f"Missing Planck coefficients: {', '.join(missing_coeffs)}", None

                # Check timestamp
                t_str = ds.attrs.get('time_coverage_start', '')
                if not t_str:
                    return False, "Missing 'time_coverage_start' attribute", None
                try:
                    dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                except ValueError:
                    return False, f"Invalid 'time_coverage_start' format: {t_str}", None

                # Extract scan mode (e.g. M6, M3, etc.)
                # E.g. OR_ABI-L1b-RadC-M6C13_G19_s20242842101171.nc
                filename = os.path.basename(filepath)
                scan_mode = "M6"
                if "-" in filename:
                    parts = filename.split("-")
                    if len(parts) > 3:
                        subparts = parts[3].split("C")
                        if len(subparts) > 0:
                            scan_mode = subparts[0]

                # Check for NaNs/corruption in the Rad variable (basic check)
                rad_shape = ds['Rad'].shape
                if len(rad_shape) < 2 or rad_shape[0] == 0 or rad_shape[1] == 0:
                    return False, f"Invalid Rad variable shape: {rad_shape}", None

                metadata = {
                    "timestamp": dt.isoformat(),
                    "satellite": ds.attrs.get('platform_ID', 'GOES19'),
                    "channel": int(ds.coords.get('band_id', [13])[0]),
                    "scan_mode": scan_mode,
                    "filepath": os.path.abspath(filepath),
                    "filesize": filesize,
                    "checksum": checksum,
                    "resolution": f"{rad_shape[0]}x{rad_shape[1]}"
                }
                return True, None, metadata

        except Exception as e:
            return False, f"NetCDF file open/read corruption check failed: {str(e)}", None

    def quarantine_file(self, filepath, reason):
        """Move an invalid file to the quarantine directory and save the reason."""
        filename = os.path.basename(filepath)
        dest_path = os.path.join(self.quarantine_dir, filename)
        
        # Avoid overwriting in quarantine if file already exists
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(self.quarantine_dir, f"{base}_{int(datetime.now().timestamp())}{ext}")
            
        print(f"Quarantining file: {filename} -> {os.path.basename(dest_path)} (Reason: {reason})")
        try:
            shutil.move(filepath, dest_path)
            # Write a small companion txt file with the quarantine reason
            with open(dest_path + ".reason.txt", 'w') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Original Path: {os.path.abspath(filepath)}\n")
                f.write(f"Reason: {reason}\n")
            return dest_path
        except Exception as e:
            print(f"Failed to move file to quarantine: {str(e)}")
            return None
