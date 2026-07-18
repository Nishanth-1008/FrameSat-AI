import os
import re
import sys
import zipfile
import urllib.request

# Google Drive IDs for the checkpoints
GDRIVE_IDS = {
    "rife_426": "1gViYvvQrtETBgU1w8axZSsr7YUuw31uy",
    "rife_425_lite": "1zlKblGuKNatulJNFf5jdB-emp9AqGK05",
    "rife_425": "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"
}

def download_gdrive(file_id, dest_path):
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    print(f"Connecting to Google Drive for ID {file_id}...")
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            html = ""
            try:
                html = content.decode('utf-8', errors='ignore')
            except Exception:
                pass
            
            # Check for Google Drive virus warning page containing confirm=XXXX
            confirm_match = re.search(r'confirm=([a-zA-Z0-9_\-]+)', html)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                print(f"Found confirmation token: {confirm_token}. Retrying download...")
                confirm_url = f"https://docs.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                confirm_req = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(confirm_req) as confirm_response:
                    content = confirm_response.read()
            
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(content)
            print(f"Successfully downloaded to {dest_path} (Size: {len(content)/1024/1024:.2f} MB)")
            return True
    except Exception as e:
        print(f"Failed to download Google Drive ID {file_id}: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Save in weights directory to avoid interfering with production inference files
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    for name, file_id in GDRIVE_IDS.items():
        zip_path = os.path.join(weights_dir, f"{name}.zip")
        extract_dir = os.path.join(weights_dir, name)

        # Check if already extracted
        flownet_path = os.path.join(extract_dir, "flownet.pkl")
        if os.path.exists(flownet_path):
            print(f"Checkpoint {name} already exists at {extract_dir}. Skipping download.")
            continue

        print(f"\n--- Processing model {name} ---")
        if download_gdrive(file_id, zip_path):
            # Extract zip
            print(f"Extracting {zip_path} to {extract_dir}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Successfully extracted {name}!")
                
                # Cleanup zip
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except Exception as e:
                print(f"Failed to extract {zip_path}: {e}")

if __name__ == "__main__":
    main()
