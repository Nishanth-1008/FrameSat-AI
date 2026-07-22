import os
import shutil
import zipfile

def create_bundle():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    bundle_name = os.path.join(project_root, "kaggle_bundle")
    zip_name = os.path.join(project_root, "kaggle_bundle.zip")
    
    if os.path.exists(bundle_name):
        shutil.rmtree(bundle_name)
        
    os.makedirs(bundle_name)
    
    print("Copying training/ module...")
    shutil.copytree(os.path.join(project_root, "training"), os.path.join(bundle_name, "training"), 
                    ignore=shutil.ignore_patterns("runs", "outputs", "experiments.csv", "__pycache__"))
                    
    print("Copying datasets/ package...")
    shutil.copytree(os.path.join(project_root, "datasets"), os.path.join(bundle_name, "datasets"), 
                    ignore=shutil.ignore_patterns("cache", "quarantine", "*.db", "__pycache__"))
                    
    print("Copying evaluation/ package...")
    shutil.copytree(os.path.join(project_root, "evaluation"), os.path.join(bundle_name, "evaluation"),
                    ignore=shutil.ignore_patterns("runs", "experiments", "outputs", "*.db", "__pycache__", "quarantine", "cache", "weights", "datasets"))
                
    weights_path = os.path.join(project_root, "evaluation", "weights", "rife_426")
    if os.path.exists(weights_path):
        print("Copying RIFE 4.26 weights...")
        shutil.copytree(weights_path, os.path.join(bundle_name, "evaluation", "weights", "rife_426"))
    else:
        print("Warning: RIFE 4.26 weights not found. You may need to download them first.")
        
    print("Copying models/ package...")
    shutil.copytree(os.path.join(project_root, "models"), os.path.join(bundle_name, "models"),
                    ignore=shutil.ignore_patterns("cache", "checkpoints", "weights", "__pycache__"))
        
    print("Copying framesat_kaggle/ setup scripts...")
    shutil.copytree(os.path.join(project_root, "framesat_kaggle"), os.path.join(bundle_name, "framesat_kaggle"))
    
    print("Copying shared/ package...")
    shutil.copytree(os.path.join(project_root, "shared"), os.path.join(bundle_name, "shared"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    
    print(f"Zipping {bundle_name} to {zip_name}...")
    shutil.make_archive(os.path.join(project_root, "kaggle_bundle"), 'zip', bundle_name)
    
    shutil.rmtree(bundle_name)
    print("Done! Bundle ready:", zip_name)

if __name__ == "__main__":
    create_bundle()
