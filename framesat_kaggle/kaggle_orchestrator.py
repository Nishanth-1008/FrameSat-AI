import os
import sys
import json
import glob
import shutil
import datetime
import subprocess
import torch
import platform

class KaggleOrchestrator:
    def __init__(self):
        self.bundle_path = None
        self.dataset_path = None
        self.cache_dir = None
        self.metadata_db = None
        self.weights_dir = None
        self.resolved_config_path = "/kaggle/working/train_config_resolved.json"
        self.manifest_path = "/kaggle/working/runtime_manifest.json"
        
        # Detected stats
        self.gpu_model = "CPU"
        self.pytorch_version = torch.__version__
        self.cuda_version = "N/A"
        self.rife_version = "Unknown"
        self.git_commit = "N/A"
        
    def setup_environment(self):
        print("=================================================")
        print(" FrameSat-AI Orchestrator: Setting up Environment")
        print("=================================================")
        
        # GPU detection
        if torch.cuda.is_available():
            self.gpu_model = torch.cuda.get_device_name(0)
            self.cuda_version = torch.version.cuda
            print(f"[OK] CUDA Available: {self.gpu_model} (CUDA {self.cuda_version})")
        else:
            print("[WARN] CUDA NOT available. Training will run on CPU.")
            
        print(f"PyTorch Version: {self.pytorch_version}")
        print(f"Python Version: {platform.python_version()}")
        return {
            "gpu": self.gpu_model,
            "pytorch": self.pytorch_version,
            "cuda": self.cuda_version,
            "python": platform.python_version()
        }

    def discover_resources(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Resource Discovery")
        print("=================================================")
        
        # Find bundle path
        for root, dirs, files in os.walk("/kaggle/input"):
            if "training" in dirs and "evaluation" in dirs:
                self.bundle_path = root
                break
        if not self.bundle_path:
            if os.path.exists("/kaggle/working/training/train.py"):
                self.bundle_path = "/kaggle/working"
                
        if not self.bundle_path:
            raise RuntimeError("FrameSat-AI source code bundle not found.")
        print(f"[OK] Source Bundle Found: {self.bundle_path}")
        
        # Find dataset path
        for root, dirs, files in os.walk("/kaggle/input"):
            if "cache" in dirs or "goes19_cache" in dirs:
                self.dataset_path = root
                break
        if not self.dataset_path:
            raise RuntimeError("GOES-19 dataset path not found.")
        print(f"[OK] Dataset Path Found: {self.dataset_path}")
        
        # Resolve sub-paths
        self.cache_dir = os.path.join(self.dataset_path, "cache")
        if not os.path.exists(self.cache_dir):
            self.cache_dir = os.path.join(self.dataset_path, "goes19_cache")
            
        self.metadata_db = os.path.join(self.dataset_path, "metadata", "metadata.db")
        if not os.path.exists(self.metadata_db):
            self.metadata_db = os.path.join(self.dataset_path, "metadata.db")
            
        # Find weights dir
        self.weights_dir = os.path.join(self.bundle_path, "weights", "rife_426", "train_log")
        if not os.path.exists(self.weights_dir):
            self.weights_dir = os.path.join(self.bundle_path, "evaluation", "weights", "rife_426", "train_log")
            
        # Git commit discovery
        try:
            self.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.bundle_path, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            self.git_commit = "No Git repo / Commit unavailable"

        # Resolve RIFE version
        for part in self.weights_dir.split(os.sep):
            if "rife_" in part:
                version_raw = part.replace("rife_", "")
                if len(version_raw) >= 3:
                    self.rife_version = f"{version_raw[0]}.{version_raw[1:]}"
                else:
                    self.rife_version = version_raw
                break
                
        print(f"Resolved RIFE version: {self.rife_version}")
        print(f"Git Commit Hash: {self.git_commit}")

    def validate_environment(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Validations")
        print("=================================================")
        
        validation_failed = False
        
        # 1. Dataset cache check
        nc_files = glob.glob(os.path.join(self.cache_dir, "*.nc"))
        print(f"Scenes found in cache: {len(nc_files)}")
        if len(nc_files) < 3:
            print(f"[FAIL] Insufficient NetCDF scenes ({len(nc_files)}) for triplets.")
            validation_failed = True
        else:
            print("[PASS] Dataset Cache verified.")
            
        # 2. Weights directory verification
        required_files = [
            "flownet.pkl", "IFNet_HDv3.py", "RIFE_HDv3.py", 
            "refine.py"
        ]
        print(f"Verifying weights directory: {self.weights_dir}")
        for file in required_files:
            file_path = os.path.join(self.weights_dir, file)
            if not os.path.exists(file_path):
                print(f"[FAIL] Required weight/model file missing: {file}")
                validation_failed = True
            else:
                print(f"[PASS] File verified: {file}")
                
        # 3. Disk space check
        _, _, free = shutil.disk_usage("/kaggle/working")
        free_gb = free / (1024 ** 3)
        print(f"Free disk space: {free_gb:.2f} GB")
        if free_gb < 5.0:
            print("[FAIL] Insufficient disk space in /kaggle/working.")
            validation_failed = True
            
        if validation_failed:
            raise RuntimeError("Environment verification failed. Aborting training run.")
        print("[OK] All pre-flight validations passed.")

    def resolve_runtime_config(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Manifest & Config")
        print("=================================================")
        
        # Load baseline config
        config_template_path = os.path.join(self.bundle_path, "training", "configs", "train_rife426.json")
        if not os.path.exists(config_template_path):
            config_template_path = os.path.join(self.bundle_path, "kaggle", "train_kaggle.json")
            
        with open(config_template_path, "r") as f:
            config = json.load(f)
            
        # Override fields
        config["dataset_path"] = self.cache_dir
        config["quarantine_dir"] = os.path.join(os.path.dirname(self.cache_dir), "quarantine")
        config["pretrained_weights"] = self.weights_dir
        
        # Set absolute output folder
        output_base_dir = "/kaggle/working/outputs"
        config["output_dir"] = output_base_dir
        
        # Check auto-resume capabilities
        resume_checkpoint = None
        runs_dir = os.path.join(output_base_dir, "runs")
        if os.path.exists(runs_dir):
            all_runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
            if all_runs:
                latest_run = all_runs[-1]
                latest_pth = os.path.join(latest_run, "latest.pth")
                if os.path.exists(latest_pth):
                    resume_checkpoint = latest_pth
                    
        if resume_checkpoint:
            print(f"[INFO] Auto-Resume detected! Resuming from: {resume_checkpoint}")
            config["resume"] = True
            config["resume_checkpoint"] = resume_checkpoint
        else:
            config["resume"] = False
            config["resume_checkpoint"] = ""
            
        # Write training configuration
        with open(self.resolved_config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        # Create manifest metadata
        manifest = {
            "gpu": self.gpu_model,
            "cuda": self.cuda_version,
            "pytorch": self.pytorch_version,
            "python": platform.python_version(),
            "rife_version": self.rife_version,
            "git_commit": self.git_commit,
            "dataset_path": self.cache_dir,
            "metadata_db": self.metadata_db,
            "weights_dir": self.weights_dir,
            "output_dir": output_base_dir,
            "timestamp": datetime.datetime.now().isoformat(),
            "resume_triggered": bool(resume_checkpoint)
        }
        
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        # Print Manifest
        print("=================================================")
        print("FrameSat AI Runtime Manifest")
        print("=================================================")
        print(f"GPU:\n  {self.gpu_model}")
        print(f"PyTorch:\n  {self.pytorch_version}")
        print(f"CUDA:\n  {self.cuda_version}")
        print(f"Project:\n  {self.bundle_path}")
        print(f"Dataset:\n  {self.cache_dir}")
        print(f"Metadata:\n  {self.metadata_db}")
        print(f"Weights:\n  {self.weights_dir}")
        print(f"Output:\n  {output_base_dir}")
        print("=================================================")
        
        return config

    def launch_training(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Training Launch")
        print("=================================================")
        
        train_script = os.path.join(self.bundle_path, "training", "train.py")
        print(f"Executing: python {train_script} --config {self.resolved_config_path}")
        
        if self.bundle_path not in sys.path:
            sys.path.insert(0, self.bundle_path)
            
        cmd = [sys.executable, train_script, "--config", self.resolved_config_path]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream logs in real-time
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Training pipeline crashed with exit code: {process.returncode}")
        print("[OK] Training execution succeeded.")

    def evaluate(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Post-Training Evaluation")
        print("=================================================")
        
        # Locate the run results
        runs_dir = "/kaggle/working/outputs/runs"
        all_runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
        if not all_runs:
            print("[WARN] No runs folder found under outputs. Skipping evaluation.")
            return
            
        latest_run = all_runs[-1]
        best_checkpoint = os.path.join(latest_run, "best.pth")
        if not os.path.exists(best_checkpoint):
            print(f"[WARN] No best checkpoint found at {best_checkpoint}. Skipping evaluation.")
            return
            
        print(f"Found best checkpoint for evaluation: {best_checkpoint}")
        
        # Generate eval config
        eval_config = {
            "experiment_name": "eval_kaggle_finetune",
            "dataset_path": self.cache_dir,
            "weights": best_checkpoint,
            "events": 10,
            "save_predictions": True
        }
        
        eval_config_path = "/kaggle/working/eval_config.json"
        with open(eval_config_path, "w") as f:
            json.dump(eval_config, f, indent=4)
            
        # Parse template end dates
        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)
            
        start_date = config.get("start_date", "2024-10-10T21:00:00")
        end_date = config.get("end_date", "2024-10-14T09:00:00")
        
        # Run the evaluator programmatically via inline script
        eval_script = f"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, '{self.bundle_path}')

from evaluation.evaluator import Evaluator
from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
from models.rife.interpolator import RIFEInterpolator

class GOES19EvaluationDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.modality = 'vis'
    def load(self):
        pass
    @property
    def num_events(self):
        return len(self.dataset)
    def get_event_triplet(self, idx):
        t0, gt, t2 = self.dataset[idx]
        return t0.squeeze(0).numpy(), gt.squeeze(0).numpy(), t2.squeeze(0).numpy()

print('Loading RIFE Model...')
model = RIFEInterpolator()
model.load_weights('{latest_run}')

raw_dataset = GOES19TripletDataset(
    start_date=datetime.fromisoformat('{start_date}'),
    end_date=datetime.fromisoformat('{end_date}'),
    cache_dir='{self.cache_dir}',
    split='val'
)
dataset = GOES19EvaluationDatasetWrapper(raw_dataset)

with open('{eval_config_path}', 'r') as f:
    eval_config = json.load(f)

evaluator = Evaluator(model, dataset, eval_config)
evaluator.run()
"""
        eval_run_path = "/kaggle/working/run_eval.py"
        with open(eval_run_path, "w") as f:
            f.write(eval_script)
            
        subprocess.check_call([sys.executable, eval_run_path])
        print("[OK] Scientific evaluation complete.")

    def package_outputs(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Exporting Artifacts")
        print("=================================================")
        
        export_root = "/kaggle/working/experiment_001"
        os.makedirs(export_root, exist_ok=True)
        
        # Find runs
        runs_dir = "/kaggle/working/outputs/runs"
        all_runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
        
        latest_run = None
        if all_runs:
            latest_run = all_runs[-1]
            
        # Copy configs
        if os.path.exists(self.resolved_config_path):
            shutil.copy(self.resolved_config_path, os.path.join(export_root, "config.json"))
        if os.path.exists(self.manifest_path):
            shutil.copy(self.manifest_path, os.path.join(export_root, "runtime_manifest.json"))
            
        # Copy logs
        if latest_run and os.path.exists(os.path.join(latest_run, "training.log")):
            shutil.copy(os.path.join(latest_run, "training.log"), os.path.join(export_root, "training.log"))
            
        # Copy checkpoints
        checkpoints_dir = os.path.join(export_root, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        if latest_run:
            for pth in glob.glob(os.path.join(latest_run, "*.pth")):
                shutil.copy(pth, checkpoints_dir)
                
        # Copy tensorboard
        tb_dir = os.path.join(export_root, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        if latest_run and os.path.exists(os.path.join(latest_run, "tensorboard")):
            shutil.copytree(os.path.join(latest_run, "tensorboard"), tb_dir, dirs_exist_ok=True)
            
        # Copy evaluation
        eval_dir = os.path.join(export_root, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Check for generated markdown and curves
        if latest_run:
            for report in ["training_report.md", "baseline_vs_finetuned.md", "README.md", "loss_curve.png", "validation_curves.png"]:
                report_path = os.path.join(latest_run, report)
                if os.path.exists(report_path):
                    shutil.copy(report_path, eval_dir)
                    
        # Copy predictions
        pred_dir = os.path.join(export_root, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        if latest_run and os.path.exists(os.path.join(latest_run, "sample_predictions")):
            shutil.copytree(os.path.join(latest_run, "sample_predictions"), pred_dir, dirs_exist_ok=True)
            
        # Create tar file
        tar_path = "/kaggle/working/experiment_001.tar.gz"
        print(f"Compressing export directory into: {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_root, arcname="experiment_001")
            
        print(f"[OK] Pack complete. Tar file created: {tar_path}")
        
        # Read final metrics if available
        epochs = "N/A"
        best_psnr = "N/A"
        best_ssim = "N/A"
        
        if latest_run:
            metrics_json = os.path.join(latest_run, "metrics.json")
            if not os.path.exists(metrics_json):
                metrics_json = os.path.join(latest_run, "metrics.csv")
                
            if os.path.exists(metrics_json) and metrics_json.endswith(".json"):
                try:
                    with open(metrics_json, "r") as f:
                        m_data = json.load(f)
                        epochs = len(m_data)
                        best_psnr = max(m["psnr"] for m in m_data)
                        best_ssim = max(m["ssim"] for m in m_data)
                except Exception:
                    pass
            elif os.path.exists(metrics_json) and metrics_json.endswith(".csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(metrics_json)
                    epochs = len(df)
                    best_psnr = df["psnr"].max()
                    best_ssim = df["ssim"].max()
                except Exception:
                    pass
                    
        # Print Notebook summary
        print("\n=================================================")
        print("Training Complete")
        print("=================================================")
        print(f"Epochs:\n  {epochs}")
        print(f"Best PSNR:\n  {best_psnr}")
        print(f"Best SSIM:\n  {best_ssim}")
        print(f"Output:\n  {export_root}")
        print(f"Checkpoint:\n  {os.path.join(checkpoints_dir, 'best.pth')}")
        print(f"Evaluation:\n  {os.path.join(eval_dir, 'training_report.md')}")
        print("=================================================")
