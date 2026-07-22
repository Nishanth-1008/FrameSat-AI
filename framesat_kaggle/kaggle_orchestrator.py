import os
import sys
import json
import glob
import shutil
import datetime
import subprocess
import torch
import platform
import tarfile

# Standard Experiment Presets
FAST_DEBUG = {
    "epochs": 2,
    "batch_size": 2,
    "train_resize": [256, 256]
}

FULL_TRAINING = {
    "epochs": 100,
    "batch_size": 8,
    "train_resize": [512, 512]
}

PAPER_RESULTS = {
    "epochs": 150,
    "batch_size": 8,
    "train_resize": [512, 512],
    "save_predictions": True
}

class KaggleOrchestrator:
    def _print_header(self, title):
        CYAN = "\033[36m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        width = 65
        padding = (width - len(title) - 2) // 2
        p_left = " " * padding
        p_right = " " * (width - len(title) - 2 - padding)
        try:
            print(f"\n{CYAN}┌{'─' * (width - 2)}┐{RESET}")
            print(f"{CYAN}│{BOLD}{p_left}{title}{p_right}{CYAN}│{RESET}")
            print(f"{CYAN}└{'─' * (width - 2)}┘{RESET}")
        except UnicodeEncodeError:
            print(f"\n{CYAN}+{'=' * (width - 2)}+{RESET}")
            print(f"{CYAN}|{BOLD}{p_left}{title}{p_right}{CYAN}|{RESET}")
            print(f"{CYAN}+{'=' * (width - 2)}+{RESET}")

    def __init__(self, config_overrides=None):
        self.config_overrides = config_overrides or {}
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

        # Resolved device string, set once in setup_environment()
        self.device = "cpu"

    def setup_environment(self):
        self._print_header("FrameSat-AI Orchestrator: Setting up Environment")

        # GPU detection
        if torch.cuda.is_available():
            self.gpu_model = torch.cuda.get_device_name(0)
            self.cuda_version = torch.version.cuda
            self.device = "cuda"
            print(f"\033[32m✔\033[0m CUDA Available: \033[1;32m{self.gpu_model}\033[0m (CUDA {self.cuda_version})")
        else:
            self.device = "cpu"
            print("\033[33m⚠\033[0m CUDA NOT available. Training will run on CPU.")

        print(f"PyTorch Version: \033[36m{self.pytorch_version}\033[0m")
        print(f"Python Version:  \033[36m{platform.python_version()}\033[0m")
        return {
            "gpu": self.gpu_model,
            "pytorch": self.pytorch_version,
            "cuda": self.cuda_version,
            "python": platform.python_version(),
            "device": self.device
        }

    def discover_resources(self):
        self._print_header("FrameSat-AI Orchestrator: Resource Discovery")

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
        print(f"\033[32m✔\033[0m Source Bundle Found: \033[34m{self.bundle_path}\033[0m")

        # Find dataset path
        for root, dirs, files in os.walk("/kaggle/input"):
            if "cache" in dirs or "goes19_cache" in dirs:
                self.dataset_path = root
                break
        if not self.dataset_path:
            raise RuntimeError("GOES-19 dataset path not found.")
        print(f"\033[32m✔\033[0m Dataset Path Found:  \033[34m{self.dataset_path}\033[0m")

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

        print(f"\033[32m✔\033[0m Resolved RIFE version: \033[36m{self.rife_version}\033[0m")
        print(f"\033[32m✔\033[0m Git Commit Hash:      \033[36m{self.git_commit}\033[0m")

    def validate_environment(self):
        self._print_header("FrameSat-AI Orchestrator: Validations")

        validation_failed = False

        # 1. CUDA enforcement — fail fast rather than silently training on CPU
        if not torch.cuda.is_available():
            print("\033[31m✘\033[0m CUDA is NOT available. This pipeline requires a GPU runtime.")
            print("\033[31m✘\033[0m Refusing to proceed with CPU-only training (set device manually to override).")
            validation_failed = True
        else:
            device_count = torch.cuda.device_count()
            print(f"\033[32m✔\033[0m CUDA verified: {torch.cuda.get_device_name(0)} "
                  f"({device_count} device{'s' if device_count != 1 else ''} visible)")

        # 2. Dataset cache check
        nc_files = glob.glob(os.path.join(self.cache_dir, "*.nc"))
        print(f"Scenes found in cache: {len(nc_files)}")
        if len(nc_files) < 3:
            print(f"\033[31m✘\033[0m Insufficient NetCDF scenes ({len(nc_files)}) for triplets.")
            validation_failed = True
        else:
            print("\033[32m✔\033[0m Dataset Cache verified.")

        # 3. Weights directory verification
        required_files = [
            "flownet.pkl"
        ]
        print(f"Verifying weights directory: {self.weights_dir}")
        for file in required_files:
            file_path = os.path.join(self.weights_dir, file)
            if not os.path.exists(file_path):
                print(f"\033[31m✘\033[0m Required weight/model file missing: {file}")
                validation_failed = True
            else:
                print(f"\033[32m✔\033[0m File verified: {file}")

        # 4. Disk space check
        _, _, free = shutil.disk_usage("/kaggle/working")
        free_gb = free / (1024 ** 3)
        print(f"Free disk space: {free_gb:.2f} GB")
        if free_gb < 5.0:
            print("\033[31m✘\033[0m Insufficient disk space in /kaggle/working.")
            validation_failed = True

        if validation_failed:
            raise RuntimeError("Environment verification failed. Aborting training run.")
        print("\033[32m✔\033[0m All pre-flight validations passed.")

    def resolve_runtime_config(self, overrides=None):
        self._print_header("FrameSat-AI Orchestrator: Manifest & Config")

        active_overrides = {}
        if self.config_overrides:
            active_overrides.update(self.config_overrides)
        if overrides:
            active_overrides.update(overrides)
        if not active_overrides:
            main_mod = sys.modules.get("__main__")
            if main_mod and hasattr(main_mod, "CONFIG_OVERRIDES"):
                notebook_overrides = getattr(main_mod, "CONFIG_OVERRIDES")
                if isinstance(notebook_overrides, dict):
                    active_overrides.update(notebook_overrides)

        # Normalize alias overrides for backend compatibility
        if "mixed_precision" in active_overrides:
            active_overrides["use_amp"] = active_overrides["mixed_precision"]
        if "scheduler" in active_overrides:
            active_overrides["lr_scheduler"] = active_overrides["scheduler"]
        if "num_eval_events" in active_overrides:
            active_overrides["events"] = active_overrides["num_eval_events"]

        config_template_path = os.path.join(self.bundle_path, "training", "configs", "train_rife426.json")
        if not os.path.exists(config_template_path):
            config_template_path = os.path.join(self.bundle_path, "framesat_kaggle", "train_kaggle.json")

        with open(config_template_path, "r") as f:
            config = json.load(f)

        # Apply notebook overrides first over default template
        if active_overrides:
            config.update(active_overrides)
            try:
                print(f"\033[32m✔\033[0m Applied {len(active_overrides)} notebook configuration override(s):")
            except UnicodeEncodeError:
                print(f"[OK] Applied {len(active_overrides)} notebook configuration override(s):")
            for k, v in active_overrides.items():
                try:
                    print(f"  └─ \033[36m{k:<20}\033[0m: {v}")
                except UnicodeEncodeError:
                    print(f"  |- \033[36m{k:<20}\033[0m: {v}")

        # Managed infrastructure fields
        config["dataset_path"] = self.cache_dir
        config["quarantine_dir"] = os.path.join(os.path.dirname(self.cache_dir), "quarantine")
        config["pretrained_weights"] = self.weights_dir

        config["device"] = self.device
        config["gpu_model"] = self.gpu_model
        config["cuda_version"] = self.cuda_version
        if torch.cuda.is_available():
            config["cudnn_benchmark"] = True

        output_base_dir = "/kaggle/working/outputs"
        config["output_dir"] = output_base_dir

        # Auto-resume should only trigger if explicitly requested in config/overrides
        user_wants_resume = active_overrides.get("resume", config.get("resume", False))
        resume_checkpoint = ""
        if user_wants_resume:
            runs_dir = os.path.join(output_base_dir, "runs")
            if os.path.exists(runs_dir):
                all_runs = sorted(glob.glob(os.path.join(runs_dir, "Experiment_*")))
                if all_runs:
                    latest_run = all_runs[-1]
                    latest_pth = os.path.join(latest_run, "latest.pth")
                    if os.path.exists(latest_pth):
                        resume_checkpoint = latest_pth

            if resume_checkpoint:
                print(f"\033[34mℹ\033[0m Resume requested! Resuming from: \033[32m{resume_checkpoint}\033[0m")
                config["resume"] = True
                config["resume_checkpoint"] = resume_checkpoint
            else:
                print(f"\033[33m⚠\033[0m Resume requested but no previous checkpoint was found. Starting fresh run.")
                config["resume"] = False
                config["resume_checkpoint"] = ""
        else:
            config["resume"] = False
            config["resume_checkpoint"] = ""

        # Assert zero divergence between active_overrides and config
        for k, v in active_overrides.items():
            if config.get(k) != v:
                raise RuntimeError(f"Configuration mismatch error: Key '{k}' declared as '{v}' in notebook overrides but resolved to '{config.get(k)}' in final config!")

        # Write training configuration
        with open(self.resolved_config_path, "w") as f:
            json.dump(config, f, indent=4)

        print("\nResolved Runtime Configuration (train_config_resolved.json):")
        print(json.dumps(config, indent=4))

        # Create manifest metadata
        manifest = {
            "gpu": self.gpu_model,
            "cuda": self.cuda_version,
            "device": self.device,
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

        # Print Manifest Table
        BOLD = "\033[1m"
        RESET = "\033[0m"
        CYAN = "\033[36m"

        try:
            print(f"\n{CYAN}┌──────────────────────┬────────────────────────────────────────────────────────┐{RESET}")
            print(f"{CYAN}│{BOLD} Key                  {RESET}{CYAN}│{BOLD} Value                                                  {RESET}{CYAN}│{RESET}")
            print(f"{CYAN}├──────────────────────┼────────────────────────────────────────────────────────┤{RESET}")

            for key, val in manifest_items:
                val_str = str(val)
                if len(val_str) > 54:
                    val_str = val_str[:51] + "..."
                print(f"{CYAN}│{RESET} {key:<20} {CYAN}│{RESET} {val_str:<54} {CYAN}│{RESET}")

            print(f"{CYAN}└──────────────────────┴────────────────────────────────────────────────────────┘{RESET}\n")
        except UnicodeEncodeError:
            print(f"\nManifest: GPU={self.gpu_model}, Device={self.device}, PyTorch={self.pytorch_version}, Output={output_base_dir}\n")

        return config

    def launch_training(self):
        self._print_header("FrameSat-AI Orchestrator: Training Launch")

        train_script = os.path.join(self.bundle_path, "training", "train.py")
        print(f"\033[34mℹ\033[0m Executing pipeline script: \033[1;36m{train_script}\033[0m")
        print(f"\033[34mℹ\033[0m Target device: \033[1;36m{self.device}\033[0m")

        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device_count > 1:
            import random
            port = random.randint(29000, 29999)
            print(f"\033[32m✔\033[0m Multi-GPU detected ({device_count} GPUs). Launching via torch.distributed.run (DDP) on port {port}...")
            cmd = [
                sys.executable, "-m", "torch.distributed.run", 
                f"--nproc_per_node={device_count}", 
                f"--master_port={port}",
                train_script, 
                "--config", self.resolved_config_path
            ]
        else:
            cmd = [sys.executable, train_script, "--config", self.resolved_config_path]

        # Make the resolved device explicit to the subprocess environment too,
        # in case train.py falls back to env vars instead of the config file.
        env = os.environ.copy()
        env["USE_LIBUV"] = "0"
        env["PYTHONPATH"] = self.bundle_path + os.pathsep + env.get("PYTHONPATH", "")
        env["NCCL_SOCKET_IFNAME"] = "lo"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["PYTHONUNBUFFERED"] = "1"
        if self.device == "cpu":
            # Force-hide GPUs from the subprocess if we ever allow a CPU run,
            # so behavior is consistent with what was validated/logged.
            env.setdefault("CUDA_VISIBLE_DEVICES", "")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )

        # Stream logs character by character to allow tqdm progress bars to overwrite dynamically
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            sys.stdout.write(char)
            sys.stdout.flush()

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Training pipeline crashed with exit code: {process.returncode}")
        print("\033[32m✔\033[0m Training execution succeeded.")

    def evaluate(self):
        self._print_header("FrameSat-AI Orchestrator: Post-Training Evaluation")

        # Load config to find the returned checkpoint path
        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)
            
        best_checkpoint = config.get("returned_checkpoint", "")
        
        # Diagnostic logging
        print(f"\033[34mℹ\033[0m Training run directory: {os.path.dirname(best_checkpoint) if best_checkpoint else 'N/A'}")
        print(f"\033[34mℹ\033[0m Resolved best_checkpoint: {best_checkpoint}")
        print(f"\033[34mℹ\033[0m Exists: {os.path.exists(best_checkpoint)}")
        
        if not best_checkpoint or not os.path.exists(best_checkpoint):
            print(f"\033[33m⚠\033[0m No best checkpoint found at {best_checkpoint}. Skipping evaluation.")
            return

        print(f"\033[32m✔\033[0m Found best checkpoint for evaluation: \033[36m{best_checkpoint}\033[0m")

        # Generate eval config
        eval_config = {
            "experiment_name": "eval_kaggle_finetune",
            "dataset_path": self.cache_dir,
            "checkpoint_path": best_checkpoint,
            "events": config.get("events", 10),
            "save_predictions": config.get("save_predictions", True),
            "device": self.device
        }

        eval_config_path = "/kaggle/working/eval_config.json"
        with open(eval_config_path, "w") as f:
            json.dump(eval_config, f, indent=4)

        # Parse template end dates and training resolution/channel
        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)

        start_date = config.get("start_date", "2024-10-10T21:00:00")
        end_date = config.get("end_date", "2024-10-14T09:00:00")

        # Read training resolution so evaluation uses the same spatial scale the model was trained on.
        # Mismatched resolution causes a severe distribution shift and near-zero predictions.
        raw_train_resize = config.get("train_resize", [384, 384])
        eval_train_resize = tuple(raw_train_resize) if isinstance(raw_train_resize, list) else raw_train_resize
        eval_channel = config.get("channel", 13)
        print(f"\033[34mℹ\033[0m Evaluation will use train_resize={eval_train_resize}, channel={eval_channel} (matched to training config)")

        # Run the evaluator programmatically via inline script
        eval_script = f"""
import sys
import os
import json
import torch
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

device = '{self.device}'
if device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('Config requested cuda but no CUDA device is visible to this process.')

print(f'Loading RIFE Model onto device: {{device}}...')
model = RIFEInterpolator()
model.load_checkpoint('{best_checkpoint}')

# Move model to the resolved device explicitly rather than assuming
# RIFEInterpolator defaults to GPU internally.
if hasattr(model, 'to'):
    model.to(device)
elif hasattr(model, 'device'):
    model.device = device

# Sanity check: fail loudly instead of silently evaluating on CPU.
if device == 'cuda' and hasattr(model, 'parameters'):
    try:
        on_cuda = next(model.parameters()).is_cuda
        if not on_cuda:
            raise RuntimeError('Model parameters are not on CUDA after .to(device) call.')
        print('Confirmed: model parameters are on CUDA.')
    except StopIteration:
        pass

# Use the same channel and train_resize as training to avoid resolution mismatch
# (a model trained at 384x384 fed full-res ~3000x3000 images produces near-zero predictions)
raw_dataset = GOES19TripletDataset(
    start_date=datetime.fromisoformat('{start_date}'),
    end_date=datetime.fromisoformat('{end_date}'),
    cache_dir='{self.cache_dir}',
    channel={eval_channel},
    train_resize={eval_train_resize},
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

        env = os.environ.copy()
        env["PYTHONPATH"] = self.bundle_path + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.check_call([sys.executable, eval_run_path], env=env)
        print("\033[32m✔\033[0m Scientific evaluation complete.")

    def package_outputs(self):
        self._print_header("FrameSat-AI Orchestrator: Exporting Artifacts")

        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)
            
        best_checkpoint = config.get("returned_checkpoint", "")
        latest_run = os.path.dirname(best_checkpoint) if best_checkpoint else None

        # Derive experiment ID dynamically from the actual run directory
        experiment_id = "experiment_001"
        if latest_run:
            experiment_id = os.path.basename(latest_run).lower()

        export_root = f"/kaggle/working/{experiment_id}"
        os.makedirs(export_root, exist_ok=True)

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
            
        # Copy evaluation outputs
        if latest_run:
            experiment_id = os.path.basename(latest_run)
            eval_source_dir = os.path.join("artifacts", "evaluation", experiment_id)
            if os.path.exists(eval_source_dir):
                eval_export_dir = os.path.join(export_root, "evaluation", experiment_id)
                shutil.copytree(eval_source_dir, eval_export_dir, dirs_exist_ok=True)

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
        tar_path = f"/kaggle/working/{experiment_id}.tar.gz"
        print(f"\033[34mℹ\033[0m Compressing export directory into: \033[36m{tar_path}\033[0m")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_root, arcname=experiment_id)

        print(f"\033[32m✔\033[0m Pack complete. Tar file created: \033[32m{tar_path}\033[0m")

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
        BOLD = "\033[1m"
        RESET = "\033[0m"
        GREEN = "\033[32m"

        print(f"\n{GREEN}┌──────────────────────┬────────────────────────────────────────────────────────┐{RESET}")
        print(f"{GREEN}│{BOLD} Metric / Asset       {RESET}{GREEN}│{BOLD} Details                                                 {RESET}{GREEN}│{RESET}")
        print(f"{GREEN}├──────────────────────┼────────────────────────────────────────────────────────┤{RESET}")

        summary_items = [
            ("Epochs Completed", epochs),
            ("Best PSNR", f"{best_psnr} dB" if best_psnr != "N/A" else "N/A"),
            ("Best SSIM", best_ssim),
            ("Export Path", export_root),
            ("Best Checkpoint", os.path.join(checkpoints_dir, 'best.pth')),
            ("Evaluation Report", os.path.join(eval_dir, 'training_report.md'))
        ]

        for key, val in summary_items:
            val_str = str(val)
            if len(val_str) > 54:
                val_str = val_str[:51] + "..."
            print(f"{GREEN}│{RESET} {key:<20} {GREEN}│{RESET} {val_str:<54} {GREEN}│{RESET}")

        print(f"{GREEN}└──────────────────────┴────────────────────────────────────────────────────────┘{RESET}\n")

    def run(self, overrides=None):
        """
        Executes the entire end-to-end experiment pipeline:
        setup, discovery, validation, config resolution, training, evaluation, and packaging.
        """
        self.setup_environment()
        self.discover_resources()
        self.validate_environment()
        self.resolve_runtime_config(overrides=overrides)
        self.launch_training()
        self.evaluate()
        self.package_outputs()