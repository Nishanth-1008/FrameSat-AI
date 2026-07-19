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

class KaggleOrchestrator:
    def _print_header(self, title):
        CYAN = "\033[36m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        width = 65
        padding = (width - len(title) - 2) // 2
        p_left = " " * padding
        p_right = " " * (width - len(title) - 2 - padding)
        print(f"\n{CYAN}┌{'─' * (width - 2)}┐{RESET}")
        print(f"{CYAN}│{BOLD}{p_left}{title}{p_right}{CYAN}│{RESET}")
        print(f"{CYAN}└{'─' * (width - 2)}┘{RESET}")

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
            "flownet.pkl", "IFNet_HDv3.py", "RIFE_HDv3.py",
            "refine.py"
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

    def resolve_runtime_config(self):
        print("\n=================================================")
        print(" FrameSat-AI Orchestrator: Manifest & Config")
        print("=================================================")

        config_template_path = os.path.join(self.bundle_path, "training", "configs", "train_rife426.json")
        if not os.path.exists(config_template_path):
            config_template_path = os.path.join(self.bundle_path, "framesat_kaggle", "train_kaggle.json")

        with open(config_template_path, "r") as f:
            config = json.load(f)

        # Override fields
        config["dataset_path"] = self.cache_dir
        config["quarantine_dir"] = os.path.join(os.path.dirname(self.cache_dir), "quarantine")
        config["pretrained_weights"] = self.weights_dir

        # Explicitly propagate device choice to train.py — do not rely on
        # train.py doing its own independent torch.cuda.is_available() check.
        config["device"] = self.device
        config["gpu_model"] = self.gpu_model
        config["cuda_version"] = self.cuda_version
        if torch.cuda.is_available():
            config["cudnn_benchmark"] = True

        # Set absolute output folder
        output_base_dir = "/kaggle/working/outputs"
        config["output_dir"] = output_base_dir

        # Check auto-resume capabilities
        resume_checkpoint = None
        runs_dir = os.path.join(output_base_dir, "runs")
        if os.path.exists(runs_dir):
            all_runs = sorted(glob.glob(os.path.join(runs_dir, "Experiment_*")))
            if all_runs:
                latest_run = all_runs[-1]
                latest_pth = os.path.join(latest_run, "latest.pth")
                if os.path.exists(latest_pth):
                    resume_checkpoint = latest_pth

        if resume_checkpoint:
            print(f"\033[34mℹ\033[0m Auto-Resume detected! Resuming from: \033[32m{resume_checkpoint}\033[0m")
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

        print(f"\n{CYAN}┌──────────────────────┬────────────────────────────────────────────────────────┐{RESET}")
        print(f"{CYAN}│{BOLD} Key                  {RESET}{CYAN}│{BOLD} Value                                                  {RESET}{CYAN}│{RESET}")
        print(f"{CYAN}├──────────────────────┼────────────────────────────────────────────────────────┤{RESET}")

        manifest_items = [
            ("GPU", self.gpu_model),
            ("Device", self.device),
            ("PyTorch", self.pytorch_version),
            ("CUDA", self.cuda_version),
            ("Project Path", self.bundle_path),
            ("Dataset Path", self.cache_dir),
            ("Metadata Path", self.metadata_db),
            ("Pretrained Weights", self.weights_dir),
            ("Output Path", output_base_dir)
        ]

        for key, val in manifest_items:
            val_str = str(val)
            if len(val_str) > 54:
                val_str = val_str[:51] + "..."
            print(f"{CYAN}│{RESET} {key:<20} {CYAN}│{RESET} {val_str:<54} {CYAN}│{RESET}")

        print(f"{CYAN}└──────────────────────┴────────────────────────────────────────────────────────┘{RESET}\n")

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

        # Load config to find the exact runs directory
        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)
            
        output_base_dir = config.get("output_dir", "/kaggle/working/outputs")
        runs_dir = os.path.join(output_base_dir, "runs")

        # Diagnostic logging
        print(f"\033[34mℹ\033[0m Expected runs directory: {runs_dir}")
        print(f"\033[34mℹ\033[0m Exists: {os.path.exists(runs_dir)}")
        
        if not os.path.exists(runs_dir):
            print("\033[33m⚠\033[0m Runs directory does not exist.")
            if os.path.exists(output_base_dir):
                print(f"    Available sibling directories in {output_base_dir}: {os.listdir(output_base_dir)}")
            print("Skipping evaluation.")
            return

        all_runs = sorted(glob.glob(os.path.join(runs_dir, "Experiment_*")))
        if not all_runs:
            print(f"\033[33m⚠\033[0m No 'Experiment_*' folders found in {runs_dir}.")
            print(f"    Contents of {runs_dir}: {os.listdir(runs_dir)}")
            print("Skipping evaluation.")
            return

        latest_run = all_runs[-1]
        best_checkpoint = os.path.join(latest_run, "best.pth")
        if not os.path.exists(best_checkpoint):
            print(f"\033[33m⚠\033[0m No best checkpoint found at {best_checkpoint}. Skipping evaluation.")
            return

        print(f"\033[32m✔\033[0m Found best checkpoint for evaluation: \033[36m{best_checkpoint}\033[0m")

        # Generate eval config
        eval_config = {
            "experiment_name": "eval_kaggle_finetune",
            "dataset_path": self.cache_dir,
            "weights": best_checkpoint,
            "events": 10,
            "save_predictions": True,
            "device": self.device
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
model.load_weights('{latest_run}')

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
        print("\033[32m✔\033[0m Scientific evaluation complete.")

    def package_outputs(self):
        self._print_header("FrameSat-AI Orchestrator: Exporting Artifacts")

        export_root = "/kaggle/working/experiment_001"
        os.makedirs(export_root, exist_ok=True)

        with open(self.resolved_config_path, "r") as f:
            config = json.load(f)
            
        output_base_dir = config.get("output_dir", "/kaggle/working/outputs")
        runs_dir = os.path.join(output_base_dir, "runs")

        # Find runs
        all_runs = sorted(glob.glob(os.path.join(runs_dir, "Experiment_*")))

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
        print(f"\033[34mℹ\033[0m Compressing export directory into: \033[36m{tar_path}\033[0m")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_root, arcname="experiment_001")

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