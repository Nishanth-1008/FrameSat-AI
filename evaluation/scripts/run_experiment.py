import os
import sys
import argparse
import json

# Add root directory to path to allow running directly from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from evaluation.datasets import SatelliteDataset
from models.rife.interpolator import RIFEInterpolator
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Run satellite interpolation evaluation experiment.")
    parser.add_argument("--config", type=str, default="", help="Path to JSON configuration file")
    
    # CLI arguments fallback when --config is not specified
    parser.add_argument("--model", type=str, default="rife", choices=["rife"], help="Model to evaluate")
    parser.add_argument("--weights", type=str, default="", help="Path to weights directory")
    parser.add_argument("--modality", type=str, default="vis", choices=["vis", "vil", "ir107"], help="Satellite modality")
    parser.add_argument("--events", type=int, default=5, help="Number of events to evaluate")
    parser.add_argument("--experiment-name", type=str, default="cli_run", help="Experiment identifier")
    
    args = parser.parse_args()
    
    if args.config:
        print(f"Loading configuration file: {args.config}")
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        # Fallback to arguments
        config = {
            "experiment_name": args.experiment_name,
            "model": args.model,
            "weights": args.weights,
            "modality": args.modality,
            "events": args.events,
            "save_predictions": False
        }
        
    model_type = config.get("model", "rife").lower()
    modality = config.get("modality", "vis").lower()
    
    print(f"Initializing {model_type.upper()} model for experiment '{config.get('experiment_name')}'...")
    
    if model_type == "rife":
        model = RIFEInterpolator()
        # Default weights path if none provided: evaluation/models/rife_src/train_log
        w_path = config.get("weights")
        if w_path:
            if not os.path.isabs(w_path):
                weights_path = os.path.abspath(os.path.join(current_dir, "..", w_path))
            else:
                weights_path = w_path
        else:
            weights_path = os.path.join(current_dir, "models", "rife_src", "train_log")
        model.load_weights(weights_path)
    else:
        raise ValueError(f"Model {model_type} is not supported.")
        
    dataset = SatelliteDataset(
        modality=modality,
        download_dir=os.path.join(current_dir, "datasets")
    )
    
    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        config=config
    )
    
    try:
        evaluator.run()
    finally:
        dataset.close()

if __name__ == "__main__":
    main()
