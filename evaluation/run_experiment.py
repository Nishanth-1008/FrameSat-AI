import os
import sys
import argparse

# Add root directory to path to allow running directly from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from evaluation.datasets import SEVIRDataset
from evaluation.models.rife import RIFEInterpolator
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Run SEVIR interpolation evaluation.")
    parser.add_argument("--model", type=str, default="rife", choices=["rife"], help="Model to evaluate")
    parser.add_argument("--weights", type=str, default="", help="Path to weights directory (if empty, uses default)")
    parser.add_argument("--modality", type=str, default="vis", choices=["vis", "vil", "ir107"], help="SEVIR modality")
    parser.add_argument("--events", type=int, default=5, help="Number of events to evaluate")
    parser.add_argument("--no-visuals", action="store_true", help="Disable saving visual comparisons")
    
    args = parser.parse_args()
    
    print(f"Initializing {args.model.upper()} evaluator on {args.modality.upper()} modality...")
    
    if args.model == "rife":
        model = RIFEInterpolator()
        # Default weights path if none provided: evaluation/models/rife_src/train_log
        weights_path = args.weights if args.weights else os.path.join(current_dir, "models", "rife_src", "train_log")
        model.load_weights(weights_path)
    else:
        raise ValueError(f"Model {args.model} is not supported yet.")
        
    dataset = SEVIRDataset(
        modality=args.modality,
        download_dir=os.path.join(current_dir, "datasets")
    )
    
    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        output_dir=os.path.join(current_dir, "outputs"),
        experiment_dir=os.path.join(current_dir, "experiments")
    )
    
    try:
        evaluator.run(num_events=args.events, save_visuals=not args.no_visuals)
    finally:
        dataset.close()

if __name__ == "__main__":
    main()
