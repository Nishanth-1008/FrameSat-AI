import os
import sys
import json
import argparse
from datetime import datetime

# Insert root folder to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from datasets.providers.goes19.pipeline import GOES19DatasetAcquisitionPipeline

def main():
    parser = argparse.ArgumentParser(description="GOES-19 Dataset Acquisition and QA Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--target-scenes", type=int, default=None, help="Target count of valid scenes to acquire")
    parser.add_argument("--sector", type=str, default=None, choices=["CONUS", "Full Disk", "Mesoscale 1", "Mesoscale 2"], help="S3 GOES-19 Sector")
    parser.add_argument("--channel", type=int, default=None, help="ABI Channel number (default: 13)")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent download threads")
    parser.add_argument("--start-date", type=str, default=None, help="Acquisition start date/hour (ISO 8601 format, e.g. 2024-10-10T21:00:00)")
    parser.add_argument("--no-verify", action="store_true", help="Disable validation check post-download")
    parser.add_argument("--no-qa", action="store_true", help="Disable automatic QA runner post-acquisition")
    
    args = parser.parse_args()
    
    config = {}
    # Load config file if provided
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            print(f"Error: Config file not found at {args.config}")
            sys.exit(1)
            
    # Command line args override configuration file parameters
    if args.target_scenes is not None:
        config["target_scene_count"] = args.target_scenes
    if args.sector is not None:
        config["sector"] = args.sector
    if args.channel is not None:
        config["channel"] = args.channel
    if args.workers is not None:
        config["workers"] = args.workers
    if args.start_date is not None:
        try:
            # Validate timestamp format
            datetime.fromisoformat(args.start_date)
            config["start_date"] = args.start_date
        except ValueError:
            print(f"Error: Start date must be in ISO 8601 format (e.g. 2024-10-10T21:00:00). Got: {args.start_date}")
            sys.exit(1)
            
    if args.no_verify:
        config["verify"] = False
    if args.no_qa:
        config["run_qa"] = False
        
    # Execute pipeline
    try:
        pipeline = GOES19DatasetAcquisitionPipeline(config)
        pipeline.run()
    except Exception as e:
        print(f"\nPipeline Execution Failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
