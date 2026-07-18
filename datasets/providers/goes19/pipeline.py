import os
import json
import time
from datetime import datetime, timedelta

from datasets.providers.goes19.db import GOES19MetadataDB
from datasets.providers.goes19.validator import GOES19DatasetValidator
from datasets.providers.goes19.downloader import GOES19Downloader
from datasets.providers.goes19.stats_generator import GOES19StatsGenerator
from datasets.providers.goes19.qa_runner import GOES19QARunner

class GOES19DatasetAcquisitionPipeline:
    def __init__(self, config):
        self.config = self._apply_defaults(config)
        
        # Setup paths
        self.cache_dir = self.config["cache_dir"]
        self.reports_dir = self.config["reports_dir"]
        self.quarantine_dir = self.config["quarantine_dir"]
        self.db_path = self.config["db_path"]
        
        # Initialize modules
        self.db_manager = GOES19MetadataDB(db_path=self.db_path)
        self.validator = GOES19DatasetValidator(quarantine_dir=self.quarantine_dir)
        
        self.downloader = GOES19Downloader(
            cache_dir=self.cache_dir,
            sector=self.config["sector"],
            channel=self.config["channel"],
            workers=self.config["workers"],
            resume=self.config["resume"],
            overwrite=self.config.get("overwrite", False)
        )
        
        self.stats_generator = GOES19StatsGenerator(db_manager=self.db_manager)
        self.qa_runner = GOES19QARunner(db_manager=self.db_manager, reports_dir=self.reports_dir)

    def _apply_defaults(self, config):
        defaults = {
            "satellite": "GOES19",
            "sector": "CONUS",
            "channel": 13,
            "target_scene_count": 1000,
            "workers": 8,
            "resume": True,
            "verify": True,
            "run_qa": True,
            "cache_dir": "datasets/cache/goes19_cache",
            "quarantine_dir": "datasets/quarantine/goes19_quarantine",
            "reports_dir": "artifacts/reports/qa",
            "db_path": "datasets/metadata/metadata.db",
            "start_date": "2024-10-10T21:00:00"
        }
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
        return config

    def run(self):
        """Execute the full data acquisition, validation, statistics, and QA loop."""
        print("="*60)
        print("STARTING GOES-19 DATASET ACQUISITION PIPELINE")
        print("="*60)
        print(f"Config: {json.dumps(self.config, indent=2)}")
        
        start_time = time.time()
        
        # 1. Check existing valid scenes count
        current_valid_count = self.db_manager.get_valid_scene_count(
            satellite=self.config["satellite"],
            channel=self.config["channel"],
            sector=self.config["sector"]
        )
        print(f"Currently indexed valid scenes in DB: {current_valid_count}")
        
        target_count = self.config["target_scene_count"]
        
        # 2. Acquisition loop
        if current_valid_count >= target_count:
            print(f"Target count of {target_count} scenes is already satisfied!")
        else:
            print(f"Need to acquire {target_count - current_valid_count} more scenes...")
            
            # Start date parsing
            current_date = datetime.fromisoformat(self.config["start_date"])
            
            # Cap execution iterations to avoid infinite loop on bad dates
            max_hours_to_check = 1000
            checked_hours = 0
            
            while current_valid_count < target_count and checked_hours < max_hours_to_check:
                print(f"Checking S3 hour prefix: {current_date.strftime('%Y-%m-%d %H:00:00')}")
                
                # List available keys on S3 for this hour
                s3_files = self.downloader.list_files_for_hour(current_date)
                print(f"  Found {len(s3_files)} files matching filter on S3.")
                
                # Download keys
                if s3_files:
                    download_results = self.downloader.download_files_concurrently(s3_files)
                    
                    # Validate and register each download
                    for local_path, status, key in download_results:
                        if not local_path:
                            print(f"  Failed download for key: {key}")
                            continue
                            
                        # If resume/cached verify logic is enabled, we can trust existing valid DB records
                        scene_id = os.path.splitext(os.path.basename(local_path))[0]
                        existing_record = self.db_manager.get_scene_by_id(scene_id)
                        
                        if existing_record and status == "skipped" and not self.config.get("overwrite", False):
                            # Already in DB and validated, skip validation
                            continue
                            
                        # Perform validation
                        if self.config["verify"]:
                            is_valid, err_msg, meta = self.validator.validate_file(local_path)
                            if is_valid:
                                self.db_manager.insert_scene(
                                    scene_id=scene_id,
                                    timestamp=meta["timestamp"],
                                    satellite=self.config["satellite"],
                                    channel=self.config["channel"],
                                    sector=self.config["sector"],
                                    scan_mode=meta["scan_mode"],
                                    filepath=meta["filepath"],
                                    filesize=meta["filesize"],
                                    checksum=meta["checksum"],
                                    download_time=datetime.now().isoformat()
                                )
                            else:
                                # Move to quarantine
                                self.validator.quarantine_file(local_path, err_msg)
                                self.db_manager.remove_scene(scene_id)
                        else:
                            # Insert without validation
                            filesize = os.path.getsize(local_path)
                            self.db_manager.insert_scene(
                                scene_id=scene_id,
                                timestamp=current_date.isoformat(),
                                satellite=self.config["satellite"],
                                channel=self.config["channel"],
                                sector=self.config["sector"],
                                scan_mode="Unknown",
                                filepath=os.path.abspath(local_path),
                                filesize=filesize,
                                checksum="none",
                                download_time=datetime.now().isoformat()
                            )
                            
                    # Update local count
                    current_valid_count = self.db_manager.get_valid_scene_count(
                        satellite=self.config["satellite"],
                        channel=self.config["channel"],
                        sector=self.config["sector"]
                    )
                    print(f"Progress: {current_valid_count}/{target_count} valid scenes.")
                    
                # Increment date by 1 hour
                current_date += timedelta(hours=1)
                checked_hours += 1
                
                # Check target limit
                if current_valid_count >= target_count:
                    print(f"Target count of {target_count} valid scenes successfully met.")
                    break
                    
            if checked_hours >= max_hours_to_check:
                print(f"Warning: Reached max search limit of {max_hours_to_check} hours without reaching target count.")

        # 3. Generate Statistics
        stats_path = os.path.join(os.path.dirname(self.db_path), "dataset_statistics.json")
        print(f"Generating dataset statistics to: {stats_path}")
        stats = self.stats_generator.generate_statistics(stats_path)
        
        # 4. Trigger Automatic QA
        if self.config["run_qa"]:
            print("Triggering Automatic QA Pipeline...")
            self.qa_runner.run_qa_pipeline(sample_size=50)
            
        duration = time.time() - start_time
        print("="*60)
        print(f"PIPELINE COMPLETED SUCCESSFULLY IN {duration:.1f}s")
        print("="*60)
        
        # Print download stats summary
        dl_stats = self.downloader.get_statistics(duration)
        print(f"Files Downloaded: {dl_stats['total_files_downloaded']}")
        print(f"Files Skipped:    {dl_stats['total_files_skipped']}")
        print(f"Files Failed:     {dl_stats['total_files_failed']}")
        print(f"Avg Speed:        {dl_stats['average_speed_mbps']} MB/s")
        
        return stats
