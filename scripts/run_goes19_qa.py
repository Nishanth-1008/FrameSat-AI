import os
import sys
from datetime import datetime, timedelta

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.providers.goes19.goes19_builder import GOES19TripletDataset
from evaluation.qa.data_qa.visualizer import TripletVisualizer
from evaluation.qa.data_qa.validator import TripletValidator
from evaluation.qa.data_qa.statistics import DatasetStatistics
from evaluation.qa.data_qa.report import QAReport

def main():
    print("Initializing GOES-19 QA Pipeline...")
    
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training', 'data', 'qa', 'reports'))
    
    visualizer = TripletVisualizer(output_dir=reports_dir)
    validator = TripletValidator(min_bt=180.0, max_bt=320.0, max_time_gap_minutes=120)
    stats = DatasetStatistics()
    report = QAReport(output_dir=reports_dir)
    
    # We will use the existing downloaded cache from test_goes19.py to avoid re-downloading
    # Set date range that we know is cached
    start_date = datetime(2024, 10, 10, 21, 0, 0)
    end_date = start_date + timedelta(hours=3)
    
    # Test dynamic training resize as well
    print("Loading GOES-19 Dataset (with 512x512 resize for testing)...")
    dataset = GOES19TripletDataset(
        start_date=start_date,
        end_date=end_date,
        cache_dir='datasets/goes19_test_cache',
        product='ABI-L1b-RadC',
        channel=13,
        split='train',
        split_ratio=1.0,
        train_resize=(512, 512)
    )
    
    stats.set_num_scenes(len(dataset.files))
    
    print(f"Running QA on {len(dataset)} triplets...")
    
    # Sample every 10th triplet to visualize, or at least the first one
    visualize_indices = set(range(0, len(dataset), max(1, len(dataset) // 5)))
    
    for i in range(len(dataset)):
        try:
            t0, t1, t2 = dataset[i]
            meta = dataset.get_metadata(i)
            timestamps = meta['timestamps']
            
            # Validation
            errs_t0 = validator.validate_tensor(t0)
            errs_t1 = validator.validate_tensor(t1)
            errs_t2 = validator.validate_tensor(t2)
            errs_trip = validator.validate_triplet(t0, t1, t2, timestamps)
            
            all_errs = errs_t0 + errs_t1 + errs_t2 + errs_trip
            for e in all_errs:
                report.add_error(i, e)
                
            if not all_errs:
                # Update stats only for valid triplets
                stats.add_triplet(t0, t1, t2, timestamps)
                
            # Visualization
            if i in visualize_indices:
                save_name = f"triplet_{i:04d}.png"
                img_path = visualizer.visualize(t0, t1, t2, timestamps, save_name)
                report.add_visualization(img_path)
                
        except Exception as e:
            report.add_error(i, f"Exception during processing: {str(e)}")
            
    print("Generating QA Report...")
    report.set_statistics(stats.get_summary())
    md_path, json_path = report.export()
    
    print(f"QA Pipeline complete! Reports saved to:\n  - {md_path}\n  - {json_path}")

if __name__ == "__main__":
    main()
