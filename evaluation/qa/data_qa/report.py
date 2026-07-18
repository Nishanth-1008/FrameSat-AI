import os
import json

class QAReport:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.errors = []
        self.visualizations = []
        self.stats = {}
        
    def add_error(self, triplet_idx, msg):
        self.errors.append({"triplet_index": triplet_idx, "error": msg})
        
    def add_visualization(self, image_path):
        # Store relative path for markdown
        rel_path = os.path.basename(image_path)
        self.visualizations.append(rel_path)
        
    def set_statistics(self, stats_dict):
        self.stats = stats_dict
        
    def export(self):
        json_path = os.path.join(self.output_dir, "qa_report.json")
        md_path = os.path.join(self.output_dir, "qa_report.md")
        
        report_data = {
            "statistics": self.stats,
            "errors": self.errors,
            "visualizations": self.visualizations
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        with open(md_path, 'w') as f:
            f.write("# GOES-19 Dataset QA Report\n\n")
            
            f.write("## 1. Statistics\n")
            f.write(f"- **Total Source Scenes:** {self.stats.get('num_scenes', 0)}\n")
            f.write(f"- **Total Valid Triplets Generated:** {self.stats.get('num_triplets', 0)}\n")
            f.write(f"- **Spatial Resolutions:** {', '.join(self.stats.get('resolutions', []))}\n")
            f.write(f"- **Mean Time Interval:** {self.stats.get('time_intervals_mean_min', 0):.2f} minutes (std: {self.stats.get('time_intervals_std_min', 0):.2f})\n")
            f.write(f"- **Mean BT (Normalized):** {self.stats.get('bt_mean_normalized', 0):.4f} (std: {self.stats.get('bt_std_normalized', 0):.4f})\n\n")
            
            f.write("## 2. Anomalies Detected\n")
            if not self.errors:
                f.write("> [!TIP]\n> No anomalies detected! Dataset is clean.\n\n")
            else:
                f.write("> [!WARNING]\n> Some anomalies were detected during QA.\n\n")
                for err in self.errors:
                    f.write(f"- **Triplet {err['triplet_index']}:** {err['error']}\n")
            f.write("\n")
            
            f.write("## 3. Representative Visualizations\n")
            if not self.visualizations:
                f.write("No visualizations available.\n")
            else:
                for img in self.visualizations:
                    f.write(f"![Triplet Visualization]({img})\n\n")
                    
        return md_path, json_path
