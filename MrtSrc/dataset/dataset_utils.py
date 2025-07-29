import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CMRDatasetUtils:
    """Utility functions for working with CMRxRecon dataset structure."""
    
    @staticmethod
    def create_data_registry(dataset_root, output_file="data_registry.json"):
        """Create a registry of all data files with their metadata.
        
        Args:
            dataset_root (str): Root directory of the CMRxRecon dataset
            output_file (str): Path to save the registry JSON file
            
        Returns:
            dict: Registry data structure
        """
        dataset_root = Path(dataset_root)
        registry = {
            "modalities": {},
            "centers": {},
            "vendors": {},
            "patients": {},
            "files": []
        }
        
        # MultiCoil directory
        multi_coil_dir = dataset_root / "MultiCoil"
        if not multi_coil_dir.exists():
            print(f"Error: MultiCoil directory not found at {multi_coil_dir}")
            return registry
        
        # Process all modalities
        for modality_dir in multi_coil_dir.iterdir():
            if not modality_dir.is_dir():
                continue
                
            modality = modality_dir.name
            registry["modalities"][modality] = {"count": 0, "centers": set()}
            
            # Training set
            training_dir = modality_dir / "TrainingSet"
            if not training_dir.exists():
                continue
                
            fullsample_dir = training_dir / "FullSample"
            if not fullsample_dir.exists():
                continue
                
            # Process centers
            for center_dir in fullsample_dir.iterdir():
                if not center_dir.is_dir():
                    continue
                    
                center = center_dir.name
                registry["modalities"][modality]["centers"].add(center)
                
                if center not in registry["centers"]:
                    registry["centers"][center] = {"count": 0, "vendors": set()}
                
                # Process vendors
                for vendor_dir in center_dir.iterdir():
                    if not vendor_dir.is_dir():
                        continue
                        
                    vendor = vendor_dir.name
                    registry["centers"][center]["vendors"].add(vendor)
                    
                    if vendor not in registry["vendors"]:
                        registry["vendors"][vendor] = {"count": 0, "patients": set()}
                    
                    # Process patients
                    for patient_dir in vendor_dir.iterdir():
                        if not patient_dir.is_dir():
                            continue
                            
                        patient = patient_dir.name
                        registry["vendors"][vendor]["patients"].add(patient)
                        
                        patient_key = f"{center}_{vendor}_{patient}"
                        if patient_key not in registry["patients"]:
                            registry["patients"][patient_key] = {
                                "center": center,
                                "vendor": vendor,
                                "patient": patient,
                                "files": {}
                            }
                        
                        # Process data files
                        for file_path in patient_dir.glob("*.mat"):
                            if "mask" in file_path.name.lower():
                                continue  # Skip mask files
                                
                            registry["modalities"][modality]["count"] += 1
                            registry["centers"][center]["count"] += 1
                            registry["vendors"][vendor]["count"] += 1
                            
                            file_info = {
                                "modality": modality,
                                "center": center,
                                "vendor": vendor,
                                "patient": patient,
                                "filename": file_path.name,
                                "path": str(file_path.relative_to(dataset_root))
                            }
                            
                            registry["files"].append(file_info)
                            registry["patients"][patient_key]["files"][file_path.name] = file_info
        
        # Convert sets to lists for JSON serialization
        for modality in registry["modalities"]:
            registry["modalities"][modality]["centers"] = list(registry["modalities"][modality]["centers"])
        
        for center in registry["centers"]:
            registry["centers"][center]["vendors"] = list(registry["centers"][center]["vendors"])
        
        for vendor in registry["vendors"]:
            registry["vendors"][vendor]["patients"] = list(registry["vendors"][vendor]["patients"])
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"Registry saved to {output_file}")
        return registry
    
    @staticmethod
    def create_batch_script(dataset_registry, output_dir="batches", batch_size=10):
        """Create batch processing scripts based on dataset registry.
        
        Args:
            dataset_registry (dict or str): Registry dict or path to registry JSON
            output_dir (str): Directory to save batch scripts
            batch_size (int): Number of files per batch
            
        Returns:
            list: Paths to created batch scripts
        """
        # Load registry if it's a file path
        if isinstance(dataset_registry, str):
            with open(dataset_registry, 'r') as f:
                registry = json.load(f)
        else:
            registry = dataset_registry
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Group files by modality
        modality_files = {}
        for file_info in registry["files"]:
            modality = file_info["modality"]
            if modality not in modality_files:
                modality_files[modality] = []
            modality_files[modality].append(file_info)
        
        batch_scripts = []
        
        # Create batch scripts for each modality
        for modality, files in modality_files.items():
            # Create batches
            num_batches = (len(files) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(files))
                batch_files = files[start_idx:end_idx]
                
                # Create batch script
                batch_script_path = os.path.join(output_dir, f"{modality}_batch_{batch_idx+1}.py")
                
                with open(batch_script_path, 'w') as f:
                    f.write("import os\n")
                    f.write("import sys\n")
                    f.write("from generalized_showcase import process_cmr_data\n\n")
                    
                    f.write(f"# Batch {batch_idx+1} for {modality}\n")
                    f.write("data_root = \"DATASET_ROOT\"  # Replace with actual dataset root\n\n")
                    
                    f.write("# Process all files in this batch\n")
                    for file_info in batch_files:
                        center = file_info["center"]
                        vendor = file_info["vendor"]
                        patient = file_info["patient"]
                        filename = file_info["filename"]
                        
                        f.write(f"process_cmr_data(\n")
                        f.write(f"    data_root=data_root,\n")
                        f.write(f"    modality=\"{modality}\",\n")
                        f.write(f"    center=\"{center}\",\n")
                        f.write(f"    vendor=\"{vendor}\",\n")
                        f.write(f"    patient=\"{patient}\",\n")
                        f.write(f"    data_file=\"{filename}\",\n")
                        f.write(f"    mask_type=\"ktRadial\",\n")
                        f.write(f"    mask_acc=8,\n")
                        f.write(f"    save_output=True\n")
                        f.write(f")\n\n")
                
                batch_scripts.append(batch_script_path)
        
        print(f"Created {len(batch_scripts)} batch scripts in {output_dir}")
        return batch_scripts
    
    @staticmethod
    def generate_dataset_visualization(dataset_registry, output_dir="visualizations"):
        """Generate visualizations of dataset structure.
        
        Args:
            dataset_registry (dict or str): Registry dict or path to registry JSON
            output_dir (str): Directory to save visualizations
        """
        # Load registry if it's a file path
        if isinstance(dataset_registry, str):
            with open(dataset_registry, 'r') as f:
                registry = json.load(f)
        else:
            registry = dataset_registry
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Files per modality
        modality_counts = {mod: info["count"] for mod, info in registry["modalities"].items()}
        plt.figure(figsize=(12, 6))
        bars = plt.bar(modality_counts.keys(), modality_counts.values())
        plt.xlabel('Modality')
        plt.ylabel('Number of Files')
        plt.title('Files per Modality')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'files_per_modality.png'), dpi=300)
        plt.close()
        
        # 2. Files per center
        center_counts = {center: info["count"] for center, info in registry["centers"].items()}
        plt.figure(figsize=(12, 6))
        bars = plt.bar(center_counts.keys(), center_counts.values())
        plt.xlabel('Center')
        plt.ylabel('Number of Files')
        plt.title('Files per Center')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'files_per_center.png'), dpi=300)
        plt.close()
        
        # 3. Files per vendor
        vendor_counts = {vendor: info["count"] for vendor, info in registry["vendors"].items()}
        plt.figure(figsize=(14, 6))
        bars = plt.bar(vendor_counts.keys(), vendor_counts.values())
        plt.xlabel('Vendor')
        plt.ylabel('Number of Files')
        plt.title('Files per Vendor')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'files_per_vendor.png'), dpi=300)
        plt.close()
        
        # 4. Patient distribution by vendor
        vendor_patient_counts = {vendor: len(info["patients"]) 
                               for vendor, info in registry["vendors"].items()}
        plt.figure(figsize=(14, 6))
        bars = plt.bar(vendor_patient_counts.keys(), vendor_patient_counts.values())
        plt.xlabel('Vendor')
        plt.ylabel('Number of Patients')
        plt.title('Patient Distribution by Vendor')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'patients_per_vendor.png'), dpi=300)
        plt.close()
        
        # 5. Heatmap of modality coverage by center
        modality_center_matrix = {}
        for mod, info in registry["modalities"].items():
            modality_center_matrix[mod] = {}
            for center in registry["centers"].keys():
                modality_center_matrix[mod][center] = 1 if center in info["centers"] else 0
        
        df_coverage = pd.DataFrame(modality_center_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_coverage.T, cmap="YlGnBu", cbar_kws={'label': 'Available'})
        plt.title('Modality Coverage by Center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'modality_center_coverage.png'), dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    @staticmethod
    def extract_matched_cases(dataset_registry, output_file="matched_cases.csv"):
        """Find patients with data across multiple modalities.
        
        Args:
            dataset_registry (dict or str): Registry dict or path to registry JSON
            output_file (str): Path to save the matched cases CSV
            
        Returns:
            pd.DataFrame: DataFrame of matched cases
        """
        # Load registry if it's a file path
        if isinstance(dataset_registry, str):
            with open(dataset_registry, 'r') as f:
                registry = json.load(f)
        else:
            registry = dataset_registry
        
        # Create a mapping of patients to their modalities
        patient_modalities = {}
        
        for file_info in registry["files"]:
            patient_key = f"{file_info['center']}_{file_info['vendor']}_{file_info['patient']}"
            modality = file_info["modality"]
            
            if patient_key not in patient_modalities:
                patient_modalities[patient_key] = {
                    "center": file_info["center"],
                    "vendor": file_info["vendor"],
                    "patient": file_info["patient"],
                    "modalities": set()
                }
            
            patient_modalities[patient_key]["modalities"].add(modality)
        
        # Find patients with multiple modalities
        all_modalities = set(registry["modalities"].keys())
        matched_cases = []
        
        for patient_key, info in patient_modalities.items():
            modality_count = len(info["modalities"])
            
            if modality_count > 1:
                matched_case = {
                    "center": info["center"],
                    "vendor": info["vendor"],
                    "patient": info["patient"],
                    "modality_count": modality_count,
                    "has_all_modalities": info["modalities"] == all_modalities
                }
                
                # Add flags for each modality
                for modality in all_modalities:
                    matched_case[f"has_{modality}"] = modality in info["modalities"]
                
                matched_cases.append(matched_case)
        
        # Create DataFrame and save to CSV
        df_matched = pd.DataFrame(matched_cases)
        if output_file:
            df_matched.to_csv(output_file, index=False)
            print(f"Matched cases saved to {output_file}")
        
        return df_matched


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CMRxRecon dataset utilities')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the CMRxRecon dataset')
    parser.add_argument('--create-registry', action='store_true', help='Create dataset registry')
    parser.add_argument('--create-batches', action='store_true', help='Create batch processing scripts')
    parser.add_argument('--create-visualizations', action='store_true', help='Create dataset visualizations')
    parser.add_argument('--find-matched-cases', action='store_true', help='Find patients with multiple modalities')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    if args.create_registry:
        registry_file = os.path.join(args.output_dir, 'data_registry.json')
        registry = CMRDatasetUtils.create_data_registry(args.root, registry_file)
        
        if args.create_batches:
            batch_dir = os.path.join(args.output_dir, 'batches')
            CMRDatasetUtils.create_batch_script(registry, batch_dir)
        
        if args.create_visualizations:
            viz_dir = os.path.join(args.output_dir, 'visualizations')
            CMRDatasetUtils.generate_dataset_visualization(registry, viz_dir)
        
        if args.find_matched_cases:
            matched_file = os.path.join(args.output_dir, 'matched_cases.csv')
            CMRDatasetUtils.extract_matched_cases(registry, matched_file)
    else:
        print("No action specified. Use --create-registry, --create-batches, --create-visualizations, or --find-matched-cases")