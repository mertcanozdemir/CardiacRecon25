import os
import json
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path


class CMRDatasetExplorer:
    """Tool for exploring and extracting structure from CMRxRecon dataset."""
    
    def __init__(self, dataset_root):
        """Initialize with the root path of the dataset.
        
        Args:
            dataset_root (str): Root directory of the CMRxRecon dataset
        """
        self.dataset_root = Path(dataset_root)
        self.structure = {
            "modalities": set(),
            "centers": set(),
            "vendors": set(),
            "patients": defaultdict(set),  # Vendor -> set of patients
            "patient_count": 0,
            "file_count": 0
        }
        self.data_files = []
        self.mask_files = []
        
    def explore(self, verbose=True):
        """Explore the dataset structure to extract all components.
        
        Args:
            verbose (bool): Whether to print progress information
        
        Returns:
            dict: Summary of dataset structure
        """
        if verbose:
            print(f"Exploring dataset at: {self.dataset_root}")
        
        # First level: MultiCoil
        multi_coil_dir = self.dataset_root / "MultiCoil"
        if not multi_coil_dir.exists():
            print(f"Error: MultiCoil directory not found at {multi_coil_dir}")
            return self.structure
            
        # Second level: Modalities
        for modality_dir in multi_coil_dir.iterdir():
            if not modality_dir.is_dir():
                continue
                
            modality = modality_dir.name
            self.structure["modalities"].add(modality)
            
            if verbose:
                print(f"Processing modality: {modality}")
            
            # Process Training Set
            training_dir = modality_dir / "TrainingSet"
            if not training_dir.exists():
                continue
                
            # Process FullSample and Mask directories
            self._process_fullsample_dir(training_dir / "FullSample", modality, verbose)
            self._process_mask_dir(training_dir / "Mask_TaskAll", modality, verbose)
                
        # Convert sets to sorted lists for better readability
        summary = {
            "modalities": sorted(list(self.structure["modalities"])),
            "centers": sorted(list(self.structure["centers"])),
            "vendors": sorted(list(self.structure["vendors"])),
            "patients_by_vendor": {vendor: sorted(list(patients)) 
                                 for vendor, patients in self.structure["patients"].items()},
            "total_patients": self.structure["patient_count"],
            "total_data_files": self.structure["file_count"],
            "total_mask_files": len(self.mask_files)
        }
        
        return summary
    
    def _process_fullsample_dir(self, fullsample_dir, modality, verbose):
        """Process the FullSample directory structure.
        
        Args:
            fullsample_dir (Path): Path to FullSample directory
            modality (str): Current modality being processed
            verbose (bool): Whether to print progress
        """
        if not fullsample_dir.exists():
            return
            
        # Third level: Centers
        for center_dir in fullsample_dir.iterdir():
            if not center_dir.is_dir():
                continue
                
            center = center_dir.name
            self.structure["centers"].add(center)
            
            # Fourth level: Vendors
            for vendor_dir in center_dir.iterdir():
                if not vendor_dir.is_dir():
                    continue
                    
                vendor = vendor_dir.name
                self.structure["vendors"].add(vendor)
                
                # Fifth level: Patients
                for patient_dir in vendor_dir.iterdir():
                    if not patient_dir.is_dir():
                        continue
                        
                    patient = patient_dir.name
                    self.structure["patients"][vendor].add(patient)
                    self.structure["patient_count"] += 1
                    
                    # Track all data files
                    for data_file in patient_dir.glob("*.mat"):
                        if data_file.is_file() and "mask" not in data_file.name.lower():
                            self.structure["file_count"] += 1
                            self.data_files.append({
                                "modality": modality,
                                "center": center,
                                "vendor": vendor,
                                "patient": patient,
                                "file": data_file.name,
                                "path": str(data_file.relative_to(self.dataset_root))
                            })
    
    def _process_mask_dir(self, mask_dir, modality, verbose):
        """Process the Mask_TaskAll directory structure.
        
        Args:
            mask_dir (Path): Path to Mask directory
            modality (str): Current modality being processed
            verbose (bool): Whether to print progress
        """
        if not mask_dir.exists():
            return
            
        # Structure is the same as FullSample
        for center_dir in mask_dir.iterdir():
            if not center_dir.is_dir():
                continue
                
            center = center_dir.name
            
            for vendor_dir in center_dir.iterdir():
                if not vendor_dir.is_dir():
                    continue
                    
                vendor = vendor_dir.name
                
                for patient_dir in vendor_dir.iterdir():
                    if not patient_dir.is_dir():
                        continue
                        
                    patient = patient_dir.name
                    
                    # Track all mask files
                    for mask_file in patient_dir.glob("*mask*.mat"):
                        if mask_file.is_file():
                            mask_type = self._get_mask_type(mask_file.name)
                            mask_acc = self._get_mask_acc(mask_file.name)
                            
                            self.mask_files.append({
                                "modality": modality,
                                "center": center,
                                "vendor": vendor,
                                "patient": patient,
                                "file": mask_file.name,
                                "mask_type": mask_type,
                                "mask_acc": mask_acc,
                                "path": str(mask_file.relative_to(self.dataset_root))
                            })
    
    def _get_mask_type(self, filename):
        """Extract mask type from filename.
        
        Args:
            filename (str): Mask filename
            
        Returns:
            str: Mask type (Uniform, ktGaussian, ktRadial, or None)
        """
        if "Uniform" in filename:
            return "Uniform"
        elif "ktGaussian" in filename:
            return "ktGaussian"
        elif "ktRadial" in filename:
            return "ktRadial"
        else:
            return None
    
    def _get_mask_acc(self, filename):
        """Extract acceleration factor from mask filename.
        
        Args:
            filename (str): Mask filename
            
        Returns:
            int or None: Acceleration factor
        """
        # Try to extract the number after mask_type
        for mask_type in ["Uniform", "ktGaussian", "ktRadial"]:
            if mask_type in filename:
                try:
                    # Extract number that follows the mask type
                    pos = filename.find(mask_type) + len(mask_type)
                    acc = ""
                    while pos < len(filename) and filename[pos].isdigit():
                        acc += filename[pos]
                        pos += 1
                    return int(acc) if acc else None
                except:
                    return None
        return None
    
    def save_summary(self, output_dir="."):
        """Save dataset summary to JSON file.
        
        Args:
            output_dir (str): Directory to save the summary files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save structure summary
        summary = self.explore(verbose=False)
        with open(output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save data files list
        if self.data_files:
            df_data = pd.DataFrame(self.data_files)
            df_data.to_csv(output_dir / "data_files.csv", index=False)
        
        # Save mask files list
        if self.mask_files:
            df_mask = pd.DataFrame(self.mask_files)
            df_mask.to_csv(output_dir / "mask_files.csv", index=False)
        
        print(f"Summary files saved to {output_dir}")
    
    def get_vendor_stats(self):
        """Get statistics about vendors and their patients.
        
        Returns:
            pd.DataFrame: DataFrame with vendor statistics
        """
        vendor_stats = []
        for vendor, patients in self.structure["patients"].items():
            vendor_stats.append({
                "vendor": vendor,
                "patient_count": len(patients),
                "field_strength": self._extract_field_strength(vendor),
                "manufacturer": self._extract_manufacturer(vendor)
            })
        
        return pd.DataFrame(vendor_stats).sort_values("patient_count", ascending=False)
    
    def _extract_field_strength(self, vendor_name):
        """Extract field strength from vendor name.
        
        Args:
            vendor_name (str): Vendor name (e.g., 'UIH_30T_umr780')
            
        Returns:
            str: Field strength or None
        """
        if "15T" in vendor_name:
            return "1.5T"
        elif "30T" in vendor_name:
            return "3.0T"
        elif "50T" in vendor_name:
            return "5.0T"
        else:
            return None
    
    def _extract_manufacturer(self, vendor_name):
        """Extract manufacturer from vendor name.
        
        Args:
            vendor_name (str): Vendor name (e.g., 'UIH_30T_umr780')
            
        Returns:
            str: Manufacturer name or None
        """
        if vendor_name.startswith("UIH"):
            return "United Imaging Healthcare"
        elif vendor_name.startswith("Siemens"):
            return "Siemens"
        elif vendor_name.startswith("GE"):
            return "GE Healthcare"
        elif vendor_name.startswith("Philips"):
            return "Philips"
        else:
            return None
    
    def get_data_paths(self, modality=None, center=None, vendor=None, patient=None):
        """Get paths to data files matching specified criteria.
        
        Args:
            modality (str, optional): Filter by modality
            center (str, optional): Filter by center
            vendor (str, optional): Filter by vendor
            patient (str, optional): Filter by patient
            
        Returns:
            list: List of matching file paths
        """
        if not self.data_files:
            self.explore(verbose=False)
            
        filtered_files = self.data_files
        
        if modality:
            filtered_files = [f for f in filtered_files if f["modality"] == modality]
        if center:
            filtered_files = [f for f in filtered_files if f["center"] == center]
        if vendor:
            filtered_files = [f for f in filtered_files if f["vendor"] == vendor]
        if patient:
            filtered_files = [f for f in filtered_files if f["patient"] == patient]
            
        return [os.path.join(self.dataset_root, f["path"]) for f in filtered_files]
    
    def get_mask_paths(self, modality=None, center=None, vendor=None, patient=None, 
                       mask_type=None, mask_acc=None):
        """Get paths to mask files matching specified criteria.
        
        Args:
            modality (str, optional): Filter by modality
            center (str, optional): Filter by center
            vendor (str, optional): Filter by vendor
            patient (str, optional): Filter by patient
            mask_type (str, optional): Filter by mask type
            mask_acc (int, optional): Filter by acceleration factor
            
        Returns:
            list: List of matching file paths
        """
        if not self.mask_files:
            self.explore(verbose=False)
            
        filtered_files = self.mask_files
        
        if modality:
            filtered_files = [f for f in filtered_files if f["modality"] == modality]
        if center:
            filtered_files = [f for f in filtered_files if f["center"] == center]
        if vendor:
            filtered_files = [f for f in filtered_files if f["vendor"] == vendor]
        if patient:
            filtered_files = [f for f in filtered_files if f["patient"] == patient]
        if mask_type:
            filtered_files = [f for f in filtered_files if f["mask_type"] == mask_type]
        if mask_acc:
            filtered_files = [f for f in filtered_files if f["mask_acc"] == mask_acc]
            
        return [os.path.join(self.dataset_root, f["path"]) for f in filtered_files]


def main():
    parser = argparse.ArgumentParser(description='Explore CMRxRecon dataset structure')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the CMRxRecon dataset')
    parser.add_argument('--output', type=str, default='.', help='Output directory for summary files')
    parser.add_argument('--save', action='store_true', help='Save summary to files')
    parser.add_argument('--vendor-stats', action='store_true', help='Print vendor statistics')
    
    args = parser.parse_args()
    
    explorer = CMRDatasetExplorer(args.root)
    summary = explorer.explore()
    
    print("\n===== Dataset Summary =====")
    print(f"Modalities: {', '.join(summary['modalities'])}")
    print(f"Centers: {', '.join(summary['centers'])}")
    print(f"Vendors: {', '.join(summary['vendors'])}")
    print(f"Total unique patients: {summary['total_patients']}")
    print(f"Total data files: {summary['total_data_files']}")
    print(f"Total mask files: {summary['total_mask_files']}")
    
    if args.vendor_stats:
        print("\n===== Vendor Statistics =====")
        vendor_stats = explorer.get_vendor_stats()
        print(vendor_stats)
    
    if args.save:
        explorer.save_summary(args.output)


if __name__ == "__main__":
    main()