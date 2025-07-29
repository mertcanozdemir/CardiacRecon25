import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import fastmri
from fastmri.data import transforms as T
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


class VendorAnalysis:
    """Analysis of vendor-specific patterns in k-space data."""
    
    def __init__(self, data_root, output_dir, registry_file=None):
        """
        Args:
            data_root (str): Root directory of the CMRxRecon dataset
            output_dir (str): Directory to save analysis results
            registry_file (str, optional): Path to pre-generated dataset registry
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize data structures
        self.vendor_data = {}
        self.vendor_stats = {}
        self.registry = None
        
        # Load registry if provided
        if registry_file:
            self.load_registry(registry_file)
    
    def load_registry(self, registry_file):
        """Load dataset registry.
        
        Args:
            registry_file (str): Path to registry JSON file
        """
        import json
        
        with open(registry_file, 'r') as f:
            self.registry = json.load(f)
    
    def create_vendor_mapping(self):
        """Create mapping of vendors and their files."""
        # This is a placeholder that would be populated from registry or by scanning
        # Example structure:
        self.vendor_mapping = {
            "Siemens_15T_Avanto": [],
            "Siemens_15T_Sola": [],
            "Siemens_30T_CIMA.X": [],
            "Siemens_30T_Prisma": [],
            "Siemens_30T_Vida": [],
            "UIH_15T_umr670": [],
            "UIH_30T_umr780": [],
            "UIH_30T_umr790": [],
            "UIH_30T_umr880": [],
            "GE_15T_voyager": []
        }
        
        if self.registry:
            # Use registry to populate mapping
            for file_info in self.registry["files"]:
                vendor = file_info["vendor"]
                if vendor not in self.vendor_mapping:
                    self.vendor_mapping[vendor] = []
                
                self.vendor_mapping[vendor].append({
                    "path": os.path.join(self.data_root, file_info["path"]),
                    "modality": file_info["modality"],
                    "center": file_info["center"],
                    "patient": file_info["patient"]
                })
        else:
            # Scan directory structure
            for modality_dir in (self.data_root / "MultiCoil").iterdir():
                if not modality_dir.is_dir():
                    continue
                
                modality = modality_dir.name
                
                # Training set
                training_dir = modality_dir / "TrainingSet" / "FullSample"
                if not training_dir.exists():
                    continue
                
                # Process centers
                for center_dir in training_dir.iterdir():
                    if not center_dir.is_dir():
                        continue
                    
                    center = center_dir.name
                    
                    # Process vendors
                    for vendor_dir in center_dir.iterdir():
                        if not vendor_dir.is_dir():
                            continue
                        
                        vendor = vendor_dir.name
                        if vendor not in self.vendor_mapping:
                            self.vendor_mapping[vendor] = []
                        
                        # Process patients
                        for patient_dir in vendor_dir.iterdir():
                            if not patient_dir.is_dir():
                                continue
                            
                            patient = patient_dir.name
                            
                            # Process data files
                            for file_path in patient_dir.glob("*.mat"):
                                if "mask" in file_path.name.lower():
                                    continue  # Skip mask files
                                
                                self.vendor_mapping[vendor].append({
                                    "path": str(file_path),
                                    "modality": modality,
                                    "center": center,
                                    "patient": patient
                                })
        
        # Print summary
        print("Vendor data summary:")
        for vendor, files in self.vendor_mapping.items():
            print(f"{vendor}: {len(files)} files")
    
    def load_kspace_sample(self, file_path, frame_idx=0, slice_idx=5):
        """Load a sample k-space from a file.
        
        Args:
            file_path (str): Path to k-space file
            frame_idx (int): Frame index to use
            slice_idx (int): Slice index to use
            
        Returns:
            numpy.ndarray: K-space data
        """
        try:
            with h5py.File(file_path, 'r') as hf:
                kspace = hf['kspace']
                kspace_data = kspace["real"] + 1j * kspace["imag"]
                
                # Handle different shapes
                if len(kspace_data.shape) == 5:  # (nframe, nslice, ncoil, ny, nx)
                    # Ensure indices are within bounds
                    frame_idx = min(frame_idx, kspace_data.shape[0] - 1)
                    slice_idx = min(slice_idx, kspace_data.shape[1] - 1)
                    
                    # Extract slice
                    kspace_slice = kspace_data[frame_idx, slice_idx]
                else:
                    # Handle other shapes
                    kspace_slice = kspace_data
                
                return kspace_slice
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def compute_kspace_features(self, kspace_data):
        """Compute features from k-space data.
        
        Args:
            kspace_data (numpy.ndarray): K-space data
            
        Returns:
            dict: Computed features
        """
        if kspace_data is None:
            return None
        
        # Compute magnitude
        kspace_mag = np.abs(kspace_data)
        
        # Compute features
        features = {
            "mean": np.mean(kspace_mag),
            "std": np.std(kspace_mag),
            "median": np.median(kspace_mag),
            "max": np.max(kspace_mag),
            "min": np.min(kspace_mag),
            "center_energy": np.mean(kspace_mag[:, kspace_mag.shape[1]//2-10:kspace_mag.shape[1]//2+10, 
                                              kspace_mag.shape[2]//2-10:kspace_mag.shape[2]//2+10]),
            "corner_energy": np.mean(kspace_mag[:, :10, :10]),
            "energy_ratio": np.sum(kspace_mag[:, kspace_mag.shape[1]//2-20:kspace_mag.shape[1]//2+20, 
                                           kspace_mag.shape[2]//2-20:kspace_mag.shape[2]//2+20]) / np.sum(kspace_mag),
            "spectral_distribution": np.mean(kspace_mag, axis=0)
        }
        
        return features
    
    def process_vendor_data(self, vendor, max_files=50):
        """Process data from a specific vendor.
        
        Args:
            vendor (str): Vendor name
            max_files (int): Maximum number of files to process
            
        Returns:
            dict: Processed data
        """
        if vendor not in self.vendor_mapping:
            print(f"Vendor {vendor} not found in mapping")
            return None
        
        files = self.vendor_mapping[vendor]
        if not files:
            print(f"No files found for vendor {vendor}")
            return None
        
        # Select a subset of files if needed
        if len(files) > max_files:
            import random
            files = random.sample(files, max_files)
        
        # Process files
        vendor_data = []
        for file_info in tqdm(files, desc=f"Processing {vendor}"):
            file_path = file_info["path"]
            
            # Load k-space data
            kspace_data = self.load_kspace_sample(file_path)
            if kspace_data is None:
                continue
            
            # Compute features
            features = self.compute_kspace_features(kspace_data)
            if features is None:
                continue
            
            # Add to vendor data
            vendor_data.append({
                "file_path": file_path,
                "modality": file_info["modality"],
                "features": features,
                "kspace_sample": kspace_data
            })
        
        return vendor_data
    
    def analyze_all_vendors(self, max_files_per_vendor=20):
        """Analyze data from all vendors.
        
        Args:
            max_files_per_vendor (int): Maximum number of files to process per vendor
        """
        print("Creating vendor mapping...")
        self.create_vendor_mapping()
        
        print("Processing vendor data...")
        for vendor in self.vendor_mapping.keys():
            self.vendor_data[vendor] = self.process_vendor_data(vendor, max_files_per_vendor)
        
        print("Computing vendor statistics...")
        self.compute_vendor_statistics()
        
        print("Generating visualizations...")
        self.generate_visualizations()
        
        print("Saving results...")
        self.save_results()
    
    def compute_vendor_statistics(self):
        """Compute statistics for each vendor."""
        for vendor, data in self.vendor_data.items():
            if not data:
                continue
            
            # Collect feature values
            features = {}
            for sample in data:
                for feature_name, feature_value in sample["features"].items():
                    if feature_name not in features:
                        features[feature_name] = []
                    
                    if feature_name != "spectral_distribution":
                        features[feature_name].append(feature_value)
            
            # Compute statistics
            stats = {}
            for feature_name, feature_values in features.items():
                if feature_name != "spectral_distribution":
                    stats[feature_name] = {
                        "mean": np.mean(feature_values),
                        "std": np.std(feature_values),
                        "median": np.median(feature_values),
                        "min": np.min(feature_values),
                        "max": np.max(feature_values)
                    }
            
            self.vendor_stats[vendor] = stats
    
    def generate_visualizations(self):
        """Generate visualizations of vendor-specific patterns."""
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Feature comparison across vendors
        self.visualize_feature_comparison(viz_dir)
        
        # 2. K-space intensity distribution
        self.visualize_intensity_distribution(viz_dir)
        
        # 3. K-space spectral distribution
        self.visualize_spectral_distribution(viz_dir)
        
        # 4. K-space center vs. periphery energy
        self.visualize_energy_distribution(viz_dir)
        
        # 5. PCA/t-SNE visualization of k-space features
        self.visualize_feature_embedding(viz_dir)
        
        # 6. K-space samples from different vendors
        self.visualize_kspace_samples(viz_dir)
    
    def visualize_feature_comparison(self, viz_dir):
        """Visualize feature comparison across vendors.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        # Collect feature data
        feature_data = []
        for vendor, stats in self.vendor_stats.items():
            for feature_name, feature_stats in stats.items():
                feature_data.append({
                    "vendor": vendor,
                    "feature": feature_name,
                    "value": feature_stats["mean"],
                    "std": feature_stats["std"]
                })
        
        # Create DataFrame
        df = pd.DataFrame(feature_data)
        
        # Create separate plots for each feature
        features = df["feature"].unique()
        for feature in features:
            if feature == "spectral_distribution":
                continue
                
            plt.figure(figsize=(12, 6))
            sns.barplot(x="vendor", y="value", data=df[df["feature"] == feature])
            plt.title(f"{feature} by Vendor")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(viz_dir / f"feature_comparison_{feature}.png", dpi=300)
            plt.close()
    
    def visualize_intensity_distribution(self, viz_dir):
        """Visualize k-space intensity distribution across vendors.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        plt.figure(figsize=(12, 8))
        
        for vendor, data in self.vendor_data.items():
            if not data:
                continue
            
            # Collect intensity values from all samples
            intensities = []
            for sample in data[:5]:  # Limit to 5 samples per vendor to avoid clutter
                kspace_sample = sample["kspace_sample"]
                if kspace_sample is None:
                    continue
                
                # Flatten and take log for better visualization
                intensity = np.log(np.abs(kspace_sample) + 1e-9).flatten()
                intensities.extend(intensity)
            
            # Plot histogram
            sns.kdeplot(intensities, label=vendor)
        
        plt.title("K-space Intensity Distribution by Vendor")
        plt.xlabel("Log Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "intensity_distribution.png", dpi=300)
        plt.close()
    
    def visualize_spectral_distribution(self, viz_dir):
        """Visualize k-space spectral distribution across vendors.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        plt.figure(figsize=(15, 10))
        
        for i, (vendor, data) in enumerate(self.vendor_data.items()):
            if not data or len(data) < 3:
                continue
            
            # Get a representative sample
            sample = data[0]
            kspace_sample = sample["kspace_sample"]
            if kspace_sample is None:
                continue
            
            # Average across coils
            kspace_mag = np.mean(np.abs(kspace_sample), axis=0)
            
            # Plot 2D spectral distribution
            plt.subplot(2, 3, i+1)
            plt.imshow(np.log(kspace_mag + 1e-9), cmap='viridis')
            plt.title(f"{vendor}")
            plt.colorbar(label="Log Magnitude")
        
        plt.suptitle("K-space Spectral Distribution by Vendor")
        plt.tight_layout()
        plt.savefig(viz_dir / "spectral_distribution.png", dpi=300)
        plt.close()
        
        # Also create line profiles
        plt.figure(figsize=(12, 8))
        
        for vendor, data in self.vendor_data.items():
            if not data or len(data) < 3:
                continue
            
            # Get a representative sample
            sample = data[0]
            kspace_sample = sample["kspace_sample"]
            if kspace_sample is None:
                continue
            
            # Average across coils
            kspace_mag = np.mean(np.abs(kspace_sample), axis=0)
            
            # Get central profiles
            profile_x = np.mean(kspace_mag, axis=0)
            profile_y = np.mean(kspace_mag, axis=1)
            
            # Plot profiles
            plt.subplot(2, 1, 1)
            plt.semilogy(profile_x, label=vendor)
            plt.title("X-axis Profile")
            plt.xlabel("Frequency")
            plt.ylabel("Log Magnitude")
            
            plt.subplot(2, 1, 2)
            plt.semilogy(profile_y, label=vendor)
            plt.title("Y-axis Profile")
            plt.xlabel("Frequency")
            plt.ylabel("Log Magnitude")
        
        plt.subplot(2, 1, 1)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.legend()
        
        plt.suptitle("K-space Frequency Profiles by Vendor")
        plt.tight_layout()
        plt.savefig(viz_dir / "frequency_profiles.png", dpi=300)
        plt.close()
    
    def visualize_energy_distribution(self, viz_dir):
        """Visualize k-space energy distribution across vendors.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        # Collect energy ratio data
        energy_data = []
        for vendor, data in self.vendor_data.items():
            if not data:
                continue
            
            for sample in data:
                energy_data.append({
                    "vendor": vendor,
                    "center_energy": sample["features"]["center_energy"],
                    "corner_energy": sample["features"]["corner_energy"],
                    "energy_ratio": sample["features"]["energy_ratio"]
                })
        
        # Create DataFrame
        df = pd.DataFrame(energy_data)
        
        # Plot energy ratio
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="vendor", y="energy_ratio", data=df)
        plt.title("K-space Energy Ratio by Vendor")
        plt.xlabel("Vendor")
        plt.ylabel("Energy Ratio (Center / Total)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(viz_dir / "energy_ratio.png", dpi=300)
        plt.close()
    
    def visualize_feature_embedding(self, viz_dir):
        """Visualize feature embedding using PCA or t-SNE.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        # Collect feature vectors
        feature_vectors = []
        vendor_labels = []
        modality_labels = []
        
        for vendor, data in self.vendor_data.items():
            if not data:
                continue
            
            for sample in data:
                # Create feature vector
                feature_vector = [
                    sample["features"]["mean"],
                    sample["features"]["std"],
                    sample["features"]["median"],
                    sample["features"]["max"],
                    sample["features"]["min"],
                    sample["features"]["center_energy"],
                    sample["features"]["corner_energy"],
                    sample["features"]["energy_ratio"]
                ]
                
                feature_vectors.append(feature_vector)
                vendor_labels.append(vendor)
                modality_labels.append(sample["modality"])
        
        if not feature_vectors:
            return
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame
        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "vendor": vendor_labels,
            "modality": modality_labels
        })
        
        # Plot PCA by vendor
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x="PC1", y="PC2", hue="vendor", data=df_pca, s=100, alpha=0.7)
        plt.title("PCA of K-space Features by Vendor")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(viz_dir / "pca_vendor.png", dpi=300)
        plt.close()
        
        # Plot PCA by modality
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x="PC1", y="PC2", hue="modality", data=df_pca, s=100, alpha=0.7)
        plt.title("PCA of K-space Features by Modality")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(viz_dir / "pca_modality.png", dpi=300)
        plt.close()
        
        # Apply t-SNE if we have enough samples
        if len(feature_vectors) >= 50:
            tsne = TSNE(n_components=2, perplexity=min(30, len(feature_vectors) // 5))
            X_tsne = tsne.fit_transform(X)
            
            # Create DataFrame
            df_tsne = pd.DataFrame({
                "t-SNE1": X_tsne[:, 0],
                "t-SNE2": X_tsne[:, 1],
                "vendor": vendor_labels,
                "modality": modality_labels
            })
            
            # Plot t-SNE by vendor
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x="t-SNE1", y="t-SNE2", hue="vendor", data=df_tsne, s=100, alpha=0.7)
            plt.title("t-SNE of K-space Features by Vendor")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(viz_dir / "tsne_vendor.png", dpi=300)
            plt.close()
            
            # Plot t-SNE by modality
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x="t-SNE1", y="t-SNE2", hue="modality", data=df_tsne, s=100, alpha=0.7)
            plt.title("t-SNE of K-space Features by Modality")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(viz_dir / "tsne_modality.png", dpi=300)
            plt.close()
    
    def visualize_kspace_samples(self, viz_dir):
        """Visualize k-space samples from different vendors.
        
        Args:
            viz_dir (Path): Directory to save visualizations
        """
        plt.figure(figsize=(15, len(self.vendor_data) * 3))
        
        for i, (vendor, data) in enumerate(self.vendor_data.items()):
            if not data or len(data) < 1:
                continue
            
            # Get a representative sample
            sample = data[0]
            kspace_sample = sample["kspace_sample"]
            if kspace_sample is None:
                continue
            
            # Get one coil for visualization
            kspace_coil = kspace_sample[0]
            
            # Convert to image
            kspace_tensor = T.to_tensor(kspace_coil)
            image = fastmri.ifft2c(kspace_tensor)
            image_abs = fastmri.complex_abs(image)
            
            # Plot k-space magnitude
            plt.subplot(len(self.vendor_data), 3, i*3+1)
            plt.imshow(np.log(np.abs(kspace_coil) + 1e-9), cmap='viridis')
            plt.title(f"{vendor} - K-space (Log)")
            plt.colorbar()
            
            # Plot k-space phase
            plt.subplot(len(self.vendor_data), 3, i*3+2)
            plt.imshow(np.angle(kspace_coil), cmap='hsv')
            plt.title(f"{vendor} - K-space (Phase)")
            plt.colorbar()
            
            # Plot image
            plt.subplot(len(self.vendor_data), 3, i*3+3)
            plt.imshow(image_abs.numpy(), cmap='gray')
            plt.title(f"{vendor} - Image")
            plt.colorbar()
        
        plt.suptitle("K-space and Image Samples by Vendor")
        plt.tight_layout()
        plt.savefig(viz_dir / "kspace_samples.png", dpi=300)
        plt.close()
    
    def save_results(self):
        """Save analysis results to files."""
        # Save vendor statistics
        stats_df = []
        for vendor, stats in self.vendor_stats.items():
            for feature_name, feature_stats in stats.items():
                if feature_name != "spectral_distribution":
                    stats_df.append({
                        "vendor": vendor,
                        "feature": feature_name,
                        "mean": feature_stats["mean"],
                        "std": feature_stats["std"],
                        "median": feature_stats["median"],
                        "min": feature_stats["min"],
                        "max": feature_stats["max"]
                    })
        
        df = pd.DataFrame(stats_df)
        df.to_csv(self.output_dir / "vendor_statistics.csv", index=False)
        
        # Save summary report
        with open(self.output_dir / "vendor_analysis_report.md", 'w') as f:
            f.write("# Vendor Analysis Report\n\n")
            
            f.write("## Vendor Data Summary\n\n")
            for vendor, data in self.vendor_data.items():
                f.write(f"- {vendor}: {len(data)} samples\n")
            f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("### Feature Comparison\n\n")
            
            # Create feature comparison table
            f.write("| Vendor | Mean Intensity | Energy Ratio | Center Energy |\n")
            f.write("|--------|---------------|--------------|---------------|\n")
            
            for vendor, stats in self.vendor_stats.items():
                f.write(f"| {vendor} | {stats['mean']['mean']:.6f} | {stats['energy_ratio']['mean']:.6f} | {stats['center_energy']['mean']:.6f} |\n")
            
            f.write("\n### Visualizations\n\n")
            f.write("See the visualizations directory for detailed plots of vendor-specific patterns.\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, consider the following for model development:\n\n")
            f.write("1. Vendor-specific normalization to account for intensity differences\n")
            f.write("2. Attention to k-space energy distribution differences between vendors\n")
            f.write("3. Model adaptation layers for cross-vendor generalization\n")
        
        print(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze vendor-specific patterns in k-space data')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of the CMRxRecon dataset')
    parser.add_argument('--output-dir', type=str, default='./vendor_analysis', help='Directory to save analysis results')
    parser.add_argument('--registry', type=str, help='Path to pre-generated dataset registry')
    parser.add_argument('--max-files', type=int, default=20, help='Maximum number of files to process per vendor')
    
    args = parser.parse_args()
    
    analyzer = VendorAnalysis(args.data_root, args.output_dir, args.registry)
    analyzer.analyze_all_vendors(args.max_files)


if __name__ == "__main__":
    main()