import numpy as np
import torch
import h5py
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import matplotlib.pyplot as plt
from tqdm import tqdm
import fastmri


class EvaluationMetrics:
    """Evaluation metrics for CMR reconstruction."""
    
    @staticmethod
    def calculate_ssim(gt, pred, crop_heart_region=True, heart_mask=None, data_range=None):
        """Calculate Structural Similarity Index Measure (SSIM).
        
        Args:
            gt (numpy.ndarray): Ground truth image
            pred (numpy.ndarray): Predicted image
            crop_heart_region (bool): Whether to crop the heart region for evaluation
            heart_mask (numpy.ndarray): Heart region mask
            data_range (float): Data range for normalization
            
        Returns:
            float: SSIM value
        """
        if data_range is None:
            data_range = np.max(gt) - np.min(gt)
        
        if crop_heart_region and heart_mask is not None:
            # Apply heart mask
            gt = gt * heart_mask
            pred = pred * heart_mask
        
        # Calculate SSIM
        ssim_value = ssim_func(
            gt, pred, 
            data_range=data_range,
            gaussian_weights=True, 
            sigma=1.5, 
            use_sample_covariance=False
        )
        
        return ssim_value
    
    @staticmethod
    def calculate_psnr(gt, pred, crop_heart_region=True, heart_mask=None, data_range=None):
        """Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            gt (numpy.ndarray): Ground truth image
            pred (numpy.ndarray): Predicted image
            crop_heart_region (bool): Whether to crop the heart region for evaluation
            heart_mask (numpy.ndarray): Heart region mask
            data_range (float): Data range for normalization
            
        Returns:
            float: PSNR value
        """
        if data_range is None:
            data_range = np.max(gt) - np.min(gt)
        
        if crop_heart_region and heart_mask is not None:
            # Apply heart mask
            gt = gt * heart_mask
            pred = pred * heart_mask
        
        # Calculate PSNR
        psnr_value = psnr_func(gt, pred, data_range=data_range)
        
        return psnr_value
    
    @staticmethod
    def calculate_nmse(gt, pred, crop_heart_region=True, heart_mask=None):
        """Calculate Normalized Mean Squared Error (NMSE).
        
        Args:
            gt (numpy.ndarray): Ground truth image
            pred (numpy.ndarray): Predicted image
            crop_heart_region (bool): Whether to crop the heart region for evaluation
            heart_mask (numpy.ndarray): Heart region mask
            
        Returns:
            float: NMSE value
        """
        if crop_heart_region and heart_mask is not None:
            # Apply heart mask
            gt = gt * heart_mask
            pred = pred * heart_mask
        
        # Calculate NMSE
        error = np.sum((gt - pred) ** 2)
        norm = np.sum(gt ** 2)
        nmse_value = error / norm
        
        return nmse_value
    
    @staticmethod
    def generate_heart_mask(image, method="otsu"):
        """Generate a binary mask for the heart region.
        
        Args:
            image (numpy.ndarray): Input image
            method (str): Method to generate mask ("otsu", "kmeans", or "manual_threshold")
            
        Returns:
            numpy.ndarray: Binary mask
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import binary_closing, binary_opening, disk
        
        # Normalize image
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Apply threshold
        if method == "otsu":
            thresh = threshold_otsu(norm_img)
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            pixels = norm_img.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
            labels = kmeans.labels_.reshape(norm_img.shape)
            
            # Determine which cluster represents the heart (higher intensity)
            cluster_means = [np.mean(norm_img[labels == i]) for i in range(2)]
            heart_cluster = np.argmax(cluster_means)
            mask = (labels == heart_cluster).astype(np.float32)
            return mask
        else:  # manual_threshold
            thresh = 0.3  # Adjust based on your data
        
        # Create binary mask
        mask = (norm_img > thresh).astype(np.float32)
        
        # Apply morphological operations to clean up the mask
        selem = disk(3)
        mask = binary_opening(mask, selem)
        mask = binary_closing(mask, selem)
        
        return mask


class EvaluationPipeline:
    """Pipeline for evaluating CMR reconstruction models."""
    
    def __init__(self, prediction_dir, ground_truth_dir, output_dir, heart_region_only=True):
        """
        Args:
            prediction_dir (str): Directory containing model predictions
            ground_truth_dir (str): Directory containing ground truth data
            output_dir (str): Directory to save evaluation results
            heart_region_only (bool): Whether to evaluate only the heart region
        """
        self.prediction_dir = Path(prediction_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.heart_region_only = heart_region_only
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics
        self.metrics = EvaluationMetrics()
        
        # Initialize results dictionary
        self.results = {
            "ssim": [],
            "psnr": [],
            "nmse": [],
            "filenames": []
        }
    
    def load_and_preprocess(self, pred_path, gt_path):
        """Load and preprocess prediction and ground truth data.
        
        Args:
            pred_path (Path): Path to prediction file
            gt_path (Path): Path to ground truth file
            
        Returns:
            tuple: (pred_img, gt_img, heart_mask)
        """
        # Load prediction
        with h5py.File(pred_path, 'r') as hf:
            # Assuming prediction is stored as 'reconstruction'
            key = 'reconstruction' if 'reconstruction' in hf else list(hf.keys())[0]
            pred_data = hf[key][()]
            
            # Handle complex data
            if 'real' in hf[key] and 'imag' in hf[key]:
                pred_data = hf[key]['real'][()] + 1j * hf[key]['imag'][()]
        
        # Load ground truth
        with h5py.File(gt_path, 'r') as hf:
            # Assuming ground truth is stored in k-space format
            if 'kspace' in hf:
                kspace = hf['kspace']['real'][()] + 1j * hf['kspace']['imag'][()]
                
                # Convert to image domain using fastmri
                kspace_tensor = torch.from_numpy(kspace)
                image_tensor = fastmri.ifft2c(kspace_tensor)
                gt_data = fastmri.complex_abs(image_tensor).numpy()
            else:
                # Try to find the appropriate key
                key = list(hf.keys())[0]
                gt_data = hf[key][()]
                
                # Handle complex data
                if 'real' in hf[key] and 'imag' in hf[key]:
                    gt_data = hf[key]['real'][()] + 1j * hf[key]['imag'][()]
        
        # Convert complex data to magnitude images if needed
        if np.iscomplexobj(pred_data):
            pred_img = np.abs(pred_data)
        else:
            pred_img = pred_data
            
        if np.iscomplexobj(gt_data):
            gt_img = np.abs(gt_data)
        else:
            gt_img = gt_data
        
        # Generate heart mask if needed
        heart_mask = None
        if self.heart_region_only:
            heart_mask = self.metrics.generate_heart_mask(gt_img)
        
        return pred_img, gt_img, heart_mask
    
    def evaluate_file(self, pred_path, gt_path):
        """Evaluate a single file.
        
        Args:
            pred_path (Path): Path to prediction file
            gt_path (Path): Path to ground truth file
            
        Returns:
            dict: Metrics for this file
        """
        # Load and preprocess data
        pred_img, gt_img, heart_mask = self.load_and_preprocess(pred_path, gt_path)
        
        # Calculate metrics
        ssim_value = self.metrics.calculate_ssim(gt_img, pred_img, self.heart_region_only, heart_mask)
        psnr_value = self.metrics.calculate_psnr(gt_img, pred_img, self.heart_region_only, heart_mask)
        nmse_value = self.metrics.calculate_nmse(gt_img, pred_img, self.heart_region_only, heart_mask)
        
        # Visualization for debugging
        self.visualize_comparison(pred_img, gt_img, heart_mask, pred_path.stem, ssim_value, psnr_value, nmse_value)
        
        return {
            "ssim": ssim_value,
            "psnr": psnr_value,
            "nmse": nmse_value,
            "filename": pred_path.name
        }
    
    def evaluate_all(self):
        """Evaluate all files in the prediction directory.
        
        Returns:
            dict: Aggregated metrics
        """
        # Find all prediction files
        pred_files = list(self.prediction_dir.glob("**/*.mat"))
        
        if not pred_files:
            print(f"No .mat files found in {self.prediction_dir}")
            return self.results
        
        print(f"Found {len(pred_files)} prediction files to evaluate")
        
        # Evaluate each file
        for pred_path in tqdm(pred_files):
            # Find corresponding ground truth file
            rel_path = pred_path.relative_to(self.prediction_dir)
            gt_path = self.ground_truth_dir / rel_path
            
            if not gt_path.exists():
                print(f"Warning: Ground truth file not found for {pred_path.name}")
                continue
            
            # Evaluate file
            metrics = self.evaluate_file(pred_path, gt_path)
            
            # Add to results
            self.results["ssim"].append(metrics["ssim"])
            self.results["psnr"].append(metrics["psnr"])
            self.results["nmse"].append(metrics["nmse"])
            self.results["filenames"].append(metrics["filename"])
        
        # Calculate average metrics
        avg_metrics = {
            "avg_ssim": np.mean(self.results["ssim"]),
            "avg_psnr": np.mean(self.results["psnr"]),
            "avg_nmse": np.mean(self.results["nmse"]),
            "median_ssim": np.median(self.results["ssim"]),
            "median_psnr": np.median(self.results["psnr"]),
            "median_nmse": np.median(self.results["nmse"])
        }
        
        # Save results
        self.save_results(avg_metrics)
        
        return avg_metrics
    
    def save_results(self, avg_metrics):
        """Save evaluation results.
        
        Args:
            avg_metrics (dict): Average metrics
        """
        # Save detailed results
        results_path = self.output_dir / "detailed_results.txt"
        with open(results_path, 'w') as f:
            f.write("Filename,SSIM,PSNR,NMSE\n")
            for i in range(len(self.results["filenames"])):
                f.write(f"{self.results['filenames'][i]},{self.results['ssim'][i]:.4f},{self.results['psnr'][i]:.2f},{self.results['nmse'][i]:.6f}\n")
            
            f.write("\nAverage Metrics:\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        # Create histograms
        self.plot_metric_histograms()
        
        print(f"Results saved to {self.output_dir}")
    
    def plot_metric_histograms(self):
        """Plot histograms of metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(self.results["ssim"], bins=20)
        axes[0].set_title(f"SSIM (Avg: {np.mean(self.results['ssim']):.4f})")
        axes[0].set_xlabel("SSIM")
        axes[0].set_ylabel("Count")
        
        axes[1].hist(self.results["psnr"], bins=20)
        axes[1].set_title(f"PSNR (Avg: {np.mean(self.results['psnr']):.2f})")
        axes[1].set_xlabel("PSNR (dB)")
        axes[1].set_ylabel("Count")
        
        axes[2].hist(self.results["nmse"], bins=20)
        axes[2].set_title(f"NMSE (Avg: {np.mean(self.results['nmse']):.6f})")
        axes[2].set_xlabel("NMSE")
        axes[2].set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metric_histograms.png")
        plt.close()
    
    def visualize_comparison(self, pred_img, gt_img, heart_mask, filename, ssim_value, psnr_value, nmse_value):
        """Visualize comparison between prediction and ground truth.
        
        Args:
            pred_img (numpy.ndarray): Predicted image
            gt_img (numpy.ndarray): Ground truth image
            heart_mask (numpy.ndarray): Heart region mask
            filename (str): File name
            ssim_value (float): SSIM value
            psnr_value (float): PSNR value
            nmse_value (float): NMSE value
        """
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ground truth
        im = axes[0, 0].imshow(gt_img, cmap='gray')
        axes[0, 0].set_title("Ground Truth")
        plt.colorbar(im, ax=axes[0, 0])
        
        # Prediction
        im = axes[0, 1].imshow(pred_img, cmap='gray')
        axes[0, 1].set_title("Prediction")
        plt.colorbar(im, ax=axes[0, 1])
        
        # Difference
        diff = np.abs(gt_img - pred_img)
        im = axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title(f"Difference\nSSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f}")
        plt.colorbar(im, ax=axes[0, 2])
        
        # Heart mask
        if heart_mask is not None:
            im = axes[1, 0].imshow(heart_mask, cmap='gray')
            axes[1, 0].set_title("Heart Mask")
            plt.colorbar(im, ax=axes[1, 0])
            
            # Ground truth with mask
            masked_gt = gt_img * heart_mask
            im = axes[1, 1].imshow(masked_gt, cmap='gray')
            axes[1, 1].set_title("Masked Ground Truth")
            plt.colorbar(im, ax=axes[1, 1])
            
            # Prediction with mask
            masked_pred = pred_img * heart_mask
            im = axes[1, 2].imshow(masked_pred, cmap='gray')
            axes[1, 2].set_title("Masked Prediction")
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"{filename}_comparison.png")
        plt.close()


class RadiologistScoring:
    """Tools for radiologist scoring interface."""
    
    def __init__(self, prediction_dir, ground_truth_dir, output_dir):
        """
        Args:
            prediction_dir (str): Directory containing model predictions
            ground_truth_dir (str): Directory containing ground truth data
            output_dir (str): Directory to save scoring results
        """
        self.prediction_dir = Path(prediction_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize scoring template
        self.scoring_template = {
            "image_quality": 0,  # 1-5 score
            "image_artifacts": 0,  # 1-5 score
            "clinical_utility": 0  # 1-5 score
        }
        
        # Initialize results
        self.scores = []
    
    def create_scoring_interface(self, sample_files=5):
        """Create scoring interface for radiologists.
        
        Args:
            sample_files (int): Number of sample files to include in the interface
            
        Returns:
            str: Path to scoring interface
        """
        # Find prediction files
        pred_files = list(self.prediction_dir.glob("**/*.mat"))
        
        if not pred_files:
            print(f"No .mat files found in {self.prediction_dir}")
            return None
        
        # Select sample files
        if len(pred_files) > sample_files:
            import random
            sample_preds = random.sample(pred_files, sample_files)
        else:
            sample_preds = pred_files
        
        # Create HTML interface
        html_path = self.output_dir / "scoring_interface.html"
        
        with open(html_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CMR Reconstruction Scoring</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .case { margin-bottom: 30px; border: 1px solid #ccc; padding: 20px; }
                    .images { display: flex; justify-content: space-between; margin-bottom: 20px; }
                    .image-container { text-align: center; }
                    .scoring { display: flex; justify-content: space-between; }
                    .score-category { width: 30%; }
                    h2 { color: #333; }
                    h3 { color: #555; }
                    input[type="radio"] { margin-right: 5px; }
                    label { margin-right: 15px; }
                    button { padding: 10px 20px; background-color: #4CAF50; color: white; 
                             border: none; cursor: pointer; margin-top: 20px; }
                    button:hover { background-color: #45a049; }
                </style>
            </head>
            <body>
                <h1>CMR Reconstruction Scoring</h1>
                <p>Please evaluate the following reconstructed images compared to the ground truth.</p>
                <form id="scoring-form">
            """)
            
            # Create cases
            for i, pred_path in enumerate(sample_preds):
                # Find corresponding ground truth file
                rel_path = pred_path.relative_to(self.prediction_dir)
                gt_path = self.ground_truth_dir / rel_path
                
                if not gt_path.exists():
                    print(f"Warning: Ground truth file not found for {pred_path.name}")
                    continue
                
                # Generate comparison image
                comparison_path = self.generate_comparison_image(pred_path, gt_path, i)
                
                # Add case to HTML
                f.write(f"""
                <div class="case">
                    <h2>Case {i+1}: {pred_path.stem}</h2>
                    <div class="images">
                        <div class="image-container">
                            <img src="{comparison_path.name}" width="100%" alt="Case {i+1}">
                        </div>
                    </div>
                    <div class="scoring">
                        <div class="score-category">
                            <h3>Image Quality (1-5)</h3>
                            <div>
                                <input type="radio" id="q{i}_1" name="quality_{i}" value="1">
                                <label for="q{i}_1">1 (Poor)</label>
                                <input type="radio" id="q{i}_2" name="quality_{i}" value="2">
                                <label for="q{i}_2">2</label>
                                <input type="radio" id="q{i}_3" name="quality_{i}" value="3">
                                <label for="q{i}_3">3</label>
                                <input type="radio" id="q{i}_4" name="quality_{i}" value="4">
                                <label for="q{i}_4">4</label>
                                <input type="radio" id="q{i}_5" name="quality_{i}" value="5">
                                <label for="q{i}_5">5 (Excellent)</label>
                            </div>
                        </div>
                        <div class="score-category">
                            <h3>Image Artifacts (1-5)</h3>
                            <div>
                                <input type="radio" id="a{i}_1" name="artifacts_{i}" value="1">
                                <label for="a{i}_1">1 (Many)</label>
                                <input type="radio" id="a{i}_2" name="artifacts_{i}" value="2">
                                <label for="a{i}_2">2</label>
                                <input type="radio" id="a{i}_3" name="artifacts_{i}" value="3">
                                <label for="a{i}_3">3</label>
                                <input type="radio" id="a{i}_4" name="artifacts_{i}" value="4">
                                <label for="a{i}_4">4</label>
                                <input type="radio" id="a{i}_5" name="artifacts_{i}" value="5">
                                <label for="a{i}_5">5 (None)</label>
                            </div>
                        </div>
                        <div class="score-category">
                            <h3>Clinical Utility (1-5)</h3>
                            <div>
                                <input type="radio" id="c{i}_1" name="clinical_{i}" value="1">
                                <label for="c{i}_1">1 (Not useful)</label>
                                <input type="radio" id="c{i}_2" name="clinical_{i}" value="2">
                                <label for="c{i}_2">2</label>
                                <input type="radio" id="c{i}_3" name="clinical_{i}" value="3">
                                <label for="c{i}_3">3</label>
                                <input type="radio" id="c{i}_4" name="clinical_{i}" value="4">
                                <label for="c{i}_4">4</label>
                                <input type="radio" id="c{i}_5" name="clinical_{i}" value="5">
                                <label for="c{i}_5">5 (Very useful)</label>
                            </div>
                        </div>
                    </div>
                </div>
                """)
            
            # Add submit button and close form
            f.write("""
                <button type="submit">Submit Scores</button>
                </form>
                
                <script>
                    document.getElementById('scoring-form').addEventListener('submit', function(e) {
                        e.preventDefault();
                        const formData = new FormData(this);
                        const scores = {};
                        
                        for (const [key, value] of formData.entries()) {
                            scores[key] = value;
                        }
                        
                        // In a real application, you would send this data to a server
                        alert('Scores submitted:\\n' + JSON.stringify(scores, null, 2));
                        
                        // For demonstration, save to local storage
                        localStorage.setItem('cmr_scores', JSON.stringify(scores));
                    });
                </script>
            </body>
            </html>
            """)
        
        print(f"Scoring interface created at {html_path}")
        return html_path
    
    def generate_comparison_image(self, pred_path, gt_path, case_idx):
        """Generate comparison image for scoring.
        
        Args:
            pred_path (Path): Path to prediction file
            gt_path (Path): Path to ground truth file
            case_idx (int): Case index
            
        Returns:
            Path: Path to comparison image
        """
        # Create images directory
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        
        # Load prediction and ground truth
        with h5py.File(pred_path, 'r') as hf:
            # Assuming prediction is stored as 'reconstruction'
            key = 'reconstruction' if 'reconstruction' in hf else list(hf.keys())[0]
            pred_data = hf[key][()]
            
            # Handle complex data
            if 'real' in hf[key] and 'imag' in hf[key]:
                pred_data = hf[key]['real'][()] + 1j * hf[key]['imag'][()]
        
        # Load ground truth
        with h5py.File(gt_path, 'r') as hf:
            # Assuming ground truth is stored in k-space format
            if 'kspace' in hf:
                kspace = hf['kspace']['real'][()] + 1j * hf['kspace']['imag'][()]
                
                # Convert to image domain using fastmri
                kspace_tensor = torch.from_numpy(kspace)
                image_tensor = fastmri.ifft2c(kspace_tensor)
                gt_data = fastmri.complex_abs(image_tensor).numpy()
            else:
                # Try to find the appropriate key
                key = list(hf.keys())[0]
                gt_data = hf[key][()]
                
                # Handle complex data
                if 'real' in hf[key] and 'imag' in hf[key]:
                    gt_data = hf[key]['real'][()] + 1j * hf[key]['imag'][()]
        
        # Convert complex data to magnitude images if needed
        if np.iscomplexobj(pred_data):
            pred_img = np.abs(pred_data)
        else:
            pred_img = pred_data
            
        if np.iscomplexobj(gt_data):
            gt_img = np.abs(gt_data)
        else:
            gt_img = gt_data
        
        # If multi-dimensional, take a slice
        if pred_img.ndim > 2:
            # For simplicity, just take the middle slice/frame
            if pred_img.ndim == 3:
                pred_img = pred_img[pred_img.shape[0]//2]
            elif pred_img.ndim == 4:
                pred_img = pred_img[pred_img.shape[0]//2, pred_img.shape[1]//2]
            elif pred_img.ndim == 5:
                pred_img = pred_img[pred_img.shape[0]//2, pred_img.shape[1]//2, pred_img.shape[2]//2]
        
        if gt_img.ndim > 2:
            if gt_img.ndim == 3:
                gt_img = gt_img[gt_img.shape[0]//2]
            elif gt_img.ndim == 4:
                gt_img = gt_img[gt_img.shape[0]//2, gt_img.shape[1]//2]
            elif gt_img.ndim == 5:
                gt_img = gt_img[gt_img.shape[0]//2, gt_img.shape[1]//2, gt_img.shape[2]//2]
        
        # Create comparison image
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        im = axes[0].imshow(gt_img, cmap='gray')
        axes[0].set_title("Ground Truth")
        plt.colorbar(im, ax=axes[0])
        
        # Prediction
        im = axes[1].imshow(pred_img, cmap='gray')
        axes[1].set_title("Reconstruction")
        plt.colorbar(im, ax=axes[1])
        
        # Difference
        diff = np.abs(gt_img - pred_img)
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title("Difference")
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        
        # Save image
        img_path = img_dir / f"comparison_{case_idx}.png"
        plt.savefig(img_path)
        plt.close()
        
        return img_path
    
    def parse_scores(self, scores_json):
        """Parse radiologist scores from JSON.
        
        Args:
            scores_json (str): JSON string of scores
            
        Returns:
            dict: Parsed scores
        """
        import json
        
        scores = json.loads(scores_json)
        parsed_scores = []
        
        # Number of cases is determined by the number of unique indices in the scores
        num_cases = len(set([int(k.split('_')[1]) for k in scores.keys()]))
        
        for i in range(num_cases):
            case_scores = {
                "case_idx": i,
                "image_quality": int(scores.get(f"quality_{i}", 0)),
                "image_artifacts": int(scores.get(f"artifacts_{i}", 0)),
                "clinical_utility": int(scores.get(f"clinical_{i}", 0)),
                "average_score": (int(scores.get(f"quality_{i}", 0)) + 
                                 int(scores.get(f"artifacts_{i}", 0)) + 
                                 int(scores.get(f"clinical_{i}", 0))) / 3
            }
            
            parsed_scores.append(case_scores)
        
        return parsed_scores
    
    def save_scores(self, parsed_scores):
        """Save radiologist scores.
        
        Args:
            parsed_scores (list): List of parsed scores
        """
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_scores)
        
        # Save to CSV
        csv_path = self.output_dir / "radiologist_scores.csv"
        df.to_csv(csv_path, index=False)
        
        # Calculate average scores
        avg_scores = {
            "avg_image_quality": df["image_quality"].mean(),
            "avg_image_artifacts": df["image_artifacts"].mean(),
            "avg_clinical_utility": df["clinical_utility"].mean(),
            "avg_total_score": df["average_score"].mean()
        }
        
        # Save summary
        summary_path = self.output_dir / "score_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Radiologist Scoring Summary\n")
            f.write("===========================\n\n")
            
            for k, v in avg_scores.items():
                f.write(f"{k}: {v:.2f}\n")
        
        print(f"Scores saved to {self.output_dir}")
        return avg_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CMR reconstruction')
    parser.add_argument('--pred-dir', type=str, required=True, help='Directory containing model predictions')
    parser.add_argument('--gt-dir', type=str, required=True, help='Directory containing ground truth data')
    parser.add_argument('--output-dir', type=str, default='./evaluation', help='Directory to save evaluation results')
    parser.add_argument('--heart-region-only', action='store_true', help='Evaluate only the heart region')
    parser.add_argument('--create-scoring-interface', action='store_true', help='Create radiologist scoring interface')
    
    args = parser.parse_args()
    
    # Evaluate model predictions
    eval_pipeline = EvaluationPipeline(
        args.pred_dir, args.gt_dir, args.output_dir, args.heart_region_only
    )
    
    metrics = eval_pipeline.evaluate_all()
    print("Evaluation complete!")
    print(f"Average SSIM: {metrics['avg_ssim']:.4f}")
    print(f"Average PSNR: {metrics['avg_psnr']:.2f} dB")
    print(f"Average NMSE: {metrics['avg_nmse']:.6f}")
    
    # Create radiologist scoring interface if requested
    if args.create_scoring_interface:
        rad_scoring = RadiologistScoring(
            args.pred_dir, args.gt_dir, os.path.join(args.output_dir, "radiologist_scoring")
        )
        
        interface_path = rad_scoring.create_scoring_interface()
        print(f"Radiologist scoring interface created at {interface_path}")