#!/usr/bin/env python3
"""
CMRxRecon Dataset Verification Tool

This script scans a CMRxRecon dataset directory and verifies the integrity of all .mat files.
It identifies corrupted or problematic files and generates a report with detailed information.
It tries to load files with scipy.io first, then falls back to h5py if that fails.

Author: Claude AI
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import concurrent.futures
from datetime import datetime
import time
import logging
from tqdm import tqdm
import json

# Import scipy.io for mat file loading
import scipy.io
import h5py


class CMRVerifier:
    """Class to verify CMRxRecon dataset integrity and identify problematic files."""
    
    def __init__(self, data_root, output_dir, verification_level="basic", file_types=None):
        """Initialize the verifier.
        
        Args:
            data_root (str): Path to dataset root directory
            output_dir (str): Path to save verification results
            verification_level (str): Level of verification ("basic" or "thorough")
            file_types (list): List of file types to verify (default: [".mat"])
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.verification_level = verification_level
        self.file_types = file_types or [".mat"]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results
        self.valid_files = []
        self.invalid_files = []
        self.summary = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "error_types": {},
            "file_types": {},
            "directories": {},
            "loader_stats": {"scipy": 0, "h5py": 0, "failed": 0},
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None
        }
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.output_dir / "verification.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("CMRVerifier")
        self.logger.info(f"CMRVerifier initialized. Data root: {self.data_root}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Verification level: {self.verification_level}")
    
    def find_files(self):
        """Find all files in the dataset directory that match the specified file types.
        
        Returns:
            list: List of file paths
        """
        self.logger.info(f"Scanning directory for files: {self.data_root}")
        all_files = []
        
        for file_type in self.file_types:
            files = list(self.data_root.glob(f"**/*{file_type}"))
            all_files.extend(files)
            self.summary["file_types"][file_type] = len(files)
        
        self.logger.info(f"Found {len(all_files)} files")
        self.summary["total_files"] = len(all_files)
        
        # Count files by directory
        for file_path in all_files:
            # Get the relative path to the data root
            rel_path = file_path.relative_to(self.data_root)
            parent_dir = str(rel_path.parent)
            
            # Count by first level directory
            first_level = parent_dir.split(os.sep)[0] if os.sep in parent_dir else parent_dir
            if first_level not in self.summary["directories"]:
                self.summary["directories"][first_level] = 0
            self.summary["directories"][first_level] += 1
        
        return all_files
    
    def verify_file_basic(self, file_path):
        """Perform basic verification of a file using scipy.io first, then h5py.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            dict: Verification result
        """
        result = {
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(self.data_root)),
            "size": file_path.stat().st_size,
            "valid": False,
            "error": None,
            "verification_level": "basic",
            "file_type": file_path.suffix,
            "keys": [],
            "shape": None,
            "duration": 0,
            "loader": None  # Which loader was successful: 'scipy' or 'h5py'
        }
        
        start_time = time.time()
        
        # Skip if file size is zero
        if result["size"] == 0:
            result["error"] = "Zero size file"
            result["duration"] = time.time() - start_time
            return result
        
        # First try with scipy.io
        try:
            # Attempt to open the file with scipy.io
            mat_data = scipy.io.loadmat(file_path)
            
            # If we get here, scipy.io succeeded
            result["valid"] = True
            result["keys"] = list(mat_data.keys())
            # Filter out default keys that scipy adds
            result["keys"] = [k for k in result["keys"] if not k.startswith('__')]
            
            # Find main data key
            main_data_key = None
            for key in result["keys"]:
                if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 0:
                    main_data_key = key
                    result["shape"] = mat_data[key].shape
                    break
            
            # Try to determine data type
            if 'kspace' in result["keys"]:
                result["data_type"] = "kspace"
            elif 'mask' in result["keys"]:
                result["data_type"] = "mask"
            elif main_data_key:
                result["data_type"] = "other"
                result["main_key"] = main_data_key
            else:
                result["data_type"] = "unknown"
            
            result["loader"] = "scipy"
            self.summary["loader_stats"]["scipy"] += 1
            
        except Exception as scipy_error:
            # If scipy.io fails, try h5py
            try:
                with h5py.File(file_path, 'r') as hf:
                    # Get top-level keys
                    result["keys"] = list(hf.keys())
                    
                    # Check for expected keys
                    if 'kspace' in hf:
                        if 'real' in hf['kspace'] and 'imag' in hf['kspace']:
                            # Get shape
                            result["shape"] = hf['kspace']['real'].shape
                            result["data_type"] = "kspace"
                            result["valid"] = True
                        else:
                            result["error"] = "Missing real/imag in kspace"
                    elif 'mask' in hf:
                        # It's a mask file
                        result["shape"] = hf['mask'].shape
                        result["data_type"] = "mask"
                        result["valid"] = True
                    elif len(result["keys"]) > 0:
                        # Try the first key
                        first_key = result["keys"][0]
                        try:
                            result["shape"] = hf[first_key].shape
                            result["data_type"] = "other"
                            result["valid"] = True
                        except Exception as e:
                            result["error"] = f"Error accessing data in key {first_key}: {str(e)}"
                    else:
                        result["error"] = "No keys found in file"
                
                result["loader"] = "h5py"
                self.summary["loader_stats"]["h5py"] += 1
                    
            except Exception as h5py_error:
                # Both methods failed
                result["error"] = f"scipy error: {str(scipy_error)}, h5py error: {str(h5py_error)}"
                self.summary["loader_stats"]["failed"] += 1
        
        result["duration"] = time.time() - start_time
        return result
    
    def verify_file_thorough(self, file_path):
        """Perform thorough verification of a file, including data integrity.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            dict: Verification result
        """
        # Start with basic verification
        result = self.verify_file_basic(file_path)
        result["verification_level"] = "thorough"
        
        # If basic verification failed, don't proceed
        if not result["valid"]:
            return result
        
        start_time = time.time()
        
        try:
            # Perform thorough checks based on which loader worked
            if result["loader"] == "scipy":
                mat_data = scipy.io.loadmat(file_path)
                
                # Check data integrity based on data type
                if result["data_type"] == "kspace":
                    # Try to get kspace data
                    if 'kspace' in mat_data:
                        kspace_data = mat_data['kspace']
                        
                        # Check for NaN or Inf values
                        if np.isnan(kspace_data).any() or np.isinf(kspace_data).any():
                            result["valid"] = False
                            result["error"] = "Data contains NaN or Inf values"
                        
                        # Check if all values are zero
                        if np.all(kspace_data == 0):
                            result["valid"] = False
                            result["error"] = "Data contains all zeros"
                
                elif result["data_type"] == "mask":
                    # Check mask data
                    if 'mask' in mat_data:
                        mask_data = mat_data['mask']
                        
                        # Check if mask contains only 0 and 1
                        unique_values = np.unique(mask_data)
                        if not np.all(np.isin(unique_values, [0, 1])):
                            result["warning"] = f"Mask contains values other than 0 and 1: {unique_values}"
                
                elif "main_key" in result and result["main_key"]:
                    # Check main data for integrity
                    main_data = mat_data[result["main_key"]]
                    
                    # Check for NaN or Inf values
                    if np.isnan(main_data).any() or np.isinf(main_data).any():
                        result["warning"] = "Data contains NaN or Inf values"
                    
                    # Check if all values are zero
                    if np.all(main_data == 0):
                        result["warning"] = "Data contains all zeros"
            
            elif result["loader"] == "h5py":
                with h5py.File(file_path, 'r') as hf:
                    # Check data integrity based on data type
                    if result["data_type"] == "kspace":
                        # Try to read a small portion of the data
                        real_data = hf['kspace']['real']
                        imag_data = hf['kspace']['imag']
                        
                        # Check for NaN or Inf values
                        if result["shape"]:
                            # For large datasets, only check a subset
                            if len(result["shape"]) > 2 and result["shape"][0] > 0 and result["shape"][1] > 0:
                                # Read first frame, first slice
                                if len(result["shape"]) == 5:  # (nframe, nslice, ncoil, ny, nx)
                                    sample_real = real_data[0, 0]
                                    sample_imag = imag_data[0, 0]
                                elif len(result["shape"]) == 4:  # (nframe, nslice, ny, nx) or similar
                                    sample_real = real_data[0, 0]
                                    sample_imag = imag_data[0, 0]
                                elif len(result["shape"]) == 3:  # (ncoil, ny, nx) or similar
                                    sample_real = real_data[0]
                                    sample_imag = imag_data[0]
                                else:
                                    sample_real = real_data[:]
                                    sample_imag = imag_data[:]
                                
                                # Check for NaN or Inf
                                if np.isnan(sample_real).any() or np.isinf(sample_real).any() or \
                                   np.isnan(sample_imag).any() or np.isinf(sample_imag).any():
                                    result["valid"] = False
                                    result["error"] = "Data contains NaN or Inf values"
                                
                                # Check if all values are zero
                                if np.all(sample_real == 0) and np.all(sample_imag == 0):
                                    result["valid"] = False
                                    result["error"] = "Data contains all zeros"
                        
                    elif result["data_type"] == "mask":
                        # Read mask data
                        mask_data = hf['mask']
                        
                        # Check if mask contains only 0 and 1
                        if result["shape"]:
                            # For large masks, only check a subset
                            if len(result["shape"]) > 1 and result["shape"][0] > 0:
                                # Read first frame
                                if len(result["shape"]) == 3:  # (nframe, ny, nx)
                                    sample_mask = mask_data[0]
                                else:
                                    sample_mask = mask_data[:]
                                
                                # Check if values are only 0 or 1
                                unique_values = np.unique(sample_mask)
                                if not np.all(np.isin(unique_values, [0, 1])):
                                    result["warning"] = f"Mask contains values other than 0 and 1: {unique_values}"
                
        except Exception as e:
            result["valid"] = False
            result["error"] = f"Thorough verification failed: {str(e)}"
        
        result["duration"] += time.time() - start_time
        return result
    
    def verify_file(self, file_path):
        """Verify a file based on the selected verification level.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            dict: Verification result
        """
        if self.verification_level == "thorough":
            return self.verify_file_thorough(file_path)
        else:
            return self.verify_file_basic(file_path)
    
    def run_verification(self, max_workers=None):
        """Run verification on all files.
        
        Args:
            max_workers (int, optional): Maximum number of worker processes
            
        Returns:
            dict: Verification summary
        """
        self.logger.info(f"Starting verification with level: {self.verification_level}")
        start_time = time.time()
        
        # Find all files
        all_files = self.find_files()
        
        # Verify files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.verify_file, all_files),
                total=len(all_files),
                desc="Verifying files"
            ))
        
        # Process results
        for result in results:
            if result["valid"]:
                self.valid_files.append(result)
            else:
                self.invalid_files.append(result)
                
                # Count error types
                error = result["error"] if result["error"] else "Unknown error"
                if error not in self.summary["error_types"]:
                    self.summary["error_types"][error] = 0
                self.summary["error_types"][error] += 1
        
        # Update summary
        self.summary["valid_files"] = len(self.valid_files)
        self.summary["invalid_files"] = len(self.invalid_files)
        self.summary["end_time"] = datetime.now()
        self.summary["duration"] = time.time() - start_time
        
        self.logger.info(f"Verification complete. Valid: {len(self.valid_files)}, Invalid: {len(self.invalid_files)}")
        self.logger.info(f"Total verification time: {self.summary['duration']:.2f} seconds")
        self.logger.info(f"Loader stats: scipy: {self.summary['loader_stats']['scipy']}, "
                        f"h5py: {self.summary['loader_stats']['h5py']}, "
                        f"failed: {self.summary['loader_stats']['failed']}")
        
        # Save results
        self.save_results()
        
        return self.summary
    
    def save_results(self):
        """Save verification results to files."""
        self.logger.info("Saving verification results...")
        
        # Save list of problematic files
        with open(self.output_dir / "problematic_files.txt", 'w') as f:
            for result in self.invalid_files:
                f.write(f"{result['path']}\n")
        
        # Save detailed results as CSV
        valid_df = pd.DataFrame(self.valid_files)
        valid_df["status"] = "valid"
        
        invalid_df = pd.DataFrame(self.invalid_files)
        invalid_df["status"] = "invalid"
        
        # Combine dataframes
        all_df = pd.concat([valid_df, invalid_df], ignore_index=True)
        all_df.to_csv(self.output_dir / "verification_results.csv", index=False)
        
        # Save summary as JSON
        with open(self.output_dir / "verification_summary.json", 'w') as f:
            # Convert datetime objects to strings
            summary_copy = self.summary.copy()
            summary_copy["start_time"] = str(summary_copy["start_time"])
            summary_copy["end_time"] = str(summary_copy["end_time"])
            json.dump(summary_copy, f, indent=2)
        
        # Generate summary visualizations
        self.create_visualizations()
        
        # Generate HTML report
        self.create_html_report()
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def create_visualizations(self):
        """Create visualizations of verification results."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Pie chart of valid vs invalid files
        plt.figure(figsize=(10, 6))
        plt.pie(
            [self.summary["valid_files"], self.summary["invalid_files"]],
            labels=["Valid", "Invalid"],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336']
        )
        plt.title('File Validation Results')
        plt.savefig(viz_dir / "validation_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pie chart of loader usage
        plt.figure(figsize=(10, 6))
        plt.pie(
            [self.summary["loader_stats"]["scipy"], 
             self.summary["loader_stats"]["h5py"],
             self.summary["loader_stats"]["failed"]],
            labels=["scipy.io", "h5py", "Failed"],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#2196F3', '#F44336']
        )
        plt.title('Loader Usage')
        plt.savefig(viz_dir / "loader_usage_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Bar chart of error types
        if self.summary["error_types"]:
            error_df = pd.DataFrame({
                'Error': list(self.summary["error_types"].keys()),
                'Count': list(self.summary["error_types"].values())
            })
            error_df = error_df.sort_values('Count', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Count', y='Error', data=error_df)
            plt.title('Error Types')
            plt.tight_layout()
            plt.savefig(viz_dir / "error_types.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Bar chart of file counts by directory
        if self.summary["directories"]:
            dir_df = pd.DataFrame({
                'Directory': list(self.summary["directories"].keys()),
                'Count': list(self.summary["directories"].values())
            })
            dir_df = dir_df.sort_values('Count', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Count', y='Directory', data=dir_df)
            plt.title('File Counts by Directory')
            plt.tight_layout()
            plt.savefig(viz_dir / "directory_counts.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Bar chart of file counts by file type
        if self.summary["file_types"]:
            type_df = pd.DataFrame({
                'File Type': list(self.summary["file_types"].keys()),
                'Count': list(self.summary["file_types"].values())
            })
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='File Type', y='Count', data=type_df)
            plt.title('File Counts by Type')
            plt.tight_layout()
            plt.savefig(viz_dir / "file_type_counts.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_html_report(self):
        """Create HTML report of verification results."""
        report_path = self.output_dir / "verification_report.html"
        
        with open(report_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CMRxRecon Dataset Verification Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
                    h1, h2, h3 { color: #2c3e50; }
                    .container { display: flex; flex-wrap: wrap; }
                    .section { margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    .full-width { width: 100%; }
                    .half-width { width: calc(50% - 50px); }
                    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .valid { color: #4CAF50; }
                    .invalid { color: #F44336; }
                    .summary { display: flex; justify-content: space-around; margin: 20px 0; }
                    .summary-item { text-align: center; padding: 10px; }
                    .summary-value { font-size: 24px; font-weight: bold; }
                    img { max-width: 100%; height: auto; border-radius: 5px; margin-top: 10px; }
                    code { background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
                    .helper { background-color: #e9f7fe; padding: 15px; border-left: 5px solid #4CAF50; margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>CMRxRecon Dataset Verification Report</h1>
                <p>Report generated on: """ + str(self.summary["end_time"]) + """</p>
                
                <div class="summary">
                    <div class="summary-item">
                        <div class="summary-value">""" + str(self.summary["total_files"]) + """</div>
                        <div>Total Files</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value valid">""" + str(self.summary["valid_files"]) + """</div>
                        <div>Valid Files</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value invalid">""" + str(self.summary["invalid_files"]) + """</div>
                        <div>Invalid Files</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + f"{self.summary['duration']:.1f}" + """s</div>
                        <div>Duration</div>
                    </div>
                </div>
                
                <div class="container">
                    <div class="section full-width">
                        <h2>Verification Overview</h2>
                        <p>Dataset root: <code>""" + str(self.data_root) + """</code></p>
                        <p>Verification level: <code>""" + self.verification_level + """</code></p>
                        <p>File types checked: <code>""" + ", ".join(self.file_types) + """</code></p>
                        
                        <h3>Loader Statistics</h3>
                        <p>Files loaded with scipy.io: <b>""" + str(self.summary["loader_stats"]["scipy"]) + """</b></p>
                        <p>Files loaded with h5py: <b>""" + str(self.summary["loader_stats"]["h5py"]) + """</b></p>
                        <p>Files failed to load: <b>""" + str(self.summary["loader_stats"]["failed"]) + """</b></p>
                        
                        <h3>Visualizations</h3>
                        <div class="container">
                            <div class="half-width">
                                <img src="visualizations/validation_pie.png" alt="Validation Results">
                            </div>
                            <div class="half-width">
                                <img src="visualizations/loader_usage_pie.png" alt="Loader Usage">
                            </div>
            """)
            
            # Add error types visualization if available
            if os.path.exists(self.output_dir / "visualizations" / "error_types.png"):
                f.write("""
                            <div class="half-width">
                                <img src="visualizations/error_types.png" alt="Error Types">
                            </div>
                """)
            
            # Add directory counts visualization if available
            if os.path.exists(self.output_dir / "visualizations" / "directory_counts.png"):
                f.write("""
                            <div class="half-width">
                                <img src="visualizations/directory_counts.png" alt="Directory Counts">
                            </div>
                """)
            
            # Add file type counts visualization if available
            if os.path.exists(self.output_dir / "visualizations" / "file_type_counts.png"):
                f.write("""
                            <div class="half-width">
                                <img src="visualizations/file_type_counts.png" alt="File Type Counts">
                            </div>
                """)
            
            f.write("""
                        </div>
                    </div>
                    
                    <div class="section full-width">
                        <h2>Common Error Types</h2>
                        <table>
                            <tr>
                                <th>Error</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """)
            
            # Add error types
            for error, count in sorted(self.summary["error_types"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.summary["invalid_files"]) * 100 if self.summary["invalid_files"] > 0 else 0
                f.write(f"""
                            <tr>
                                <td>{error}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """)
            
            f.write("""
                        </table>
                    </div>
                    
                    <div class="section full-width">
                        <h2>How to Handle Invalid Files</h2>
                        <p>A list of problematic files has been saved to <code>problematic_files.txt</code>. You can use this list to exclude these files from your analysis.</p>
                        
                        <div class="helper">
                            <h3>Helper Code for Skipping Invalid Files</h3>
                            <p>Here's a simple function to check if a file is in the problematic files list:</p>
                            <pre><code>def is_valid_file(file_path, problematic_files_path):
    with open(problematic_files_path, 'r') as f:
        problematic_files = [line.strip() for line in f]
    
    return file_path not in problematic_files

# Example usage:
if is_valid_file(file_path, 'problematic_files.txt'):
    # Process the file
    pass
else:
    # Skip the file
    print(f"Skipping problematic file: {file_path}")
</code></pre>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """)
        
        self.logger.info(f"HTML report saved to {report_path}")
    
    def generate_helper_code(self):
        """Generate helper code for handling problematic files."""
        helper_path = self.output_dir / "file_handling_helper.py"
        
        with open(helper_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
CMRxRecon File Handling Helper

This module provides functions for safely handling CMRxRecon dataset files,
including skipping problematic files and safely loading data.

Usage:
    from file_handling_helper import load_safely, is_valid_file

    # Check if a file is valid
    if is_valid_file(file_path, 'problematic_files.txt'):
        # Process the file
        data = load_safely(file_path)
        if data is not None:
            # Use the data
            pass
\"\"\"

import os
import h5py
import numpy as np
import logging
import time


def load_problematic_files(file_path):
    \"\"\"Load list of problematic files from a text file.
    
    Args:
        file_path (str): Path to the file containing problematic file paths
        
    Returns:
        set: Set of problematic file paths
    \"\"\"
    problematic_files = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    problematic_files.add(line)
    except Exception as e:
        print(f"Error loading problematic file list: {e}")
    
    return problematic_files


def is_valid_file(file_path, problematic_files_path=None):
    \"\"\"Check if a file is valid (not in the problematic files list).
    
    Args:
        file_path (str): Path to the file to check
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        bool: True if the file is valid, False otherwise
    \"\"\"
    if problematic_files_path is None or not os.path.exists(problematic_files_path):
        return True
    
    problematic_files = load_problematic_files(problematic_files_path)
    
    # Normalize path for comparison
    normalized_path = os.path.normpath(file_path)
    
    return normalized_path not in problematic_files


def load_safely(file_path, max_retries=3, retry_delay=1.0, problematic_files_path=None):
    \"\"\"Safely load data from a file, handling exceptions.
    
    Args:
        file_path (str): Path to the file
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        dict or None: Dictionary containing loaded data, or None if loading failed
    \"\"\"
    # Check if file is in problematic list
    if problematic_files_path and not is_valid_file(file_path, problematic_files_path):
        print(f"Skipping known problematic file: {file_path}")
        return None
    
    # Try to load the file with retries
    for attempt in range(max_retries):
        try:
            result = {}
            
            with h5py.File(file_path, 'r') as hf:
                # Check for kspace data
                if 'kspace' in hf:
                    if 'real' in hf['kspace'] and 'imag' in hf['kspace']:
                        result['kspace'] = hf['kspace']['real'][()] + 1j * hf['kspace']['imag'][()]
                        result['type'] = 'kspace'
                
                # Check for mask data
                if 'mask' in hf:
                    result['mask'] = hf['mask'][()]
                    result['type'] = 'mask'
                
                # If no recognized data was found, get the first available data
                if not result:
                    keys = list(hf.keys())
                    if keys:
                        first_key = keys[0]
                        result['data'] = hf[first_key][()]
                        result['type'] = 'other'
                        result['key'] = first_key
                
                return result
            
        except Exception as e:
            print(f"Error loading {file_path} (attempt {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print(f"Failed to load {file_path} after {max_retries} attempts")
    return None


def load_kspace(file_path, frame_idx=0, slice_idx=None, problematic_files_path=None):
    \"\"\"Safely load k-space data from a file.
    
    Args:
        file_path (str): Path to the file
        frame_idx (int): Frame index to extract
        slice_idx (int, optional): Slice index to extract
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        numpy.ndarray or None: K-space data, or None if loading failed
    \"\"\"
    result = load_safely(file_path, problematic_files_path=problematic_files_path)
    
    if result is None or 'kspace' not in result:
        return None
    
    kspace_data = result['kspace']
    
    # Handle different shapes
    if len(kspace_data.shape) == 5:  # (nframe, nslice, ncoil, ny, nx)
        # Ensure frame_idx is within bounds
        frame_idx = min(frame_idx, kspace_data.shape[0] - 1)
        
        # If slice_idx is not provided, use the middle slice
        if slice_idx is None:
            slice_idx = kspace_data.shape[1] // 2
        else:
            slice_idx = min(slice_idx, kspace_data.shape[1] - 1)
        
        return kspace_data[frame_idx, slice_idx]
        
    elif len(kspace_data.shape) == 4:  # Could be (nframe, nslice, ny, nx) or similar
        frame_idx = min(frame_idx, kspace_data.shape[0] - 1)
        
        if slice_idx is None:
            slice_idx = kspace_data.shape[1] // 2
        else:
            slice_idx = min(slice_idx, kspace_data.shape[1] - 1)
        
        return kspace_data[frame_idx, slice_idx]
        
    elif len(kspace_data.shape) == 3:  # Could be (ncoil, ny, nx) or similar
        return kspace_data
    
    return kspace_data


def load_mask(file_path, problematic_files_path=None):
    \"\"\"Safely load mask data from a file.
    
    Args:
        file_path (str): Path to the file
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        numpy.ndarray or None: Mask data, or None if loading failed
    \"\"\"
    result = load_safely(file_path, problematic_files_path=problematic_files_path)
    
    if result is None:
        return None
    
    if 'mask' in result:
        return result['mask']
    elif result['type'] == 'mask':
        return result['data']
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CMRxRecon File Handling Helper')
    parser.add_argument('--file', type=str, required=True, help='Path to a file to test')
    parser.add_argument('--problematic-files', type=str, help='Path to problematic files list')
    
    args = parser.parse_args()
    
    # Check if the file is valid
    is_valid = is_valid_file(args.file, args.problematic_files)
    print(f"Is valid file: {is_valid}")
    
    if is_valid:
        # Try to load the file
        data = load_safely(args.file, problematic_files_path=args.problematic_files)
        if data is not None:
            print(f"Successfully loaded file: {args.file}")
            print(f"Data type: {data['type']}")
            if 'kspace' in data:
                print(f"K-space shape: {data['kspace'].shape}")
            elif 'mask' in data:
                print(f"Mask shape: {data['mask'].shape}")
            elif 'data' in data:
                print(f"Data shape: {data['data'].shape}")
        else:
            print(f"Failed to load file: {args.file}")
""")
        
        self.logger.info(f"Helper code saved to {helper_path}")
        
        # Make the file executable
        os.chmod(helper_path, 0o755)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CMRxRecon Dataset Verification Tool')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of the CMRxRecon dataset')
    parser.add_argument('--output-dir', type=str, default='./verification_results', help='Directory to save verification results')
    parser.add_argument('--verification-level', type=str, choices=['basic', 'thorough'], default='basic', help='Level of verification')
    parser.add_argument('--file-types', type=str, nargs='+', default=['.mat'], help='File types to verify')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = CMRVerifier(
        args.data_root,
        args.output_dir,
        args.verification_level,
        args.file_types
    )
    
    # Run verification
    verifier.run_verification(args.workers)
    
    # Generate helper code
    verifier.generate_helper_code()
    
    print(f"Verification complete. Results saved to {args.output_dir}")
    print(f"See {args.output_dir}/verification_report.html for a detailed report")


if __name__ == "__main__":
    main()