#!/usr/bin/env python3
"""
Enhanced File Handling Helper for CMRxRecon Dataset

A comprehensive script to handle .mat files in the CMRxRecon dataset with improved
functionality including data extraction, progress tracking, and detailed reporting.

Usage:
    python enhanced_mat_handler.py --file path/to/file.mat
    python enhanced_mat_handler.py --dir path/to/directory --output results.csv
    python enhanced_mat_handler.py --dir path/to/directory --extract --output-dir extracted_data/
"""

import os
import sys
import scipy.io
import h5py
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
import traceback


class MatFileHandler:
    """Enhanced handler for .mat files with comprehensive loading and analysis."""
    
    def __init__(self, verbose=True, patient_id_pattern=None):
        self.verbose = verbose
        self.patient_id_pattern = patient_id_pattern
        self.stats = {
            'total_files': 0,
            'scipy_success': 0,
            'h5py_success': 0,
            'failed': 0,
            'errors': []
        }
    
    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_mat_file(self, file_path, extract_data=False):
        """Load a .mat file with comprehensive error handling and data extraction.
        
        Args:
            file_path (str): Path to the .mat file
            extract_data (bool): Whether to extract actual data arrays
            
        Returns:
            dict: Result dictionary with status, data, metadata, etc.
        """
        result = {
            'file_path': file_path,
            'status': 'failed',
            'loader': None,
            'error': None,
            'metadata': {},
            'data': None,
            'file_size': 0,
            'variables': []
        }
        
        try:
            # Get file size
            result['file_size'] = os.path.getsize(file_path)
            
            # Try scipy.io first
            try:
                data = scipy.io.loadmat(file_path)
                result['status'] = 'success'
                result['loader'] = 'scipy'
                result['variables'] = [k for k in data.keys() if not k.startswith('__')]
                
                # Extract metadata
                result['metadata'] = {
                    'matlab_version': data.get('__version__', 'unknown'),
                    'header': str(data.get('__header__', 'unknown')),
                    'globals': data.get('__globals__', [])
                }
                
                if extract_data:
                    # Only keep non-metadata variables
                    result['data'] = {k: v for k, v in data.items() if not k.startswith('__')}
                
                self.stats['scipy_success'] += 1
                return result
                
            except Exception as scipy_error:
                # Try h5py for MATLAB v7.3+ files
                try:
                    result = self._load_with_h5py(file_path, extract_data, result)
                    self.stats['h5py_success'] += 1
                    return result
                    
                except Exception as h5py_error:
                    # Both methods failed
                    result['error'] = {
                        'scipy_error': str(scipy_error),
                        'h5py_error': str(h5py_error),
                        'traceback': traceback.format_exc()
                    }
                    self.stats['failed'] += 1
                    self.stats['errors'].append(result['error'])
                    
        except Exception as e:
            result['error'] = {
                'general_error': str(e),
                'traceback': traceback.format_exc()
            }
            self.stats['failed'] += 1
            
        return result
    
    def _load_with_h5py(self, file_path, extract_data, result):
        """Load file using h5py with detailed data extraction."""
        with h5py.File(file_path, 'r') as hf:
            result['status'] = 'success'
            result['loader'] = 'h5py'
            result['variables'] = list(hf.keys())
            
            # Extract metadata
            result['metadata'] = {
                'hdf5_version': getattr(hf, 'libver', 'unknown'),
                'file_format': 'HDF5 (MATLAB v7.3+)'
            }
            
            if extract_data:
                result['data'] = {}
                
            # Analyze each variable
            variable_info = {}
            for key in hf.keys():
                try:
                    dataset = hf[key]
                    var_info = {
                        'shape': dataset.shape if hasattr(dataset, 'shape') else 'unknown',
                        'dtype': str(dataset.dtype) if hasattr(dataset, 'dtype') else 'unknown',
                        'size_mb': (dataset.size * dataset.dtype.itemsize / 1024 / 1024) if hasattr(dataset, 'size') and hasattr(dataset, 'dtype') else 0
                    }
                    variable_info[key] = var_info
                    
                    # Special handling for CMRxRecon data structures
                    if key == 'kspace':
                        if 'real' in dataset and 'imag' in dataset:
                            var_info['type'] = 'complex_kspace'
                            var_info['real_shape'] = dataset['real'].shape
                            var_info['imag_shape'] = dataset['imag'].shape
                    elif key == 'mask':
                        var_info['type'] = 'mask'
                    
                    if extract_data and dataset.size < 1e8:  # Only extract if less than 100MB
                        try:
                            result['data'][key] = np.array(dataset)
                        except Exception:
                            result['data'][key] = f"<Large dataset: {var_info['shape']}>"
                    
                except Exception as e:
                    variable_info[key] = {'error': str(e)}
            
            result['metadata']['variables'] = variable_info
            
        return result
    
    def check_single_file(self, file_path, extract_data=False):
        """Check a single .mat file and return detailed results."""
        self.log(f"Checking file: {file_path}")
        
        if not os.path.exists(file_path):
            return {'error': 'File not found', 'file_path': file_path}
        
        result = self.load_mat_file(file_path, extract_data)
        self.stats['total_files'] += 1
        
        # Print results
        if result['status'] == 'success':
            self.log(f"✓ Successfully loaded with {result['loader']}")
            self.log(f"  Variables: {result['variables']}")
            self.log(f"  File size: {result['file_size'] / 1024 / 1024:.2f} MB")
            
            if result['loader'] == 'h5py' and 'variables' in result['metadata']:
                for var, info in result['metadata']['variables'].items():
                    if 'shape' in info:
                        self.log(f"    {var}: {info['shape']} ({info.get('dtype', 'unknown')})")
        else:
            self.log(f"✗ Failed to load")
            if result['error']:
                self.log(f"  Error: {result['error']}")
        
        return result
    
    def check_directory(self, directory_path, output_file=None, extract_data=False, output_dir=None):
        """Check all .mat files in a directory with progress tracking."""
        self.log(f"Scanning directory: {directory_path}")
        
        # Find all .mat files
        mat_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        
        self.log(f"Found {len(mat_files)} .mat files")
        
        results = []
        
        # Process files with progress
        for i, file_path in enumerate(mat_files, 1):
            self.log(f"[{i}/{len(mat_files)}] Processing: {os.path.basename(file_path)}")
            
            result = self.load_mat_file(file_path, extract_data)
            results.append(result)
            self.stats['total_files'] += 1
            
            # Save extracted data if requested
            if extract_data and output_dir and result['status'] == 'success' and result['data']:
                self._save_extracted_data(result, output_dir)
        
        # Save results to file
        if output_file:
            self._save_results(results, output_file)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _extract_patient_id(self, file_path, data=None):
        """Extract patient ID from file path or data.
        
        Args:
            file_path (str): Path to the .mat file
            data (dict, optional): Loaded data dictionary
            
        Returns:
            str: Patient ID or 'unknown' if not found
        """
        try:
            # Method 0: Use custom pattern if provided
            if self.patient_id_pattern:
                import re
                match = re.search(self.patient_id_pattern, file_path)
                if match:
                    return match.group(1) if match.groups() else match.group(0)
            
            # Method 1: Extract from file path (common patterns)
            path_parts = Path(file_path).parts
            
            # Look for common patient ID patterns in path
            for part in path_parts:
                # Pattern like "P001", "Patient001", "patient_001", etc.
                if part.lower().startswith('p') and any(c.isdigit() for c in part):
                    return part
                # Pattern like "001" (3+ digits)
                if part.isdigit() and len(part) >= 3:
                    return f"P{part}"
                # Pattern like "sub-001", "subject001"
                if 'sub' in part.lower() and any(c.isdigit() for c in part):
                    return part.replace('sub-', 'P').replace('subject', 'P')
            
            # Method 2: Extract from filename
            filename = Path(file_path).stem
            # Look for patterns like "kspace_P001_slice1" or "mask_001_cardiac"
            parts = filename.split('_')
            for part in parts:
                if part.lower().startswith('p') and any(c.isdigit() for c in part):
                    return part
                if part.isdigit() and len(part) >= 3:
                    return f"P{part}"
            
            # Method 3: Extract from data (if available)
            if data:
                # Look for patient ID in variable names or data
                for key in data.keys():
                    if 'patient' in key.lower() or 'subject' in key.lower():
                        try:
                            if isinstance(data[key], (str, int)):
                                return f"P{data[key]}"
                        except:
                            pass
            
            # Method 4: Use directory structure
            # Look for parent directories that might contain patient info
            parent_dirs = Path(file_path).parents
            for parent in parent_dirs:
                dir_name = parent.name
                if dir_name.lower().startswith('p') and any(c.isdigit() for c in dir_name):
                    return dir_name
                if dir_name.isdigit() and len(dir_name) >= 3:
                    return f"P{dir_name}"
            
            return "unknown"
            
        except Exception as e:
            self.log(f"Warning: Failed to extract patient ID from {file_path}: {e}")
            return "unknown"

    def _save_extracted_data(self, result, output_dir):
        """Save extracted data to files with patient ID in filename."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_stem = Path(result['file_path']).stem
            
            # Extract patient ID
            patient_id = self._extract_patient_id(result['file_path'], result['data'])
            
            # Create filename with patient ID
            if patient_id != "unknown":
                base_filename = f"{patient_id}_{file_stem}"
            else:
                base_filename = file_stem
                self.log(f"Warning: Could not extract patient ID for {result['file_path']}")
            
            # Save as numpy arrays
            for var_name, data in result['data'].items():
                if isinstance(data, np.ndarray):
                    save_path = output_path / f"{base_filename}_{var_name}.npy"
                    np.save(save_path, data)
                    self.log(f"  Saved: {save_path.name}")
            
            # Save metadata as JSON (include patient ID in metadata)
            metadata = self._convert_for_json(result['metadata'])
            metadata['patient_id'] = patient_id
            metadata['original_filename'] = Path(result['file_path']).name
            metadata['extraction_timestamp'] = datetime.now().isoformat()
            
            metadata_path = output_path / f"{base_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log(f"  Saved: {metadata_path.name}")
                
        except Exception as e:
            self.log(f"Warning: Failed to save extracted data for {result['file_path']}: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _save_results(self, results, output_file):
        """Save results to CSV file."""
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write("file_path,patient_id,status,loader,file_size_mb,variables,error\n")
                
                # Write data
                for result in results:
                    file_path = result['file_path']
                    # Extract patient ID for CSV
                    patient_id = self._extract_patient_id(file_path, result.get('data'))
                    status = result['status']
                    loader = result.get('loader', '')
                    file_size_mb = result['file_size'] / 1024 / 1024
                    variables = ';'.join(result['variables']) if result['variables'] else ''
                    error = str(result['error']) if result['error'] else ''
                    
                    f.write(f'"{file_path}",{patient_id},{status},{loader},{file_size_mb:.2f},"{variables}","{error}"\n')
            
            self.log(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.log(f"Failed to save results: {e}")
    
    def _print_summary(self):
        """Print comprehensive summary statistics."""
        total = self.stats['total_files']
        
        self.log("\n" + "="*60)
        self.log("SUMMARY REPORT")
        self.log("="*60)
        self.log(f"Total files processed: {total}")
        self.log(f"Successfully loaded: {self.stats['scipy_success'] + self.stats['h5py_success']}")
        self.log(f"  - via scipy.io: {self.stats['scipy_success']}")
        self.log(f"  - via h5py: {self.stats['h5py_success']}")
        self.log(f"Failed to load: {self.stats['failed']}")
        
        if total > 0:
            success_rate = (self.stats['scipy_success'] + self.stats['h5py_success']) / total * 100
            self.log(f"Success rate: {success_rate:.1f}%")
        
        # Save problematic files list
        if self.stats['failed'] > 0:
            with open("problematic_files.txt", 'w') as f:
                f.write(f"# Problematic .mat files - Generated on {datetime.now()}\n")
                for i, error in enumerate(self.stats['errors']):
                    f.write(f"Error {i+1}: {error}\n\n")
            
            self.log(f"\nDetailed error information saved to: problematic_files.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced CMRxRecon .mat File Handler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file data.mat
  %(prog)s --dir /path/to/data --output results.csv
  %(prog)s --dir /path/to/data --extract --output-dir extracted/
  %(prog)s --dir /path/to/data --extract --patient-pattern "P(\d+)" --output-dir extracted/
        """
    )
    
    parser.add_argument('--file', type=str, help='Path to a specific .mat file to analyze')
    parser.add_argument('--dir', type=str, help='Path to directory containing .mat files')
    parser.add_argument('--output', type=str, help='Output CSV file for directory scan results')
    parser.add_argument('--extract', action='store_true', help='Extract data arrays from files')
    parser.add_argument('--output-dir', type=str, help='Directory to save extracted data')
    parser.add_argument('--patient-pattern', type=str, help='Regex pattern to extract patient ID (use parentheses for capture group)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.print_help()
        print("\nError: Please provide either --file or --dir argument")
        sys.exit(1)
    
    # Create handler
    handler = MatFileHandler(verbose=not args.quiet, patient_id_pattern=args.patient_pattern)
    
    if args.file:
        # Check single file
        result = handler.check_single_file(args.file, extract_data=args.extract)
        
        if args.extract and args.output_dir and result['status'] == 'success':
            handler._save_extracted_data(result, args.output_dir)
    
    if args.dir:
        # Check directory
        handler.check_directory(
            args.dir, 
            output_file=args.output,
            extract_data=args.extract,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()