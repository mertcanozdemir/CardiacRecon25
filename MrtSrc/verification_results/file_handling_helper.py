#!/usr/bin/env python3
"""
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
"""

import os
import h5py
import numpy as np
import logging
import time


def load_problematic_files(file_path):
    """Load list of problematic files from a text file.
    
    Args:
        file_path (str): Path to the file containing problematic file paths
        
    Returns:
        set: Set of problematic file paths
    """
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
    """Check if a file is valid (not in the problematic files list).
    
    Args:
        file_path (str): Path to the file to check
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        bool: True if the file is valid, False otherwise
    """
    if problematic_files_path is None or not os.path.exists(problematic_files_path):
        return True
    
    problematic_files = load_problematic_files(problematic_files_path)
    
    # Normalize path for comparison
    normalized_path = os.path.normpath(file_path)
    
    return normalized_path not in problematic_files


def load_safely(file_path, max_retries=3, retry_delay=1.0, problematic_files_path=None):
    """Safely load data from a file, handling exceptions.
    
    Args:
        file_path (str): Path to the file
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        dict or None: Dictionary containing loaded data, or None if loading failed
    """
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
    """Safely load k-space data from a file.
    
    Args:
        file_path (str): Path to the file
        frame_idx (int): Frame index to extract
        slice_idx (int, optional): Slice index to extract
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        numpy.ndarray or None: K-space data, or None if loading failed
    """
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
    """Safely load mask data from a file.
    
    Args:
        file_path (str): Path to the file
        problematic_files_path (str, optional): Path to the problematic files list
        
    Returns:
        numpy.ndarray or None: Mask data, or None if loading failed
    """
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
