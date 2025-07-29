import scipy.io
import h5py
import numpy as np
import os

def open_mat_file(file_path):
    """
    Opens a .mat file using appropriate method based on MATLAB version
    
    Args:
        file_path (str): Path to the .mat file
    
    Returns:
        dict: Dictionary containing the loaded data
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    try:
        # Method 1: Try scipy.io.loadmat first (works for MATLAB v7.3 and earlier)
        print("Attempting to load with scipy.io.loadmat...")
        data = scipy.io.loadmat(file_path)
        print("✓ Successfully loaded with scipy.io.loadmat")
        return data
        
    except NotImplementedError:
        # Method 2: Use h5py for MATLAB v7.3+ files (HDF5 format)
        print("scipy.io.loadmat failed, trying h5py for v7.3+ files...")
        try:
            data = {}
            with h5py.File(file_path, 'r') as f:
                # Convert h5py dataset to dictionary
                for key in f.keys():
                    data[key] = np.array(f[key])
            print("✓ Successfully loaded with h5py")
            return data
        except Exception as e:
            print(f"✗ h5py failed: {e}")
            raise
    
    except Exception as e:
        print(f"✗ scipy.io.loadmat failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Replace with your .mat file path
    mat_file_path = 'F:/CMRxRecon2025/ChallengeData/MultiCoil/Flow2d/TrainingSet/FullSample/Center006/Siemens_30T_Prisma/P001/flow2d.mat'
    
    try:
        # Load the .mat file
        mat_data = open_mat_file(mat_file_path)
        
        # Display basic information about the loaded data
        print("\n" + "="*50)
        print("MAT FILE CONTENTS:")
        print("="*50)
        
        for key, value in mat_data.items():
            if not key.startswith('__'):  # Skip MATLAB metadata
                print(f"Variable: {key}")
                print(f"  Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"  Data type: {value.dtype}")
                print()
        
        # Example: Access specific variables
        # Replace 'variable_name' with actual variable names from your .mat file
        # my_variable = mat_data['variable_name']
        # print(f"My variable: {my_variable}")
        
    except FileNotFoundError:
        print("Please update 'mat_file_path' with the correct path to your .mat file")
    except Exception as e:
        print(f"Error loading .mat file: {e}")

# Alternative simple methods:

def simple_load_mat(file_path):
    """Simple method using only scipy.io.loadmat"""
    return scipy.io.loadmat(file_path)

def load_mat_h5py(file_path):
    """Method specifically for MATLAB v7.3+ files"""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
    return data

# Usage examples:
# data = simple_load_mat('file.mat')
# data = load_mat_h5py('file_v73.mat')  # For v7.3+ files
# For most .mat files (simple approach)
