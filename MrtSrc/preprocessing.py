"""
CMRxRecon2025 Preprocessing Module

Complete preprocessing pipeline for CMR reconstruction with multi-center generalization (Task 1).
This module includes all necessary functions for preprocessing k-space data, including:
- Loading and saving data
- K-space normalization
- Coil sensitivity map estimation
- Data augmentation
- Dimension standardization
- Undersampling mask application/generation
- Visualization utilities
"""

import os
import numpy as np
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
import time
from tqdm import tqdm
import torch
import fastmri
from fastmri.data import transforms as T


class CMRPreprocessor:
    """
    Comprehensive preprocessing pipeline for CMR reconstruction data.
    Handles multi-center data with various normalization and preprocessing strategies.
    """
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor with configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with preprocessing parameters
        """
        # Default configuration
        self.config = {
            'target_size': (320, 320),  # Standard size for matrix dimensions
            'norm_method': 'center_aware',  # Normalization method
            'norm_percentile': 99.5,  # For percentile normalization
            'center_fraction': 0.04,  # For central region normalization
            'sensitivity_method': 'espirit',  # Method for sensitivity map estimation
            'augmentation': True,  # Whether to apply data augmentation
            'augment_prob': 0.5,  # Probability of applying each augmentation
            'standardize_dims': True,  # Whether to standardize dimensions
            'zero_fill_missing': True,  # Whether to zero-fill missing k-space lines
            'verbose': True,  # Whether to print progress information
        }
        
        # Update with user configuration if provided
        if config is not None:
            self.config.update(config)
        
        # Initialize vendor-specific scaling factors
        self.vendor_scale_factors = {
            'UIH': 1.2,
            'Siemens': 0.9,
            'GE': 1.1,
            'Philips': 1.0,
        }
    
    # -------------------- DATA LOADING AND SAVING --------------------
    
    def load_kspace(self, file_path):
        """
        Load k-space data from .mat file.
        
        Parameters:
        -----------
        file_path : str
            Path to the .mat file containing k-space data
            
        Returns:
        --------
        complex ndarray
            K-space data with shape [nframes, nslices, ncoils, ny, nx] or [nframes, nslices, ny, nx] for single-coil
        """
        try:
            with h5py.File(file_path, 'r') as hf:
                kdata = hf['kspace']
                # Handle complex data stored as real and imaginary parts
                kspace = kdata['real'][()] + 1j * kdata['imag'][()]
                
            if self.config['verbose']:
                print(f"Loaded k-space with shape {kspace.shape} from {file_path}")
            
            return kspace
        except Exception as e:
            print(f"Error loading k-space data from {file_path}: {e}")
            return None
    
    def load_mask(self, file_path):
        """
        Load undersampling mask from .mat file.
        
        Parameters:
        -----------
        file_path : str
            Path to the .mat file containing mask data
            
        Returns:
        --------
        ndarray
            Undersampling mask with shape [nframes, ny, nx]
        """
        try:
            with h5py.File(file_path, 'r') as hf:
                # Handle different mask formats
                if 'mask' in hf:
                    mask = hf['mask'][()]
                else:
                    # Some mask files have different keys
                    keys = list(hf.keys())
                    if len(keys) > 0:
                        mask = hf[keys[0]][()]
                    else:
                        raise ValueError("No mask data found in file")
                
            if self.config['verbose']:
                print(f"Loaded mask with shape {mask.shape} from {file_path}")
            
            return mask
        except Exception as e:
            print(f"Error loading mask data from {file_path}: {e}")
            return None
    
    def save_to_mat(self, data, var_name, file_path):
        """
        Save data to .mat file format using hdf5storage.
        
        Parameters:
        -----------
        data : ndarray
            Data to be saved
        var_name : str
            Variable name in the .mat file
        file_path : str
            Path to save the .mat file
        """
        save_dict = {var_name: data}
        try:
            hdf5storage.savemat(file_path, save_dict)
            if self.config['verbose']:
                print(f"Saved {var_name} to {file_path}")
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")
    
    def get_metadata_from_path(self, file_path):
        """
        Extract center and scanner information from file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
            
        Returns:
        --------
        dict
            Dictionary with center_id and scanner_type
        """
        # Extract center and scanner info from path structure
        parts = file_path.split(os.sep)
        
        # Initialize with defaults
        metadata = {
            'center_id': 'Unknown',
            'scanner_type': 'Unknown',
            'patient_id': 'Unknown',
            'modality': 'Unknown'
        }
        
        # Look for center and scanner information in path
        for i, part in enumerate(parts):
            if part.startswith('Center'):
                metadata['center_id'] = part
                # Scanner type is typically the next folder
                if i+1 < len(parts):
                    metadata['scanner_type'] = parts[i+1]
            
            # Look for patient ID
            if part.startswith('P') and part[1:].isdigit():
                metadata['patient_id'] = part
            
            # Look for modality information
            for modality in ['Cine', 'LGE', 'T1map', 'T2map', 'Flow2d', 'BlackBlood']:
                if modality.lower() in file_path.lower():
                    metadata['modality'] = modality
                    break
        
        return metadata
    
    # -------------------- K-SPACE NORMALIZATION METHODS --------------------
    
    def normalize_kspace(self, kspace, metadata=None):
        """
        Normalize k-space data using the specified method.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        metadata : dict, optional
            Dictionary with metadata including center_id and scanner_type
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data
        """
        method = self.config['norm_method']
        
        if method == 'instance':
            return self._instance_normalize_kspace(kspace)
        elif method == 'percentile':
            return self._percentile_normalize_kspace(kspace, self.config['norm_percentile'])
        elif method == 'central_region':
            return self._central_region_normalize_kspace(kspace, self.config['center_fraction'])
        elif method == 'coil_wise':
            return self._coil_wise_normalize_kspace(kspace)
        elif method == 'center_aware':
            # Check if metadata is provided
            if metadata is None:
                print("Warning: No metadata provided for center-aware normalization. Falling back to instance normalization.")
                return self._instance_normalize_kspace(kspace)
            return self._center_aware_normalize_kspace(kspace, metadata)
        else:
            print(f"Unknown normalization method: {method}. Using instance normalization.")
            return self._instance_normalize_kspace(kspace)
    
    def _instance_normalize_kspace(self, kspace):
        """
        Normalize each k-space volume independently by its maximum magnitude.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data
        """
        # Calculate the maximum magnitude across all dimensions
        k_max = np.max(np.abs(kspace))
        
        # Avoid division by zero
        eps = 1e-9
        
        # Normalize by maximum value
        normalized = kspace / (k_max + eps)
        
        return normalized
    
    def _percentile_normalize_kspace(self, kspace, percentile=99.5):
        """
        Normalize k-space using a high percentile value instead of maximum.
        More robust to extreme outliers.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        percentile : float
            Percentile value for normalization (default: 99.5)
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data
        """
        # Calculate the specified percentile of k-space magnitude
        k_percentile = np.percentile(np.abs(kspace), percentile)
        
        # Normalize
        normalized = kspace / (k_percentile + 1e-9)
        
        return normalized
    
    def _central_region_normalize_kspace(self, kspace, center_fraction=0.04):
        """
        Normalize based on the central region of k-space where most energy is concentrated.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data with shape [nframes, nslices, ncoils, ny, nx]
        center_fraction : float
            Fraction of central k-space to use for normalization
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data
        """
        # Get dimensions
        ny, nx = kspace.shape[-2], kspace.shape[-1]
        
        # Calculate center region size
        center_y = int(ny * center_fraction)
        center_x = int(nx * center_fraction)
        
        # Extract center region
        y_start = (ny - center_y) // 2
        y_end = y_start + center_y
        x_start = (nx - center_x) // 2
        x_end = x_start + center_x
        
        center_kspace = kspace[..., y_start:y_end, x_start:x_end]
        
        # Calculate maximum of center region
        center_max = np.max(np.abs(center_kspace))
        
        # Normalize using center region statistics
        normalized = kspace / (center_max + 1e-9)
        
        return normalized
    
    def _coil_wise_normalize_kspace(self, kspace):
        """
        Normalize each coil independently.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data with shape [nframes, nslices, ncoils, ny, nx]
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data with each coil normalized independently
        """
        # Create output array
        normalized = np.zeros_like(kspace)
        
        # Normalize each coil independently
        for coil_idx in range(kspace.shape[2]):
            coil_data = kspace[:, :, coil_idx, :, :]
            coil_max = np.max(np.abs(coil_data))
            normalized[:, :, coil_idx, :, :] = coil_data / (coil_max + 1e-9)
        
        return normalized
    
    def _center_aware_normalize_kspace(self, kspace, metadata):
        """
        Apply center and scanner-aware normalization.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        metadata : dict
            Dictionary with center_id and scanner_type
            
        Returns:
        --------
        complex ndarray
            Normalized k-space data
        """
        scanner_type = metadata.get('scanner_type', 'Unknown')
        
        # Extract field strength from scanner_type
        if "15T" in scanner_type:
            field_strength = 1.5
        elif "30T" in scanner_type:
            field_strength = 3.0
        elif "50T" in scanner_type:
            field_strength = 5.0
        else:
            field_strength = 3.0  # Default
        
        # Calculate normalization factor
        # Scale factor compensates for field strength differences
        # Higher field strength typically produces stronger signal
        scale_factor = (field_strength / 3.0) ** 2
        
        # Apply vendor-specific corrections
        for vendor, factor in self.vendor_scale_factors.items():
            if vendor in scanner_type:
                scale_factor *= factor
                break
        
        # Calculate maximum for scaling
        k_max = np.max(np.abs(kspace))
        
        # Apply normalization with center-aware scaling
        normalized = kspace / (k_max * scale_factor + 1e-9)
        
        return normalized
    
    # -------------------- COIL SENSITIVITY MAP ESTIMATION --------------------
    
    def estimate_sensitivity_maps(self, kspace, method=None):
        """
        Estimate coil sensitivity maps from k-space data.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data with shape [nframes, nslices, ncoils, ny, nx]
        method : str, optional
            Method to use for sensitivity map estimation. If None, uses the configured method.
            
        Returns:
        --------
        complex ndarray
            Estimated sensitivity maps with shape [ncoils, ny, nx]
        """
        if method is None:
            method = self.config['sensitivity_method']
        
        # Select a single frame and slice for sensitivity estimation
        frame_idx = 0
        slice_idx = kspace.shape[1] // 2  # Middle slice
        
        # Extract k-space for the selected frame and slice
        k_frame_slice = kspace[frame_idx, slice_idx]
        
        if method == 'espirit':
            return self._espirit_sensitivity_maps(k_frame_slice)
        elif method == 'walsh':
            return self._walsh_sensitivity_maps(k_frame_slice)
        elif method == 'simple':
            return self._simple_sensitivity_maps(k_frame_slice)
        else:
            print(f"Unknown sensitivity map estimation method: {method}. Using simple method.")
            return self._simple_sensitivity_maps(k_frame_slice)
    
    def _espirit_sensitivity_maps(self, kspace_slice):
        """
        Estimate sensitivity maps using ESPIRiT method.
        
        Parameters:
        -----------
        kspace_slice : complex ndarray
            K-space data for a single slice with shape [ncoils, ny, nx]
            
        Returns:
        --------
        complex ndarray
            Estimated sensitivity maps
        """
        # Note: Full ESPIRiT implementation requires additional dependencies
        # For a complete implementation, consider using the sigpy package
        # Here we use a simplified approach based on calibration region
        
        # Extract calibration region (center of k-space)
        ny, nx = kspace_slice.shape[-2], kspace_slice.shape[-1]
        calib_size = min(32, ny // 4)  # Use at most 32 lines or 1/4 of k-space
        
        y_start = (ny - calib_size) // 2
        y_end = y_start + calib_size
        x_start = (nx - calib_size) // 2
        x_end = x_start + calib_size
        
        calib = kspace_slice[:, y_start:y_end, x_start:x_end]
        
        # Transform calibration region to image domain
        calib_img = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(calib, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # Calculate coil combination weights (simplified ESPIRiT)
        coil_images = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        coil_combined = np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))
        
        # Normalize by combined image to get sensitivity maps
        sens_maps = np.zeros_like(coil_images)
        for i in range(coil_images.shape[0]):
            sens_maps[i] = coil_images[i] / (coil_combined + 1e-9)
        
        # Smooth sensitivity maps
        for i in range(sens_maps.shape[0]):
            sens_maps[i] = ndimage.gaussian_filter(np.real(sens_maps[i]), sigma=1) + \
                          1j * ndimage.gaussian_filter(np.imag(sens_maps[i]), sigma=1)
        
        return sens_maps
    
    def _walsh_sensitivity_maps(self, kspace_slice):
        """
        Estimate sensitivity maps using Walsh method.
        
        Parameters:
        -----------
        kspace_slice : complex ndarray
            K-space data for a single slice with shape [ncoils, ny, nx]
            
        Returns:
        --------
        complex ndarray
            Estimated sensitivity maps
        """
        # Transform to image domain
        coil_images = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # Calculate coil combination weights
        coil_combined = np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))
        
        # Get sensitivity maps
        sens_maps = np.zeros_like(coil_images)
        for i in range(coil_images.shape[0]):
            sens_maps[i] = coil_images[i] / (coil_combined + 1e-9)
        
        return sens_maps
    
    def _simple_sensitivity_maps(self, kspace_slice):
        """
        Simple sensitivity map estimation using low-resolution images.
        
        Parameters:
        -----------
        kspace_slice : complex ndarray
            K-space data for a single slice with shape [ncoils, ny, nx]
            
        Returns:
        --------
        complex ndarray
            Estimated sensitivity maps
        """
        # Get dimensions
        ncoils, ny, nx = kspace_slice.shape
        
        # Create low-pass filter mask (central 25% of k-space)
        mask = np.zeros((ny, nx))
        y_start = ny // 2 - ny // 8
        y_end = ny // 2 + ny // 8
        x_start = nx // 2 - nx // 8
        x_end = nx // 2 + nx // 8
        mask[y_start:y_end, x_start:x_end] = 1
        
        # Apply low-pass filter
        low_res_kspace = kspace_slice * mask
        
        # Transform to image domain
        low_res_images = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(low_res_kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # Compute sum-of-squares combination
        sos = np.sqrt(np.sum(np.abs(low_res_images)**2, axis=0))
        
        # Calculate sensitivity maps
        sens_maps = np.zeros_like(low_res_images)
        for i in range(ncoils):
            sens_maps[i] = low_res_images[i] / (sos + 1e-9)
        
        return sens_maps
    
    # -------------------- DIMENSION STANDARDIZATION --------------------
    
    def standardize_dimensions(self, kspace, target_size=None):
        """
        Standardize k-space dimensions to a target size through padding or cropping.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        target_size : tuple, optional
            Target dimensions (ny, nx). If None, uses the configured target size.
            
        Returns:
        --------
        complex ndarray
            K-space data with standardized dimensions
        """
        if not self.config['standardize_dims']:
            return kspace
            
        if target_size is None:
            target_size = self.config['target_size']
            
        # Get current dimensions
        ny, nx = kspace.shape[-2], kspace.shape[-1]
        
        # If dimensions already match, return original
        if ny == target_size[0] and nx == target_size[1]:
            return kspace
        
        # Determine shape of output array
        out_shape = list(kspace.shape)
        out_shape[-2] = target_size[0]
        out_shape[-1] = target_size[1]
        
        # Create output array
        standardized = np.zeros(out_shape, dtype=kspace.dtype)
        
        # Zero-padding for smaller matrices
        if ny < target_size[0] or nx < target_size[1]:
            y_offset = (target_size[0] - ny) // 2
            x_offset = (target_size[1] - nx) // 2
            
            # Handle y dimension
            y_start_out = y_offset
            y_end_out = y_offset + min(ny, target_size[0])
            y_start_in = 0
            y_end_in = min(ny, target_size[0])
            
            # Handle x dimension
            x_start_out = x_offset
            x_end_out = x_offset + min(nx, target_size[1])
            x_start_in = 0
            x_end_in = min(nx, target_size[1])
            
            # Copy data with appropriate slicing
            if len(kspace.shape) == 5:  # Multi-coil, multi-frame, multi-slice
                standardized[:, :, :, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, :, :, y_start_in:y_end_in, x_start_in:x_end_in]
            elif len(kspace.shape) == 4:  # Multi-coil, single-frame or multi-frame, single-slice
                standardized[:, :, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, :, y_start_in:y_end_in, x_start_in:x_end_in]
            elif len(kspace.shape) == 3:  # Single-coil, single-frame, multi-slice or multi-coil, single-frame, single-slice
                standardized[:, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, y_start_in:y_end_in, x_start_in:x_end_in]
            else:
                standardized[y_start_out:y_end_out, x_start_out:x_end_out] = kspace[y_start_in:y_end_in, x_start_in:x_end_in]
        
        # Center cropping for larger matrices
        elif ny > target_size[0] or nx > target_size[1]:
            y_offset = (ny - target_size[0]) // 2
            x_offset = (nx - target_size[1]) // 2
            
            # Handle y dimension
            y_start_in = y_offset
            y_end_in = y_offset + target_size[0]
            y_start_out = 0
            y_end_out = target_size[0]
            
            # Handle x dimension
            x_start_in = x_offset
            x_end_in = x_offset + target_size[1]
            x_start_out = 0
            x_end_out = target_size[1]
            
            # Copy data with appropriate slicing
            if len(kspace.shape) == 5:  # Multi-coil, multi-frame, multi-slice
                standardized[:, :, :, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, :, :, y_start_in:y_end_in, x_start_in:x_end_in]
            elif len(kspace.shape) == 4:  # Multi-coil, single-frame or multi-frame, single-slice
                standardized[:, :, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, :, y_start_in:y_end_in, x_start_in:x_end_in]
            elif len(kspace.shape) == 3:  # Single-coil, single-frame, multi-slice or multi-coil, single-frame, single-slice
                standardized[:, y_start_out:y_end_out, x_start_out:x_end_out] = kspace[:, y_start_in:y_end_in, x_start_in:x_end_in]
            else:
                standardized[y_start_out:y_end_out, x_start_out:x_end_out] = kspace[y_start_in:y_end_in, x_start_in:x_end_in]
        
        return standardized
    
    # -------------------- UNDERSAMPLING METHODS --------------------
    
    def apply_undersampling(self, kspace, mask=None, acceleration=None, pattern=None):
        """
        Apply undersampling to k-space data.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        mask : ndarray, optional
            Undersampling mask to apply. If None, generates a mask.
        acceleration : float, optional
            Acceleration factor for generated mask
        pattern : str, optional
            Undersampling pattern type ('uniform', 'gaussian', 'radial')
            
        Returns:
        --------
        tuple
            (undersampled_kspace, applied_mask)
        """
        # If mask is provided, apply it
        if mask is not None:
            # Ensure mask dimensions match k-space
            if len(mask.shape) == 3 and len(kspace.shape) == 5:  # Mask: [nframes, ny, nx], K-space: [nframes, nslices, ncoils, ny, nx]
                # Expand mask dimensions to match k-space
                mask_expanded = mask[:, np.newaxis, np.newaxis, :, :]
                undersampled = kspace * mask_expanded
                return undersampled, mask
            else:
                # Apply mask with appropriate broadcasting
                undersampled = kspace * mask
                return undersampled, mask
        
        # Generate and apply mask if not provided
        if acceleration is None:
            acceleration = 4.0  # Default acceleration factor
            
        if pattern is None:
            pattern = 'radial'  # Default pattern
            
        # Generate mask
        mask = self.generate_mask(kspace.shape, acceleration, pattern)
        
        # Apply mask
        undersampled = kspace * mask
        
        return undersampled, mask
    
    def generate_mask(self, shape, acceleration, pattern='radial'):
        """
        Generate undersampling mask.
        
        Parameters:
        -----------
        shape : tuple
            Shape of the k-space data
        acceleration : float
            Acceleration factor
        pattern : str
            Undersampling pattern ('uniform', 'gaussian', 'radial')
            
        Returns:
        --------
        ndarray
            Undersampling mask
        """
        # Extract relevant dimensions
        if len(shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            nframes, ny, nx = shape[0], shape[3], shape[4]
        elif len(shape) == 4:  # [nframes, nslices, ny, nx] or [nframes, ncoils, ny, nx]
            nframes, ny, nx = shape[0], shape[2], shape[3]
        else:
            # Assume shape includes at least [ny, nx]
            ny, nx = shape[-2], shape[-1]
            nframes = 1
        
        # Initialize mask
        mask = np.zeros((nframes, ny, nx))
        
        # Define central calibration region
        calib_size = 20  # Number of central lines to keep fully sampled
        center_y = ny // 2
        calib_start = center_y - calib_size // 2
        calib_end = calib_start + calib_size
        
        # Fill calibration region
        mask[:, calib_start:calib_end, :] = 1.0
        
        # Apply undersampling pattern outside calibration region
        if pattern == 'uniform':
            # Uniform undersampling (regular skip)
            # Determine sampling stride
            stride = int(np.ceil(acceleration))
            for frame in range(nframes):
                # Sample lines outside calibration region
                for y in range(0, calib_start, stride):
                    mask[frame, y, :] = 1.0
                for y in range(calib_end, ny, stride):
                    mask[frame, y, :] = 1.0
                    
        elif pattern == 'gaussian':
            # Gaussian sampling with higher density near center
            # Create probability density function
            pdf = np.exp(-0.5 * ((np.arange(ny) - center_y) / (ny/8))**2)
            pdf[calib_start:calib_end] = 0  # Don't sample in the calibration region again
            
            # Normalize to desired sampling rate
            num_samples = int(ny / acceleration) - calib_size
            pdf = pdf / np.sum(pdf) * num_samples
            
            # Sample according to pdf
            for frame in range(nframes):
                # Temporal interleaving: offset sampling pattern in each frame
                frame_offset = frame % 2
                for y in range(ny):
                    if calib_start <= y < calib_end:
                        continue  # Skip calibration region
                    # Probability-based sampling
                    if np.random.rand() < pdf[y]:
                        mask[frame, y, :] = 1.0
                        
        elif pattern == 'radial':
            # Radial sampling pattern
            center_x = nx // 2
            
            # Create radial spokes
            num_spokes = int(ny / acceleration)
            for frame in range(nframes):
                # Temporal interleaving: rotate spokes in each frame
                frame_angle_offset = frame * np.pi / nframes
                
                for spoke in range(num_spokes):
                    angle = spoke * np.pi / num_spokes + frame_angle_offset
                    for y in range(ny):
                        if calib_start <= y < calib_end and abs(center_x - nx//2) < calib_size//2:
                            continue  # Skip calibration region
                        
                        # Calculate corresponding x for this y along the spoke
                        dy = y - center_y
                        dx = int(dy * np.tan(angle)) if angle != np.pi/2 else 0
                        x = center_x + dx
                        
                        # Check if x is within bounds
                        if 0 <= x < nx:
                            mask[frame, y, x] = 1.0
            
            # Ensure central calibration region is fully sampled
            mask[:, calib_start:calib_end, center_x-calib_size//2:center_x+calib_size//2] = 1.0
        
        else:
            raise ValueError(f"Unknown undersampling pattern: {pattern}")
        
        # Reshape mask to match the input data dimensions
        if len(shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            mask_reshaped = np.zeros((nframes, 1, 1, ny, nx))
            mask_reshaped[:, 0, 0, :, :] = mask
            return mask_reshaped
        elif len(shape) == 4:  # [nframes, nslices, ny, nx] or [nframes, ncoils, ny, nx]
            mask_reshaped = np.zeros((nframes, 1, ny, nx))
            mask_reshaped[:, 0, :, :] = mask
            return mask_reshaped
        else:
            return mask
    
    # -------------------- DATA AUGMENTATION --------------------
    
    def augment_kspace(self, kspace, sensitivity_maps=None):
        """
        Apply data augmentation to k-space data.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
        sensitivity_maps : complex ndarray, optional
            Sensitivity maps to be augmented in the same way
            
        Returns:
        --------
        tuple
            (augmented_kspace, augmented_sensitivity_maps)
        """
        if not self.config['augmentation']:
            return kspace, sensitivity_maps
        
        augmented = kspace.copy()
        sens_augmented = None if sensitivity_maps is None else sensitivity_maps.copy()
        
        # Phase augmentation (k-space shift = image rotation)
        if np.random.rand() < self.config['augment_prob']:
            phase_factor = np.exp(1j * np.random.uniform(0, 2*np.pi))
            augmented = augmented * phase_factor
        
        # Flip augmentation (axis=1: phase-encoding direction)
        if np.random.rand() < self.config['augment_prob']:
            augmented = np.flip(augmented, axis=-1)
            if sens_augmented is not None:
                sens_augmented = np.flip(sens_augmented, axis=-1)
        
        # Noise augmentation
        if np.random.rand() < self.config['augment_prob']:
            noise_level = np.random.uniform(0, 0.01)  # Adjust range as needed
            noise = noise_level * (np.random.normal(0, 1, augmented.shape) + 
                                  1j * np.random.normal(0, 1, augmented.shape))
            augmented = augmented + noise
        
        # Random masking (simulate motion artifacts)
        if np.random.rand() < self.config['augment_prob'] * 0.5:  # Lower probability for this augmentation
            # Randomly mask a few phase-encoding lines
            num_lines = int(np.random.uniform(1, 5))
            for _ in range(num_lines):
                line_idx = np.random.randint(0, augmented.shape[-2])
                augmented[..., line_idx, :] = 0
        
        return augmented, sens_augmented
    
    # -------------------- TRANSFORMATION METHODS --------------------
    
    def kspace_to_image(self, kspace):
        """
        Transform k-space data to image domain.
        
        Parameters:
        -----------
        kspace : complex ndarray
            Input k-space data
            
        Returns:
        --------
        complex ndarray
            Image domain data
        """
        # Determine data dimensions
        if len(kspace.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            nframes, nslices, ncoils, ny, nx = kspace.shape
            image = np.zeros_like(kspace)
            
            # Process each frame, slice, and coil
            for f in range(nframes):
                for s in range(nslices):
                    for c in range(ncoils):
                        image[f, s, c] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace[f, s, c])))
            
        elif len(kspace.shape) == 4:  # [nframes, nslices, ny, nx] or [nframes, ncoils, ny, nx]
            nframes = kspace.shape[0]
            image = np.zeros_like(kspace)
            
            # Process each frame
            for f in range(nframes):
                for i in range(kspace.shape[1]):
                    image[f, i] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace[f, i])))
                    
        elif len(kspace.shape) == 3:  # [ncoils, ny, nx] or [nslices, ny, nx]
            ncoils = kspace.shape[0]
            image = np.zeros_like(kspace)
            
            # Process each coil/slice
            for c in range(ncoils):
                image[c] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace[c])))
                
        else:  # [ny, nx]
            image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
            
        return image
    
    def combine_coils(self, coil_images, sensitivity_maps=None):
        """
        Combine multi-coil images into a single image.
        
        Parameters:
        -----------
        coil_images : complex ndarray
            Coil images
        sensitivity_maps : complex ndarray, optional
            Sensitivity maps for coil combination. If None, uses RSS.
            
        Returns:
        --------
        complex ndarray
            Combined image
        """
        if sensitivity_maps is None:
            # Root sum of squares combination
            if len(coil_images.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
                return np.sqrt(np.sum(np.abs(coil_images)**2, axis=2))
            elif len(coil_images.shape) == 3:  # [ncoils, ny, nx]
                return np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))
            else:
                raise ValueError(f"Unexpected shape for coil_images: {coil_images.shape}")
        else:
            # Sensitivity-weighted combination
            if len(coil_images.shape) == 5 and len(sensitivity_maps.shape) == 3:  # [nframes, nslices, ncoils, ny, nx] and [ncoils, ny, nx]
                nframes, nslices, ncoils, ny, nx = coil_images.shape
                combined = np.zeros((nframes, nslices, ny, nx), dtype=complex)
                
                # Conjugate of sensitivity maps
                sens_conj = np.conj(sensitivity_maps)
                
                # Apply combination for each frame and slice
                for f in range(nframes):
                    for s in range(nslices):
                        weighted_sum = np.sum(coil_images[f, s] * sens_conj, axis=0)
                        sens_square_sum = np.sum(np.abs(sensitivity_maps)**2, axis=0)
                        combined[f, s] = weighted_sum / (sens_square_sum + 1e-9)
                
                return combined
            
            elif len(coil_images.shape) == 3 and len(sensitivity_maps.shape) == 3:  # [ncoils, ny, nx] and [ncoils, ny, nx]
                # Conjugate of sensitivity maps
                sens_conj = np.conj(sensitivity_maps)
                
                # Apply combination
                weighted_sum = np.sum(coil_images * sens_conj, axis=0)
                sens_square_sum = np.sum(np.abs(sensitivity_maps)**2, axis=0)
                
                return weighted_sum / (sens_square_sum + 1e-9)
            
            else:
                raise ValueError(f"Incompatible shapes: coil_images {coil_images.shape}, sensitivity_maps {sensitivity_maps.shape}")
    
    # -------------------- VISUALIZATION METHODS --------------------
    
    def visualize_kspace(self, kspace, slice_idx=0, frame_idx=0, coil_idx=None, log_scale=True, cmap='viridis', figsize=(12, 10)):
        """
        Visualize k-space data.
        
        Parameters:
        -----------
        kspace : complex ndarray
            K-space data to visualize
        slice_idx : int
            Slice index to visualize
        frame_idx : int
            Frame index to visualize
        coil_idx : int or list, optional
            Coil index(es) to visualize. If None, displays RSS of all coils.
        log_scale : bool
            Whether to use log scale for visualization
        cmap : str
            Colormap to use
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Extract data to visualize
        if len(kspace.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            if coil_idx is None:
                # Display RSS of all coils
                k_display = np.sqrt(np.sum(np.abs(kspace[frame_idx, slice_idx])**2, axis=0))
                title = f"K-space (RSS) - Frame {frame_idx}, Slice {slice_idx}"
            elif isinstance(coil_idx, list):
                # Display multiple coils
                fig, axs = plt.subplots(1, len(coil_idx), figsize=figsize)
                for i, c in enumerate(coil_idx):
                    k_view = np.abs(kspace[frame_idx, slice_idx, c])
                    if log_scale:
                        k_view = np.log(k_view + 1e-9)
                    axs[i].imshow(k_view, cmap=cmap)
                    axs[i].set_title(f"Coil {c}")
                    axs[i].axis('off')
                plt.tight_layout()
                return fig
            else:
                # Display single coil
                k_display = kspace[frame_idx, slice_idx, coil_idx]
                title = f"K-space - Frame {frame_idx}, Slice {slice_idx}, Coil {coil_idx}"
        
        elif len(kspace.shape) == 4:  # [nframes, nslices, ny, nx] or [nframes, ncoils, ny, nx]
            k_display = kspace[frame_idx, slice_idx]
            title = f"K-space - Frame {frame_idx}, Index {slice_idx}"
            
        elif len(kspace.shape) == 3:  # [ncoils, ny, nx] or [nslices, ny, nx]
            if coil_idx is None:
                # Display RSS of all slices/coils
                k_display = np.sqrt(np.sum(np.abs(kspace)**2, axis=0))
                title = "K-space (RSS)"
            else:
                # Display single slice/coil
                k_display = kspace[coil_idx]
                title = f"K-space - Index {coil_idx}"
                
        else:  # [ny, nx]
            k_display = kspace
            title = "K-space"
            
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Apply log scaling if requested
        if log_scale:
            k_display = np.log(np.abs(k_display) + 1e-9)
            title += " (Log Scale)"
        else:
            k_display = np.abs(k_display)
            
        # Display image
        im = ax.imshow(k_display, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return fig
    
    def visualize_image(self, image_data, slice_idx=0, frame_idx=0, coil_idx=None, cmap='gray', figsize=(12, 10)):
        """
        Visualize image domain data.
        
        Parameters:
        -----------
        image_data : complex ndarray
            Image data to visualize
        slice_idx : int
            Slice index to visualize
        frame_idx : int
            Frame index to visualize
        coil_idx : int or list, optional
            Coil index(es) to visualize. If None, displays RSS of all coils.
        cmap : str
            Colormap to use
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Extract data to visualize
        if len(image_data.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            if coil_idx is None:
                # Display RSS of all coils
                img_display = np.sqrt(np.sum(np.abs(image_data[frame_idx, slice_idx])**2, axis=0))
                title = f"Image (RSS) - Frame {frame_idx}, Slice {slice_idx}"
            elif isinstance(coil_idx, list):
                # Display multiple coils
                fig, axs = plt.subplots(1, len(coil_idx), figsize=figsize)
                for i, c in enumerate(coil_idx):
                    axs[i].imshow(np.abs(image_data[frame_idx, slice_idx, c]), cmap=cmap)
                    axs[i].set_title(f"Coil {c}")
                    axs[i].axis('off')
                plt.tight_layout()
                return fig
            else:
                # Display single coil
                img_display = image_data[frame_idx, slice_idx, coil_idx]
                title = f"Image - Frame {frame_idx}, Slice {slice_idx}, Coil {coil_idx}"
        
        elif len(image_data.shape) == 4:  # [nframes, nslices, ny, nx] or [nframes, ncoils, ny, nx]
            img_display = image_data[frame_idx, slice_idx]
            title = f"Image - Frame {frame_idx}, Index {slice_idx}"
            
        elif len(image_data.shape) == 3:  # [ncoils, ny, nx] or [nslices, ny, nx]
            if coil_idx is None:
                # Display RSS of all slices/coils
                img_display = np.sqrt(np.sum(np.abs(image_data)**2, axis=0))
                title = "Image (RSS)"
            else:
                # Display single slice/coil
                img_display = image_data[coil_idx]
                title = f"Image - Index {coil_idx}"
                
        else:  # [ny, nx]
            img_display = image_data
            title = "Image"
            
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Display image
        im = ax.imshow(np.abs(img_display), cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return fig
    
    def visualize_comparison(self, fully_sampled, undersampled, reconstructed=None, slice_idx=0, frame_idx=0, cmap='gray', figsize=(18, 6)):
        """
        Visualize comparison between fully sampled, undersampled, and reconstructed images.
        
        Parameters:
        -----------
        fully_sampled : complex ndarray
            Fully sampled image data
        undersampled : complex ndarray
            Undersampled image data
        reconstructed : complex ndarray, optional
            Reconstructed image data
        slice_idx : int
            Slice index to visualize
        frame_idx : int
            Frame index to visualize
        cmap : str
            Colormap to use
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Determine number of subplots
        n_plots = 2 if reconstructed is None else 3
        
        # Create figure
        fig, axs = plt.subplots(1, n_plots, figsize=figsize)
        
        # Extract data to visualize
        if len(fully_sampled.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
            # Display RSS of all coils
            full_img = np.sqrt(np.sum(np.abs(fully_sampled[frame_idx, slice_idx])**2, axis=0))
            under_img = np.sqrt(np.sum(np.abs(undersampled[frame_idx, slice_idx])**2, axis=0))
            title_prefix = f"Frame {frame_idx}, Slice {slice_idx}"
        
        elif len(fully_sampled.shape) == 4:  # [nframes, nslices, ny, nx]
            full_img = fully_sampled[frame_idx, slice_idx]
            under_img = undersampled[frame_idx, slice_idx]
            title_prefix = f"Frame {frame_idx}, Slice {slice_idx}"
            
        elif len(fully_sampled.shape) == 3:  # [nslices, ny, nx]
            full_img = fully_sampled[slice_idx]
            under_img = undersampled[slice_idx]
            title_prefix = f"Slice {slice_idx}"
                
        else:  # [ny, nx]
            full_img = fully_sampled
            under_img = undersampled
            title_prefix = ""
            
        # Display fully sampled image
        axs[0].imshow(np.abs(full_img), cmap=cmap)
        axs[0].set_title(f"Fully Sampled\n{title_prefix}")
        axs[0].axis('off')
        
        # Display undersampled image
        axs[1].imshow(np.abs(under_img), cmap=cmap)
        axs[1].set_title(f"Undersampled\n{title_prefix}")
        axs[1].axis('off')
        
        # Display reconstructed image if provided
        if reconstructed is not None:
            if len(reconstructed.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
                recon_img = np.sqrt(np.sum(np.abs(reconstructed[frame_idx, slice_idx])**2, axis=0))
            elif len(reconstructed.shape) == 4:  # [nframes, nslices, ny, nx]
                recon_img = reconstructed[frame_idx, slice_idx]
            elif len(reconstructed.shape) == 3:  # [nslices, ny, nx]
                recon_img = reconstructed[slice_idx]
            else:  # [ny, nx]
                recon_img = reconstructed
                
            axs[2].imshow(np.abs(recon_img), cmap=cmap)
            axs[2].set_title(f"Reconstructed\n{title_prefix}")
            axs[2].axis('off')
            
            # Compute error
            error = np.abs(full_img - recon_img)
            max_error = np.max(error)
            mean_error = np.mean(error)
            ssim_value = self.compute_ssim(full_img, recon_img)
            psnr_value = self.compute_psnr(full_img, recon_img)
            
            plt.suptitle(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB, Mean Error: {mean_error:.4f}", fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def visualize_center_comparison(self, data_dict, slice_idx=0, frame_idx=0, cmap='gray', figsize=(20, 10)):
        """
        Visualize comparison of data from different centers.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with center IDs as keys and image data as values
        slice_idx : int
            Slice index to visualize
        frame_idx : int
            Frame index to visualize
        cmap : str
            Colormap to use
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Determine number of centers
        n_centers = len(data_dict)
        
        # Create figure
        fig, axs = plt.subplots(1, n_centers, figsize=figsize)
        
        # Display images from each center
        for i, (center_id, data) in enumerate(data_dict.items()):
            # Extract data to visualize
            if len(data.shape) == 5:  # [nframes, nslices, ncoils, ny, nx]
                img_display = np.sqrt(np.sum(np.abs(data[frame_idx, slice_idx])**2, axis=0))
            elif len(data.shape) == 4:  # [nframes, nslices, ny, nx]
                img_display = data[frame_idx, slice_idx]
            elif len(data.shape) == 3:  # [nslices, ny, nx]
                img_display = data[slice_idx]
            else:  # [ny, nx]
                img_display = data
                
            # Display image
            axs[i].imshow(np.abs(img_display), cmap=cmap)
            axs[i].set_title(f"Center: {center_id}")
            axs[i].axis('off')
        
        plt.suptitle(f"Center Comparison - Frame {frame_idx}, Slice {slice_idx}", fontsize=14)
        plt.tight_layout()
        return fig
    
    # -------------------- EVALUATION METRICS --------------------
    
    def compute_ssim(self, reference, test):
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        Parameters:
        -----------
        reference : ndarray
            Reference image
        test : ndarray
            Test image
            
        Returns:
        --------
        float
            SSIM value
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to magnitude images if complex
        if np.iscomplexobj(reference):
            reference = np.abs(reference)
        if np.iscomplexobj(test):
            test = np.abs(test)
        
        # Normalize images to [0, 1] range
        ref_norm = reference / np.max(reference)
        test_norm = test / np.max(test)
        
        # Compute SSIM
        return ssim(ref_norm, test_norm, data_range=1.0)
    
    def compute_psnr(self, reference, test):
        """
        Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
        
        Parameters:
        -----------
        reference : ndarray
            Reference image
        test : ndarray
            Test image
            
        Returns:
        --------
        float
            PSNR value in dB
        """
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Convert to magnitude images if complex
        if np.iscomplexobj(reference):
            reference = np.abs(reference)
        if np.iscomplexobj(test):
            test = np.abs(test)
        
        # Normalize images to [0, 1] range
        ref_norm = reference / np.max(reference)
        test_norm = test / np.max(test)
        
        # Compute PSNR
        return psnr(ref_norm, test_norm, data_range=1.0)
    
    def compute_nmse(self, reference, test):
        """
        Compute Normalized Mean Squared Error (NMSE) between two images.
        
        Parameters:
        -----------
        reference : ndarray
            Reference image
        test : ndarray
            Test image
            
        Returns:
        --------
        float
            NMSE value
        """
        # Convert to magnitude images if complex
        if np.iscomplexobj(reference):
            reference = np.abs(reference)
        if np.iscomplexobj(test):
            test = np.abs(test)
        
        # Compute NMSE
        error = np.sum((reference - test) ** 2)
        norm = np.sum(reference ** 2)
        
        return error / (norm + 1e-9)
    
    # -------------------- COMPLETE PREPROCESSING PIPELINE --------------------
    
    def preprocess_single_case(self, kspace_file, mask_file=None, metadata=None):
        """
        Apply complete preprocessing pipeline to a single case.
        
        Parameters:
        -----------
        kspace_file : str
            Path to k-space data file
        mask_file : str, optional
            Path to mask file
        metadata : dict, optional
            Metadata for the case
            
        Returns:
        --------
        dict
            Dictionary with preprocessed data
        """
        # Load k-space data
        kspace = self.load_kspace(kspace_file)
        
        # Extract metadata if not provided
        if metadata is None:
            metadata = self.get_metadata_from_path(kspace_file)
        
        # Normalize k-space
        kspace_norm = self.normalize_kspace(kspace, metadata)
        
        # Standardize dimensions
        kspace_std = self.standardize_dimensions(kspace_norm)
        
        # Estimate sensitivity maps
        sensitivity_maps = self.estimate_sensitivity_maps(kspace_std)
        
        # Apply data augmentation if enabled
        kspace_aug, sensitivity_maps_aug = self.augment_kspace(kspace_std, sensitivity_maps)
        
        # Apply undersampling
        if mask_file is not None:
            mask = self.load_mask(mask_file)
            undersampled_kspace, mask = self.apply_undersampling(kspace_aug, mask)
        else:
            undersampled_kspace, mask = self.apply_undersampling(kspace_aug)
        
        # Transform to image domain
        full_image = self.kspace_to_image(kspace_aug)
        under_image = self.kspace_to_image(undersampled_kspace)
        
        # Combine coils
        full_combined = self.combine_coils(full_image, sensitivity_maps_aug)
        under_combined = self.combine_coils(under_image, sensitivity_maps_aug)
        
        return {
            'kspace_orig': kspace,
            'kspace_norm': kspace_norm,
            'kspace_std': kspace_std,
            'kspace_aug': kspace_aug,
            'undersampled_kspace': undersampled_kspace,
            'mask': mask,
            'sensitivity_maps': sensitivity_maps_aug,
            'full_image': full_image,
            'under_image': under_image,
            'full_combined': full_combined,
            'under_combined': under_combined,
            'metadata': metadata
        }
    
    def preprocess_dataset(self, data_list, output_dir=None, n_samples=None, batch_size=1):
        """
        Apply preprocessing to a list of data files.
        
        Parameters:
        -----------
        data_list : list
            List of dictionaries, each containing paths to kspace and mask files
        output_dir : str, optional
            Directory to save preprocessed data
        n_samples : int, optional
            Number of samples to process (for debugging)
        batch_size : int
            Batch size for processing
            
        Returns:
        --------
        list
            List of dictionaries with preprocessed data
        """
        if n_samples is not None:
            data_list = data_list[:n_samples]
            
        results = []
        
        # Process in batches to save memory
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            
            for j, case in enumerate(tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{len(data_list)//batch_size + 1}")):
                try:
                    # Preprocess single case
                    kspace_file = case['kspace_file']
                    mask_file = case.get('mask_file', None)
                    metadata = case.get('metadata', None)
                    
                    result = self.preprocess_single_case(kspace_file, mask_file, metadata)
                    
                    # Save preprocessed data if output directory is specified
                    if output_dir is not None:
                        # Create subdirectories based on metadata
                        center_id = result['metadata']['center_id']
                        scanner_type = result['metadata']['scanner_type']
                        patient_id = result['metadata']['patient_id']
                        
                        # Create directory structure
                        case_dir = os.path.join(output_dir, center_id, scanner_type, patient_id)
                        os.makedirs(case_dir, exist_ok=True)
                        
                        # Save k-space and images
                        self.save_to_mat(result['undersampled_kspace'], 'kspace', os.path.join(case_dir, 'undersampled_kspace.mat'))
                        self.save_to_mat(result['full_combined'], 'image', os.path.join(case_dir, 'full_combined.mat'))
                        self.save_to_mat(result['mask'], 'mask', os.path.join(case_dir, 'mask.mat'))
                        self.save_to_mat(result['sensitivity_maps'], 'sensitivity_maps', os.path.join(case_dir, 'sensitivity_maps.mat'))
                        
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing case {kspace_file}: {e}")
        
        return results
    
    def create_dataset_list(self, root_dir, pattern='**/*.mat', exclude_patterns=None):
        """
        Create a list of data files for preprocessing.
        
        Parameters:
        -----------
        root_dir : str
            Root directory containing data
        pattern : str
            Glob pattern for finding files
        exclude_patterns : list, optional
            List of patterns to exclude
            
        Returns:
        --------
        list
            List of dictionaries, each containing paths to kspace and mask files
        """
        import glob
        
        # Find all k-space files
        kspace_files = glob.glob(os.path.join(root_dir, pattern), recursive=True)
        
        # Filter out unwanted files
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                kspace_files = [f for f in kspace_files if pattern not in f]
        
        # Create list of data files
        data_list = []
        
        for kspace_file in kspace_files:
            # Find corresponding mask file
            mask_files = glob.glob(os.path.join(os.path.dirname(kspace_file), '*_mask_*.mat'))
            mask_file = mask_files[0] if mask_files else None
            
            # Extract metadata
            metadata = self.get_metadata_from_path(kspace_file)
            
            data_list.append({
                'kspace_file': kspace_file,
                'mask_file': mask_file,
                'metadata': metadata
            })
        
        return data_list


class CMRDataLoader:
    """
    Data loader for CMR reconstruction.
    Handles loading and preprocessing of data for training and evaluation.
    """
    
    def __init__(self, data_dir, preprocessor=None, batch_size=1, shuffle=True):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        preprocessor : CMRPreprocessor, optional
            Preprocessor for the data
        batch_size : int
            Batch size for loading data
        shuffle : bool
            Whether to shuffle the data
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = CMRPreprocessor()
        else:
            self.preprocessor = preprocessor
        
        # Create list of data files
        self.data_list = self.preprocessor.create_dataset_list(data_dir)
        
        # Split data into training and validation sets
        self.train_list, self.val_list = self._split_train_val()
    
    def _split_train_val(self, val_ratio=0.2):
        """
        Split data into training and validation sets.
        
        Parameters:
        -----------
        val_ratio : float
            Ratio of validation data
            
        Returns:
        --------
        tuple
            (train_list, val_list)
        """
        # Group data by center
        centers = {}
        for case in self.data_list:
            center_id = case['metadata']['center_id']
            if center_id not in centers:
                centers[center_id] = []
            centers[center_id].append(case)
        
        # Split data for each center
        train_list, val_list = [], []
        
        for center_id, cases in centers.items():
            # Shuffle cases
            import random
            random.shuffle(cases)
            
            # Split cases
            split_idx = int(len(cases) * (1 - val_ratio))
            train_list.extend(cases[:split_idx])
            val_list.extend(cases[split_idx:])
        
        return train_list, val_list
    
    def get_train_loader(self):
        """
        Get data loader for training set.
        
        Returns:
        --------
        generator
            Data loader for training set
        """
        return self._get_loader(self.train_list)
    
    def get_val_loader(self):
        """
        Get data loader for validation set.
        
        Returns:
        --------
        generator
            Data loader for validation set
        """
        return self._get_loader(self.val_list, shuffle=False)
    
    def _get_loader(self, data_list, shuffle=None):
        """
        Get data loader for a list of data files.
        
        Parameters:
        -----------
        data_list : list
            List of data files
        shuffle : bool, optional
            Whether to shuffle the data
            
        Returns:
        --------
        generator
            Data loader
        """
        if shuffle is None:
            shuffle = self.shuffle
            
        # Create copy of data list
        data = data_list.copy()
        
        # Shuffle data if requested
        if shuffle:
            import random
            random.shuffle(data)
        
        # Yield batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            
            # Preprocess batch
            results = self.preprocessor.preprocess_dataset(batch)
            
            # Yield batch
            yield results


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CMRPreprocessor(config={
        'norm_method': 'center_aware',
        'augmentation': True,
        'standardize_dims': True,
        'verbose': True
    })
    
    # Load and preprocess a single file
    kspace_file = "path/to/kspace.mat"
    mask_file = "path/to/mask.mat"
    
    # Process single case
    result = preprocessor.preprocess_single_case(kspace_file, mask_file)
    
    # Visualize results
    preprocessor.visualize_comparison(
        result['full_combined'],
        result['under_combined'],
        slice_idx=0,
        frame_idx=0
    )
    plt.show()
    
    # Create dataset list and preprocess
    data_list = preprocessor.create_dataset_list("path/to/data")
    results = preprocessor.preprocess_dataset(data_list[:10], output_dir="path/to/output")
    
    # Create data loader
    loader = CMRDataLoader("path/to/data", preprocessor)
    
    # Get training and validation loaders
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    
    # Train model using these loaders
    for batch in train_loader:
        # Process batch
        pass