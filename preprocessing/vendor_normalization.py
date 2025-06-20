import numpy as np
import os
import re
import scipy.io as sio
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def extract_center_kspace(kspace, center_size=(24, 24)):
    """
    Extract the central calibration region of k-space.
    
    Parameters:
    -----------
    kspace : ndarray
        Complex k-space data with shape [nx, ny, nc, nz, nt] or [nx, ny, nc, nz]
        nx, ny: matrix size in x and y directions
        nc: number of coils
        nz: number of slices
        nt: number of temporal frames (if applicable)
    center_size : tuple
        Size of the central region to extract (kx, ky)
        
    Returns:
    --------
    center_kspace : ndarray
        Central region of k-space with same dimensionality
    """
    # Get dimensions
    dims = kspace.shape
    nx, ny = dims[0], dims[1]
    
    # Calculate start and end indices for center extraction
    x_start = nx//2 - center_size[0]//2
    x_end = x_start + center_size[0]
    y_start = ny//2 - center_size[1]//2
    y_end = y_start + center_size[1]
    
    # Handle boundary cases
    x_start = max(0, x_start)
    x_end = min(nx, x_end)
    y_start = max(0, y_start)
    y_end = min(ny, y_end)
    
    # Extract center
    if len(dims) == 5:  # Has temporal dimension
        center_kspace = kspace[x_start:x_end, y_start:y_end, :, :, :]
    elif len(dims) == 4:  # No temporal dimension
        center_kspace = kspace[x_start:x_end, y_start:y_end, :, :]
    else:
        raise ValueError(f"Unexpected k-space dimensions: {dims}")
        
    return center_kspace

def siemens_noise_decorrelation(kspace):
    """
    Apply Siemens-specific noise decorrelation.
    
    Parameters:
    -----------
    kspace : ndarray
        Complex k-space data [nx, ny, nc, nz, nt] or [nx, ny, nc, nz]
        
    Returns:
    --------
    decorrelated_kspace : ndarray
        Noise-decorrelated k-space data
    """
    # Get dimensions
    dims = kspace.shape
    nc = dims[2]  # Number of coils
    
    # Extract noise region (often in corners of k-space)
    # For Siemens, we can use the corner regions which typically contain mostly noise
    nx, ny = dims[0], dims[1]
    corner_size = min(nx, ny) // 8
    
    # Extract top-left corner for noise estimation
    if len(dims) == 5:
        noise_samples = kspace[:corner_size, :corner_size, :, 0, 0].reshape(-1, nc)
    else:
        noise_samples = kspace[:corner_size, :corner_size, :, 0].reshape(-1, nc)
    
    # Calculate noise covariance matrix
    noise_cov = np.cov(noise_samples.T)
    
    # Add small regularization to ensure matrix is invertible
    noise_cov = noise_cov + np.eye(nc) * 1e-6
    
    # Calculate whitening matrix
    whitening_matrix = np.linalg.cholesky(np.linalg.inv(noise_cov))
    
    # Apply whitening transformation
    if len(dims) == 5:  # Has temporal dimension
        decorrelated_kspace = np.zeros_like(kspace, dtype=complex)
        for z in range(dims[3]):
            for t in range(dims[4]):
                for x in range(nx):
                    for y in range(ny):
                        decorrelated_kspace[x, y, :, z, t] = whitening_matrix @ kspace[x, y, :, z, t]
    else:  # No temporal dimension
        decorrelated_kspace = np.zeros_like(kspace, dtype=complex)
        for z in range(dims[3]):
            for x in range(nx):
                for y in range(ny):
                    decorrelated_kspace[x, y, :, z] = whitening_matrix @ kspace[x, y, :, z]
    
    return decorrelated_kspace

def uih_noise_decorrelation(kspace):
    """
    Apply UIH-specific noise decorrelation.
    
    Parameters:
    -----------
    kspace : ndarray
        Complex k-space data [nx, ny, nc, nz, nt] or [nx, ny, nc, nz]
        
    Returns:
    --------
    decorrelated_kspace : ndarray
        Noise-decorrelated k-space data
    """
    # UIH scanners may have different noise characteristics
    
    # Get dimensions
    dims = kspace.shape
    nc = dims[2]  # Number of coils
    
    # For UIH, sometimes the noise characteristics are better estimated from
    # the high-frequency regions of k-space
    nx, ny = dims[0], dims[1]
    edge_width = min(nx, ny) // 10
    
    # Extract high-frequency samples for noise estimation
    if len(dims) == 5:
        high_freq_samples = []
        for z in range(min(dims[3], 2)):  # Use up to 2 slices for efficiency
            for t in range(min(dims[4], 2)):  # Use up to 2 timepoints for efficiency
                for c in range(nc):
                    high_freq_samples.extend(kspace[:edge_width, :, c, z, t].flatten())
                    high_freq_samples.extend(kspace[-edge_width:, :, c, z, t].flatten())
                    high_freq_samples.extend(kspace[:, :edge_width, c, z, t].flatten())
                    high_freq_samples.extend(kspace[:, -edge_width:, c, z, t].flatten())
        high_freq_samples = np.array(high_freq_samples).reshape(-1, nc)
    else:
        high_freq_samples = []
        for z in range(min(dims[3], 2)):  # Use up to 2 slices for efficiency
            for c in range(nc):
                high_freq_samples.extend(kspace[:edge_width, :, c, z].flatten())
                high_freq_samples.extend(kspace[-edge_width:, :, c, z].flatten())
                high_freq_samples.extend(kspace[:, :edge_width, c, z].flatten())
                high_freq_samples.extend(kspace[:, -edge_width:, c, z].flatten())
        high_freq_samples = np.array(high_freq_samples).reshape(-1, nc)
    
    # Calculate noise covariance matrix
    noise_cov = np.cov(high_freq_samples.T)
    
    # Add small regularization to ensure matrix is invertible
    noise_cov = noise_cov + np.eye(nc) * 1e-5
    
    # Calculate whitening matrix (UIH might need different regularization)
    whitening_matrix = np.linalg.cholesky(np.linalg.inv(noise_cov))
    
    # Apply whitening transformation
    if len(dims) == 5:  # Has temporal dimension
        decorrelated_kspace = np.zeros_like(kspace, dtype=complex)
        for z in range(dims[3]):
            for t in range(dims[4]):
                for x in range(nx):
                    for y in range(ny):
                        decorrelated_kspace[x, y, :, z, t] = whitening_matrix @ kspace[x, y, :, z, t]
    else:  # No temporal dimension
        decorrelated_kspace = np.zeros_like(kspace, dtype=complex)
        for z in range(dims[3]):
            for x in range(nx):
                for y in range(ny):
                    decorrelated_kspace[x, y, :, z] = whitening_matrix @ kspace[x, y, :, z]
    
    return decorrelated_kspace

def ge_phase_correction(kspace):
    """
    Apply GE-specific phase correction.
    
    Parameters:
    -----------
    kspace : ndarray
        Complex k-space data [nx, ny, nc, nz, nt] or [nx, ny, nc, nz]
        
    Returns:
    --------
    corrected_kspace : ndarray
        Phase-corrected k-space data
    """
    # GE scanners often have systematic phase errors that need correction
    # This is a simplified version based on typical GE characteristics
    
    # Get dimensions
    dims = kspace.shape
    
    # Convert to image domain
    image = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    # Extract phase from central region
    nx, ny = dims[0], dims[1]
    center_size = min(nx, ny) // 4
    x_center = nx // 2
    y_center = ny // 2
    
    if len(dims) == 5:  # Has temporal dimension
        center_phase = np.angle(image[x_center-center_size//2:x_center+center_size//2, 
                                     y_center-center_size//2:y_center+center_size//2, 
                                     :, :, :])
        # Smooth the phase in the central region
        smoothed_phase = np.zeros_like(center_phase)
        for c in range(dims[2]):
            for z in range(dims[3]):
                for t in range(dims[4]):
                    smoothed_phase[:, :, c, z, t] = gaussian_filter(center_phase[:, :, c, z, t], sigma=2)
    else:  # No temporal dimension
        center_phase = np.angle(image[x_center-center_size//2:x_center+center_size//2, 
                                     y_center-center_size//2:y_center+center_size//2, 
                                     :, :])
        # Smooth the phase in the central region
        smoothed_phase = np.zeros_like(center_phase)
        for c in range(dims[2]):
            for z in range(dims[3]):
                smoothed_phase[:, :, c, z] = gaussian_filter(center_phase[:, :, c, z], sigma=2)
    
    # Create phase correction map by extrapolating from central region
    # This is a simplified approach - real GE phase correction would be more sophisticated
    phase_corr_map = np.zeros(dims, dtype=complex)
    
    if len(dims) == 5:
        for c in range(dims[2]):
            for z in range(dims[3]):
                for t in range(dims[4]):
                    # Create 2D phase map for this coil/slice/timepoint
                    full_phase_map = np.zeros((nx, ny))
                    full_phase_map[x_center-center_size//2:x_center+center_size//2, 
                                  y_center-center_size//2:y_center+center_size//2] = smoothed_phase[:, :, c, z, t]
                    
                    # Extrapolate to full FOV using Gaussian blur
                    full_phase_map = gaussian_filter(full_phase_map, sigma=center_size//2)
                    
                    # Create complex phase correction map
                    phase_corr_map[:, :, c, z, t] = np.exp(-1j * full_phase_map)
    else:
        for c in range(dims[2]):
            for z in range(dims[3]):
                # Create 2D phase map for this coil/slice
                full_phase_map = np.zeros((nx, ny))
                full_phase_map[x_center-center_size//2:x_center+center_size//2, 
                              y_center-center_size//2:y_center+center_size//2] = smoothed_phase[:, :, c, z]
                
                # Extrapolate to full FOV using Gaussian blur
                full_phase_map = gaussian_filter(full_phase_map, sigma=center_size//2)
                
                # Create complex phase correction map
                phase_corr_map[:, :, c, z] = np.exp(-1j * full_phase_map)
    
    # Apply phase correction to image domain
    corrected_image = image * phase_corr_map
    
    # Convert back to k-space
    corrected_kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(corrected_image, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    return corrected_kspace

def extract_vendor_info(filepath):
    """
    Extract vendor information from file path.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
        
    Returns:
    --------
    vendor_info : dict
        Dictionary containing vendor, field_strength, and model
    """
    vendor_info = {}
    
    # Extract vendor and model from path
    # Example path: /ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/Center002/UIH_30T_umr880/P003/
    path_parts = filepath.split('/')
    
    # Find scanner information
    scanner_part = None
    for part in path_parts:
        if any(vendor in part for vendor in ['UIH', 'Siemens', 'GE']):
            scanner_part = part
            break
    
    if scanner_part is None:
        # Default values if scanner info cannot be extracted
        vendor_info['vendor'] = 'unknown'
        vendor_info['field_strength'] = 3.0
        vendor_info['model'] = 'unknown'
        return vendor_info
    
    # Extract vendor
    if 'UIH' in scanner_part:
        vendor_info['vendor'] = 'UIH'
    elif 'Siemens' in scanner_part:
        vendor_info['vendor'] = 'Siemens'
    elif 'GE' in scanner_part:
        vendor_info['vendor'] = 'GE'
    else:
        vendor_info['vendor'] = 'unknown'
    
    # Extract field strength
    if '15T' in scanner_part:
        vendor_info['field_strength'] = 1.5
    elif '30T' in scanner_part:
        vendor_info['field_strength'] = 3.0
    elif '50T' in scanner_part:
        vendor_info['field_strength'] = 5.0
    else:
        # Default to 3T if not found
        vendor_info['field_strength'] = 3.0
    
    # Extract model
    model_match = re.search(r'(?:umr|Prisma|Vida|Aera|Avanto|Sola|CIMA\.X|voyager)\d*', scanner_part, re.IGNORECASE)
    if model_match:
        vendor_info['model'] = model_match.group(0)
    else:
        vendor_info['model'] = 'unknown'
    
    return vendor_info

def detect_and_reorder_dimensions(kspace):
    """
    Detect and reorder dimensions to standard [nx, ny, nc, nz, nt] format.
    
    Parameters:
    -----------
    kspace : ndarray
        K-space data with arbitrary dimension ordering
        
    Returns:
    --------
    reordered_kspace : ndarray
        K-space data with dimensions ordered as [nx, ny, nc, nz, nt]
    """
    shape = kspace.shape
    
    if len(shape) == 5:
        # Check for UIH specific pattern: [nt, nz, nc, nx, ny]
        if shape[0] < 30 and shape[1] < 20 and shape[2] < 32 and shape[3] > 100 and shape[4] > 100:
            print(f"Detected UIH dimension ordering [nt, nz, nc, nx, ny]: {shape}")
            print(f"Transposing to [nx, ny, nc, nz, nt] format")
            return np.transpose(kspace, (3, 4, 2, 1, 0))
        
        # Try to identify dimensions based on typical size ranges
        nx_idx, ny_idx, nc_idx, nz_idx, nt_idx = None, None, None, None, None
        
        # Find the two largest dimensions (likely nx and ny)
        sizes = [(i, s) for i, s in enumerate(shape)]
        sizes.sort(key=lambda x: x[1], reverse=True)
        
        # The two largest dimensions are likely nx and ny
        if sizes[0][1] > 100 and sizes[1][1] > 100:
            ny_idx, nx_idx = sizes[0][0], sizes[1][0]
        
        # Find the dimension most likely to be nc (usually 8-32 channels)
        for i, s in enumerate(shape):
            if 8 <= s <= 32 and i != nx_idx and i != ny_idx:
                nc_idx = i
                break
        
        # Find the dimension most likely to be nz (usually 1-20 slices)
        for i, s in enumerate(shape):
            if 1 <= s <= 20 and i != nx_idx and i != ny_idx and i != nc_idx:
                nz_idx = i
                break
        
        # The remaining dimension is likely nt
        for i in range(5):
            if i != nx_idx and i != ny_idx and i != nc_idx and i != nz_idx:
                nt_idx = i
                break
        
        if all(idx is not None for idx in [nx_idx, ny_idx, nc_idx, nz_idx, nt_idx]):
            print(f"Detected dimension ordering: nx={shape[nx_idx]}, ny={shape[ny_idx]}, "
                  f"nc={shape[nc_idx]}, nz={shape[nz_idx]}, nt={shape[nt_idx]}")
            
            # Reorder dimensions to [nx, ny, nc, nz, nt]
            return np.transpose(kspace, (nx_idx, ny_idx, nc_idx, nz_idx, nt_idx))
    
    # If we couldn't determine a better ordering, return as is
    return kspace

def normalize_kspace_by_vendor(kspace, filepath=None, vendor=None, field_strength=None, model=None):
    """
    Normalize k-space data according to vendor-specific characteristics.
    
    Parameters:
    -----------
    kspace : ndarray
        Raw k-space data [nx, ny, nc, nz, nt] (complex) or [nx, ny, nc, nz]
    filepath : str, optional
        Path to the data file, used to extract vendor information if not provided
    vendor : str, optional
        Scanner manufacturer (Siemens, UIH, GE)
    field_strength : float, optional
        Field strength in Tesla (1.5T, 3.0T, 5.0T)
    model : str, optional
        Scanner model (e.g., Prisma, umr780)
        
    Returns:
    --------
    normalized_kspace : ndarray
        Normalized k-space data
    """
    # Extract vendor information if not provided
    if (vendor is None or field_strength is None or model is None) and filepath is not None:
        vendor_info = extract_vendor_info(filepath)
        vendor = vendor_info['vendor']
        field_strength = vendor_info['field_strength']
        model = vendor_info['model']
    elif vendor is None or field_strength is None or model is None:
        raise ValueError("Either filepath or vendor, field_strength, and model must be provided")
    
    print(f"Processing data from {vendor} {field_strength}T {model} scanner")
    
    # Extract central region for statistics (ACS region - auto-calibration signal)
    center_kspace = extract_center_kspace(kspace, center_size=(24, 24))
    
    # Calculate appropriate scale factor based on vendor and field strength
    if vendor == "Siemens":
        # Siemens scanners typically have higher SNR but more structured noise
        if field_strength == 1.5:
            # 1.5T specific normalization (lower SNR)
            scale_factor = np.mean(np.abs(center_kspace)) * 1.2
            print(f"Siemens 1.5T scale factor: {scale_factor}")
        elif field_strength == 3.0:
            # 3.0T has higher signal intensity
            scale_factor = np.mean(np.abs(center_kspace))
            print(f"Siemens 3.0T scale factor: {scale_factor}")
        else:
            # Default scaling
            scale_factor = np.mean(np.abs(center_kspace))
            print(f"Siemens {field_strength}T scale factor: {scale_factor}")
            
        # Siemens-specific noise decorrelation
        if model.lower() in ["prisma", "vida", "cima.x"]:
            print(f"Applying Siemens {model} noise decorrelation")
            kspace = siemens_noise_decorrelation(kspace)
            
    elif vendor == "UIH":
        # UIH scanners have different scaling conventions
        scale_factor = np.mean(np.abs(center_kspace)) * 1.5
        print(f"UIH {field_strength}T scale factor: {scale_factor}")
        
        # UIH-specific noise handling (their noise characteristics differ)
        if "umr" in model.lower():
            print(f"Applying UIH {model} noise decorrelation")
            kspace = uih_noise_decorrelation(kspace)
            
    elif vendor == "GE":
        # GE has unique preprocessing needs
        scale_factor = np.mean(np.abs(center_kspace)) * 1.3
        print(f"GE {field_strength}T scale factor: {scale_factor}")
        
        # GE-specific phase correction
        print(f"Applying GE phase correction")
        kspace = ge_phase_correction(kspace)
    
    else:
        # Generic normalization for unknown vendors
        scale_factor = np.mean(np.abs(center_kspace))
        print(f"Unknown vendor, using generic scale factor: {scale_factor}")
    
    # Prevent division by zero
    if scale_factor < 1e-8:
        scale_factor = 1.0
        print("Warning: Very small scale factor detected, using default value of 1.0")
    
    # Apply normalization
    normalized_kspace = kspace / scale_factor
    
    # Compute and print statistics before and after normalization
    orig_mean = np.mean(np.abs(kspace))
    orig_std = np.std(np.abs(kspace))
    norm_mean = np.mean(np.abs(normalized_kspace))
    norm_std = np.std(np.abs(normalized_kspace))
    
    print(f"Original k-space stats: mean={orig_mean:.4e}, std={orig_std:.4e}")
    print(f"Normalized k-space stats: mean={norm_mean:.4e}, std={norm_std:.4e}")
    
    return normalized_kspace

def load_and_normalize_kspace(file_path, save_original=True):
    """
    Load and normalize k-space data from a .mat file.
    Handles both older MATLAB formats and newer HDF5-based v7.3 format.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing k-space data
    save_original : bool, optional
        Whether to return the original k-space data as well
        
    Returns:
    --------
    If save_original is True:
        (original_kspace, normalized_kspace) : tuple
            Tuple containing both the original and normalized k-space data
    Otherwise:
        normalized_kspace : ndarray
            Normalized k-space data
    """
    try:
        # First try loading with scipy.io.loadmat (for older MATLAB formats)
        try:
            mat_data = sio.loadmat(file_path)
            print(f"Loaded {file_path} using scipy.io.loadmat")
            
            # Find the k-space data - usually the largest array in the file
            kspace = None
            max_size = 0
            for key in mat_data.keys():
                if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > max_size:
                    kspace = mat_data[key]
                    max_size = kspace.size
            
            if kspace is None:
                raise ValueError(f"Could not find k-space data in {file_path}")
            
            # Check if k-space is complex
            if not np.iscomplexobj(kspace):
                if kspace.shape[-1] == 2:  # Real and imaginary parts in last dimension
                    kspace = kspace[..., 0] + 1j * kspace[..., 1]
                else:
                    raise ValueError(f"K-space data in {file_path} is not complex and does not have separate real/imag channels")
        
        except NotImplementedError:
            # If scipy.io.loadmat fails, try h5py for MATLAB v7.3 files
            print(f"Loading {file_path} using h5py (MATLAB v7.3 format)")
            with h5py.File(file_path, 'r') as f:
                # Print all keys at the root level to help identify the right dataset
                print(f"Available keys in {file_path}: {list(f.keys())}")
                
                # First look for common k-space variable names
                kspace_var_names = ['kspace', 'kSpace', 'kdata', 'rawdata', 'data']
                kspace_data_key = None
                
                for key in f.keys():
                    if key in kspace_var_names:
                        kspace_data_key = key
                        break
                
                # If not found, look for the largest dataset
                if kspace_data_key is None:
                    max_size = 0
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            if f[key].size > max_size:
                                max_size = f[key].size
                                kspace_data_key = key
                
                if kspace_data_key is None:
                    # Try to find nested datasets
                    for key in f.keys():
                        if isinstance(f[key], h5py.Group):
                            for subkey in f[key].keys():
                                if isinstance(f[key][subkey], h5py.Dataset) and f[key][subkey].size > max_size:
                                    max_size = f[key][subkey].size
                                    kspace_data_key = f"{key}/{subkey}"
                
                if kspace_data_key is None:
                    raise ValueError(f"Could not find k-space data in {file_path}")
                
                print(f"Using dataset '{kspace_data_key}' from {file_path}")
                
                # Read the dataset
                if '/' in kspace_data_key:
                    parts = kspace_data_key.split('/')
                    dataset = f[parts[0]][parts[1]]
                else:
                    dataset = f[kspace_data_key]
                
                # Print shape and type to help with debugging
                print(f"Dataset shape: {dataset.shape}, dtype: {dataset.dtype}")
                
                # Load the dataset
                kspace_data = dataset[()]
                
                # Convert to complex if needed
                if np.iscomplexobj(kspace_data):
                    kspace = kspace_data
                elif hasattr(kspace_data, 'dtype') and kspace_data.dtype.names is not None:
                    if 'real' in kspace_data.dtype.names and 'imag' in kspace_data.dtype.names:
                        print("Converting from structured array with real/imag fields")
                        kspace = kspace_data['real'] + 1j * kspace_data['imag']
                    else:
                        print("Warning: Unknown complex data format. Using real part only.")
                        kspace = kspace_data.astype(complex)
                elif len(dataset.shape) > 0 and dataset.shape[-1] == 2:
                    print("Converting from real/imag in last dimension")
                    kspace = kspace_data[..., 0] + 1j * kspace_data[..., 1]
                else:
                    print("Warning: Could not determine complex format. Converting to complex data type.")
                    kspace = kspace_data.astype(complex)
        
        # Print shape information for debugging
        print(f"Loaded k-space data shape: {kspace.shape}")
        
        # Check if we need to reorder dimensions
        kspace = detect_and_reorder_dimensions(kspace)
        print(f"After reordering: k-space data shape: {kspace.shape}")
        
        # Save the original k-space before normalization
        original_kspace = kspace.copy()
        
        # Normalize k-space based on vendor
        normalized_kspace = normalize_kspace_by_vendor(kspace, filepath=file_path)
        
        # Return based on save_original flag
        if save_original:
            return original_kspace, normalized_kspace
        else:
            return normalized_kspace
        
    except Exception as e:
        print(f"Error loading and normalizing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None if not save_original else (None, None)

def save_kspace_to_mat(kspace, save_path):
    """
    Save k-space data to a .mat file.
    
    Parameters:
    -----------
    kspace : ndarray
        K-space data to save
    save_path : str
        Path to save the .mat file
    """
    try:
        # Create a dictionary with the k-space data
        data_dict = {'kspace': kspace}
        
        # Save to .mat file
        sio.savemat(save_path, data_dict)
        print(f"Saved k-space data to {save_path}")
        return True
    except Exception as e:
        print(f"Error saving k-space data to {save_path}: {e}")
        return False

def visualize_kspace(kspace, title="K-space Magnitude", slice_idx=0, coil_idx=0, time_idx=0, save_path=None):
    """
    Visualize k-space magnitude.
    
    Parameters:
    -----------
    kspace : ndarray
        K-space data to visualize
    title : str
        Plot title
    slice_idx : int
        Slice index to visualize
    coil_idx : int
        Coil index to visualize
    time_idx : int
        Time index to visualize (for cine data)
    save_path : str, optional
        Path to save the figure instead of displaying it
    """
    plt.figure(figsize=(10, 8))
    
    if len(kspace.shape) == 5:  # Has temporal dimension
        k_mag = np.log(np.abs(kspace[:, :, coil_idx, slice_idx, time_idx]) + 1e-8)
    else:  # No temporal dimension
        k_mag = np.log(np.abs(kspace[:, :, coil_idx, slice_idx]) + 1e-8)
        
    plt.imshow(k_mag, cmap='viridis')
    plt.colorbar(label='Log Magnitude')
    plt.title(f"{title} - Slice {slice_idx}, Coil {coil_idx}" + 
              (f", Time {time_idx}" if len(kspace.shape) == 5 else ""))
    plt.xlabel('ky')
    plt.ylabel('kx')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display figure: {e}")
            print("Use save_path parameter to save the figure instead.")

# Example usage
if __name__ == "__main__":
    # Example file path (modify to match your actual data structure)
    example_file = "/mnt/f/CMRxRecon2025/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/Center002/UIH_30T_umr880/P003/cine_sax.mat"
    output_dir = "preprocessed_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and normalize k-space data, keeping the original
    original_kspace, normalized_kspace = load_and_normalize_kspace(example_file, save_original=True)
    
    if original_kspace is not None and normalized_kspace is not None:
        # Save the original k-space data
        original_save_path = os.path.join(output_dir, "original_kspace.mat")
        save_kspace_to_mat(original_kspace, original_save_path)
        
        # Save the normalized k-space data
        normalized_save_path = os.path.join(output_dir, "normalized_kspace.mat")
        save_kspace_to_mat(normalized_kspace, normalized_save_path)
        
        # Visualize original k-space
        original_viz_path = os.path.join(output_dir, "original_kspace.png")
        visualize_kspace(original_kspace, title="Original K-space Magnitude", save_path=original_viz_path)
        
        # Visualize normalized k-space
        normalized_viz_path = os.path.join(output_dir, "normalized_kspace.png")
        visualize_kspace(normalized_kspace, title="Normalized K-space Magnitude", save_path=normalized_viz_path)