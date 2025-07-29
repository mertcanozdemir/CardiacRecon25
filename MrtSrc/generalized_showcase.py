import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
import fastmri
from fastmri.data import transforms as T


class CMRReconUtils:
    """Utility class for CMRxRecon data processing and visualization."""

    @staticmethod
    def load_kspace(file_path):
        """Load k-space data from a .mat file.
        
        Args:
            file_path (str): Path to the .mat file
            
        Returns:
            tuple: (kspace_data, data_shape, is_multi_coil)
        """
        try:
            hf = h5py.File(file_path)
            print('Keys:', list(hf.keys()))
            
            # Load the kspace data
            kspace_data = hf['kspace']
            complex_data = kspace_data["real"] + 1j * kspace_data["imag"]
            
            # Get shape information
            data_shape = complex_data.shape
            
            # Determine if this is multi-coil data
            is_multi_coil = len(data_shape) > 3
            
            print(f"Data shape: {data_shape}")
            if is_multi_coil:
                print(f"Multi-coil data with {data_shape[2]} coils")
            else:
                print("Single-coil data")
                
            return complex_data, data_shape, is_multi_coil
            
        except Exception as e:
            print(f"Error loading k-space data: {e}")
            return None, None, None

    @staticmethod
    def load_mask(mask_path):
        """Load mask data from a .mat file.
        
        Args:
            mask_path (str): Path to the mask .mat file
            
        Returns:
            numpy.ndarray: Mask data
        """
        try:
            mask_file = h5py.File(mask_path)
            print('Mask keys:', list(mask_file.keys()))
            
            # The key might be 'mask' or something else
            mask_key = 'mask' if 'mask' in mask_file else list(mask_file.keys())[0]
            mask = mask_file[mask_key]
            print(f"Mask shape: {mask.shape}")
            
            return mask
        except Exception as e:
            print(f"Error loading mask: {e}")
            return None

    @staticmethod
    def apply_mask(kspace, mask, frame_idx=0, slice_idx=0):
        """Apply a mask to k-space data to simulate undersampling.
        
        Args:
            kspace (numpy.ndarray): K-space data
            mask (numpy.ndarray): Mask data
            frame_idx (int): Frame index to use
            slice_idx (int): Slice index to use
            
        Returns:
            numpy.ndarray: Undersampled k-space data
        """
        # Handle different shapes for multi-coil vs single-coil
        if len(kspace.shape) == 5:  # Multi-coil: (nframe, nslice, ncoil, ny, nx)
            slice_kspace = kspace[frame_idx, slice_idx]
            
            # Check mask dimensions
            if len(mask.shape) == 3:  # (nframe, ny, nx)
                return slice_kspace * mask[frame_idx, :, :]
            else:  # (ny, nx)
                return slice_kspace * mask[:, :]
                
        elif len(kspace.shape) == 4:  # Single-coil: (nframe, nslice, ny, nx)
            slice_kspace = kspace[frame_idx, slice_idx]
            
            # Check mask dimensions
            if len(mask.shape) == 3:  # (nframe, ny, nx)
                return slice_kspace * mask[frame_idx, :, :]
            else:  # (ny, nx)
                return slice_kspace * mask[:, :]
        else:
            print("Unsupported k-space shape")
            return None

    @staticmethod
    def reconstruct_image(kspace, is_multi_coil=True):
        """Reconstruct image from k-space data.
        
        Args:
            kspace (numpy.ndarray): K-space data
            is_multi_coil (bool): Whether this is multi-coil data
            
        Returns:
            numpy.ndarray: Reconstructed image
        """
        kspace_tensor = T.to_tensor(kspace)
        image_complex = fastmri.ifft2c(kspace_tensor)
        image_abs = fastmri.complex_abs(image_complex)
        
        if is_multi_coil:
            image_rss = fastmri.rss(image_abs, dim=0)
            return image_rss.numpy()
        else:
            return image_abs.numpy()

    @staticmethod
    def save_to_mat(data, var_name, filepath):
        """Save numpy array to .mat file.
        
        Args:
            data (numpy.ndarray): Data to save
            var_name (str): Variable name in the .mat file
            filepath (str): Path to save the .mat file
        """
        savedict = {}
        savedict[var_name] = data
        print(f"Saving {var_name} to {filepath}...")
        hdf5storage.savemat(filepath, savedict)
        print("Done.")

    @staticmethod
    def visualize_kspace(kspace, coil_indices=None, log_scale=True, vmax=0.0001):
        """Visualize k-space data.
        
        Args:
            kspace (numpy.ndarray): K-space data
            coil_indices (list): List of coil indices to visualize (for multi-coil)
            log_scale (bool): Whether to use log scale for visualization
            vmax (float): Maximum value for visualization
        """
        # Default to first 3 coils if not specified
        if coil_indices is None and len(kspace.shape) > 2:
            coil_indices = [0, min(1, kspace.shape[0]-1), min(2, kspace.shape[0]-1)]
        
        # Handle different shapes
        if len(kspace.shape) > 2:  # Multi-coil
            fig = plt.figure(figsize=(15, 5))
            for i, num in enumerate(coil_indices):
                plt.subplot(1, len(coil_indices), i + 1)
                if log_scale:
                    plt.imshow(np.log(np.abs(kspace[num]) + 1e-9), cmap='gray', vmax=vmax)
                else:
                    plt.imshow(np.abs(kspace[num]), cmap='gray', vmax=vmax)
                plt.title(f'Coil {num}')
        else:  # Single-coil
            plt.figure(figsize=(5, 5))
            if log_scale:
                plt.imshow(np.log(np.abs(kspace) + 1e-9), cmap='gray', vmax=vmax)
            else:
                plt.imshow(np.abs(kspace), cmap='gray', vmax=vmax)
            plt.title('K-space')
        plt.colorbar()
        plt.show()

    @staticmethod
    def visualize_image(image, title="Reconstructed Image", vmax=0.0015):
        """Visualize reconstructed image.
        
        Args:
            image (numpy.ndarray): Reconstructed image
            title (str): Title for the visualization
            vmax (float): Maximum value for visualization
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(image), cmap='gray', vmax=vmax)
        plt.colorbar()
        plt.title(title)
        plt.show()

    @staticmethod
    def compare_images(full_image, under_image, titles=["Fully-sampled", "Undersampled"], vmax=0.0015):
        """Compare fully-sampled and undersampled images.
        
        Args:
            full_image (numpy.ndarray): Fully-sampled image
            under_image (numpy.ndarray): Undersampled image
            titles (list): Titles for the visualizations
            vmax (float): Maximum value for visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        im1 = axes[0].imshow(np.abs(full_image), cmap='gray', vmax=vmax)
        axes[0].set_title(titles[0])
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(np.abs(under_image), cmap='gray', vmax=vmax)
        axes[1].set_title(titles[1])
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
        # Show difference image
        diff_image = np.abs(full_image - under_image)
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_image, cmap='hot', vmax=vmax/4)
        plt.colorbar()
        plt.title("Difference Image")
        plt.show()


def process_cmr_data(data_root, modality, center, vendor, patient, 
                     data_file, mask_type=None, mask_acc=None, 
                     frame_idx=0, slice_idx=None, save_output=False):
    """Process CMR data for visualization and analysis.
    
    Args:
        data_root (str): Root directory of the dataset
        modality (str): Modality name (e.g., 'Cine', 'LGE')
        center (str): Center name (e.g., 'Center001')
        vendor (str): Vendor name (e.g., 'UIH_30T_umr780')
        patient (str): Patient ID (e.g., 'P003')
        data_file (str): Data file name (e.g., 'cine_sax.mat')
        mask_type (str): Mask type (e.g., 'Uniform', 'ktGaussian', 'ktRadial')
        mask_acc (int): Acceleration factor (e.g., 4, 8)
        frame_idx (int): Frame index to use
        slice_idx (int): Slice index to use
        save_output (bool): Whether to save output to .mat files
    """
    utils = CMRReconUtils()
    
    # Build full path to the data file
    full_path = os.path.join(data_root, "MultiCoil", modality, "TrainingSet", 
                           "FullSample", center, vendor, patient, data_file)
    
    # Load k-space data
    print(f"Loading data from {full_path}")
    kspace, data_shape, is_multi_coil = utils.load_kspace(full_path)
    
    if kspace is None:
        print("Failed to load k-space data. Exiting.")
        return
    
    # If slice_idx is not specified, use the middle slice
    if slice_idx is None:
        if is_multi_coil:
            slice_idx = data_shape[1] // 2  # For multi-coil data
        else:
            slice_idx = data_shape[0] // 2  # For single-coil data
    
    # Extract a slice for visualization
    if is_multi_coil:
        slice_kspace = kspace[frame_idx, slice_idx]
        print(f"Extracted frame {frame_idx}, slice {slice_idx} with shape {slice_kspace.shape}")
    else:
        slice_kspace = kspace[frame_idx, slice_idx]
        print(f"Extracted frame {frame_idx}, slice {slice_idx} with shape {slice_kspace.shape}")
    
    # Visualize k-space
    utils.visualize_kspace(slice_kspace)
    
    # Reconstruct image from fully-sampled k-space
    full_image = utils.reconstruct_image(slice_kspace, is_multi_coil=is_multi_coil)
    utils.visualize_image(full_image, title="Fully-sampled Reconstruction")
    
    # If mask is provided, apply undersampling
    if mask_type is not None and mask_acc is not None:
        # Build path to mask file
        mask_filename = f"{os.path.splitext(data_file)[0]}_mask_{mask_type}{mask_acc}.mat"
        mask_path = os.path.join(data_root, "MultiCoil", modality, "TrainingSet", 
                                "Mask_TaskAll", center, vendor, patient, mask_filename)
        
        print(f"Loading mask from {mask_path}")
        mask = utils.load_mask(mask_path)
        
        if mask is not None:
            # Apply undersampling
            undersampled_kspace = utils.apply_mask(kspace, mask, frame_idx, slice_idx)
            
            # Visualize undersampled k-space
            utils.visualize_kspace(undersampled_kspace, coil_indices=[0, 1, 2] if is_multi_coil else None)
            
            # Reconstruct image from undersampled k-space
            under_image = utils.reconstruct_image(undersampled_kspace, is_multi_coil=is_multi_coil)
            
            # Compare fully-sampled and undersampled reconstructions
            utils.compare_images(full_image, under_image)
            
            # Save undersampled k-space if requested
            if save_output:
                output_dir = os.path.join("output", modality, center, vendor, patient)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"under_{os.path.splitext(data_file)[0]}_{mask_type}{mask_acc}.mat")
                utils.save_to_mat(undersampled_kspace, 'underkspace', output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CMRxRecon2025 data for visualization and analysis')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the CMRxRecon2025 dataset')
    parser.add_argument('--modality', type=str, default='Cine', help='Modality (e.g., Cine, LGE)')
    parser.add_argument('--center', type=str, default='Center005', help='Center name (e.g., Center005)')
    parser.add_argument('--vendor', type=str, default='Siemens_30T_Vida', help='Vendor name (e.g., Siemens_30T_Vida)')
    parser.add_argument('--patient', type=str, default='P003', help='Patient ID (e.g., P003)')
    parser.add_argument('--file', type=str, default='cine_sax.mat', help='Data file name (e.g., cine_sax.mat)')
    parser.add_argument('--mask-type', type=str, choices=['Uniform', 'ktGaussian', 'ktRadial'], 
                        help='Mask type for undersampling')
    parser.add_argument('--mask-acc', type=int, choices=[4, 8], help='Acceleration factor for undersampling')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to use')
    parser.add_argument('--slice', type=int, help='Slice index to use (defaults to middle slice)')
    parser.add_argument('--save', action='store_true', help='Save output to .mat files')
    
    args = parser.parse_args()
    
    process_cmr_data(
        data_root=args.root,
        modality=args.modality,
        center=args.center,
        vendor=args.vendor,
        patient=args.patient,
        data_file=args.file,
        mask_type=args.mask_type,
        mask_acc=args.mask_acc,
        frame_idx=args.frame,
        slice_idx=args.slice,
        save_output=args.save
    )