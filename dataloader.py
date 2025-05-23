import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py


def ifft2c(kspace):
    """Merkezlenmiş 2D inverse FFT (son 2 boyut)"""
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def load_mat_file(path, keys):
    """MATLAB dosyasından veri yükler"""
    try:
        mat = sio.loadmat(path)
        for key in keys:
            if key in mat:
                return mat[key]
        raise KeyError(f"Keys {keys} not found in mat file.")
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            for key in keys:
                if key in f:
                    return np.array(f[key])
            raise KeyError(f"Keys {keys} not found in HDF5 file.")


class CMRKspaceWithMaskDataset(Dataset):
    def __init__(self, root_dir, mask_type="Gaussian", kt=8):
        self.full_sample_dir = os.path.join(root_dir, "full_sample")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.mask_name = f"kt{mask_type}{kt}"

        self.patient_ids = sorted([
            d for d in os.listdir(self.full_sample_dir)
            if os.path.isdir(os.path.join(self.full_sample_dir, d)) and not d.startswith('.')
        ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # Load k-space: [H, W, n_coils, n_frames]
        kspace_path = os.path.join(self.full_sample_dir, pid, "T1w.mat")
        kspace = load_mat_file(kspace_path, keys=['kspace', 'T1w'])
        kspace = np.asarray(kspace, dtype=np.complex64)
        print(f"[DEBUG] kspace shape: {kspace.shape}")

        # Normalize kspace
        kspace /= np.max(np.abs(kspace))

        # Load mask: assume shape [H, W]
        mask_path = os.path.join(self.mask_dir, pid, f"T1w_mask_{self.mask_name}.mat")
        mask = load_mat_file(mask_path, keys=['mask'])
        mask = np.asarray(mask, dtype=np.float32)
        print(f"[DEBUG] mask shape: {mask.shape}")

        H, W, n_coils, n_frames = kspace.shape

        # Choose a frame (e.g. first frame)
        frame_idx = 0
        kspace_frame = kspace[:, :, :, frame_idx]  # [H, W, n_coils]

        # Apply mask on each coil kspace
        undersampled_kspace = kspace_frame * mask.T[:, :, None]  # broadcast mask to coils

        # IFFT coil bazında
        img_full_coils = ifft2c(kspace_frame.transpose(2, 0, 1))  # [n_coils, H, W]
        img_undersampled_coils = ifft2c(undersampled_kspace.transpose(2, 0, 1))  # [n_coils, H, W]

        # RSS coil combine (magnitude)
        rss_full = np.sqrt(np.sum(np.abs(img_full_coils) ** 2, axis=0))  # [H, W]
        rss_undersampled = np.sqrt(np.sum(np.abs(img_undersampled_coils) ** 2, axis=0))  # [H, W]

        # Normalize images
        rss_full /= np.max(rss_full)
        rss_undersampled /= np.max(rss_undersampled)

        # Convert to tensor, 1 channel (real-valued RSS magnitude)
        gt_image = torch.from_numpy(rss_full).float().unsqueeze(0)             # [1, H, W]
        zero_filled = torch.from_numpy(rss_undersampled).float().unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)              # [1, H, W]

        return {
            'patient_id': pid,
            'gt_image': gt_image,
            'zero_filled': zero_filled,
            'mask': mask_tensor
        }


# --- Dataset oluştur ---
dataset = CMRKspaceWithMaskDataset(
    root_dir="cmr_dataset",
    mask_type="Gaussian",
    kt=8
)

# --- Örnek veri al ---
sample = dataset[0]

# --- Görselleştir ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(sample['gt_image'][0].numpy().T, cmap='gray')
axs[0].set_title("Fully Sampled Image")

axs[1].imshow(sample['mask'][0].numpy(), cmap='gray')
axs[1].set_title("Sampling Mask")

axs[2].imshow(sample['zero_filled'][0].numpy().T, cmap='gray')
axs[2].set_title("Mask Applied Image")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
