import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.io as sio
import h5py


def ifft2c(kspace):
    """Merkezlenmiş 2D inverse FFT"""
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def load_mat_file(path, keys):
    """MATLAB dosyasından veri yükler (v7.3 için h5py)"""
    try:
        mat = sio.loadmat(path)
        for key in keys:
            if key in mat:
                return mat[key]
        raise KeyError(f"Verilen anahtarlar {keys} bulunamadı.")
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            for key in keys:
                if key in f:
                    return np.array(f[key])
            raise KeyError(f"Verilen anahtarlar {keys} HDF5 dosyasında bulunamadı.")


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

        # === Load k-space ===
        kspace_path = os.path.join(self.full_sample_dir, pid, "T1w.mat")
        kspace = load_mat_file(kspace_path, keys=['kspace', 'T1w'])
        kspace = np.asarray(kspace, dtype=np.complex64)  # [H, W, D1, D2]

        print(f"[DEBUG] kspace shape: {kspace.shape}")

        # Normalize
        kspace /= np.max(np.abs(kspace))

        # === Load mask ===
        mask_path = os.path.join(self.mask_dir, pid, f"T1w_mask_{self.mask_name}.mat")
        mask = load_mat_file(mask_path, keys=['mask'])
        mask = np.asarray(mask, dtype=np.float32)

        print(f"[DEBUG] mask shape: {mask.shape}")

        # Reshape for 2D processing
        kspace_2d = kspace.reshape(512, 207, -1)  # [H, W, N]
        kspace_2d = kspace_2d.transpose(1, 0, 2)  # [W, H, N]
        kspace_2d_mean = kspace_2d.mean(axis=2)  # [W, H]

        # Apply mask
        undersampled_kspace = kspace_2d_mean * mask  # [W, H]

        # IFFT to image
        gt_image = ifft2c(kspace)  # [512, 207, D1, D2]
        zero_filled = ifft2c(undersampled_kspace)  # [207, 512]

        # Convert to tensor [2, H, W]
        def to_tensor_complex(x):
            return torch.from_numpy(np.stack([x.real, x.imag], axis=0)).float()

        return {
            'patient_id': pid,
            'gt_image': to_tensor_complex(gt_image[:, :, 0, 0]),              # [2, H, W]
            'zero_filled': to_tensor_complex(zero_filled),                   # [2, H, W]
            'undersampled_kspace': to_tensor_complex(undersampled_kspace),   # [2, H, W]
            'mask': torch.from_numpy(mask).unsqueeze(0)                       # [1, H, W]
        }


# --- Dataset oluştur ---
dataset = CMRKspaceWithMaskDataset(
    root_dir="cmr_dataset",  # Kendi veri yoluna göre değiştir
    mask_type="Gaussian",
    kt=8
)

# --- Örnek veri al ---
sample = dataset[0]

# --- Görselleri normalize et ---


def normalize(img):
    img = np.abs(img)
    return img / np.max(img)

# --- Kompleks tensörleri numpy ve tek slice haline getir ---


def complex_to_numpy(tensor):
    return tensor[0].numpy() + 1j * tensor[1].numpy()


# --- Görselleştir ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(np.log(1 + np.abs(complex_to_numpy(sample['undersampled_kspace']))), cmap='gray')
axs[0].set_title("K-space (undersampled)")

axs[1].imshow(sample['mask'].squeeze(0), cmap='gray')
axs[1].set_title("Sampling Mask")

axs[2].imshow(normalize(complex_to_numpy(sample['gt_image']).T), cmap='gray')
axs[2].set_title("GT Image (IFFT of full)")

axs[3].imshow(normalize(complex_to_numpy(sample['zero_filled'])), cmap='gray')
axs[3].set_title("Zero-filled Image")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
