import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import fastmri
from fastmri.data import transforms as T

# Import your model
from model_architecture import DualDomainCMRRecon, MultiModalityModel, VendorAdaptiveModel


class CMRReconDataset(Dataset):
    """Dataset for CMR reconstruction."""
    
    def __init__(self, data_paths, mask_paths=None, modality=None, transform=None, is_test=False):
        """
        Args:
            data_paths (list): List of paths to data files
            mask_paths (list, optional): List of paths to mask files
            modality (str, optional): Modality name
            transform (callable, optional): Transform to apply to the data
            is_test (bool): Whether this is a test dataset
        """
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.modality = modality
        self.transform = transform
        self.is_test = is_test
        
        # Cache metadata about files for faster access
        self.metadata = []
        for i, data_path in enumerate(data_paths):
            # Determine vendor and center from path
            parts = Path(data_path).parts
            center_idx = [j for j, part in enumerate(parts) if part.startswith("Center")][0]
            
            metadata = {
                "data_path": data_path,
                "mask_path": mask_paths[i] if mask_paths else None,
                "center": parts[center_idx],
                "vendor": parts[center_idx + 1],
                "patient": parts[center_idx + 2],
                "modality": modality or Path(data_path).stem
            }
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        data_path = metadata["data_path"]
        mask_path = metadata["mask_path"]
        
        # Load k-space data
        with h5py.File(data_path, 'r') as hf:
            kspace = hf['kspace']['real'] + 1j * hf['kspace']['imag']
            
            # Handle different shapes for different modalities
            if len(kspace.shape) == 5:  # (nframe, nslice, ncoil, ny, nx)
                # For simplicity, just pick a random frame and slice
                frame_idx = np.random.randint(0, kspace.shape[0])
                slice_idx = np.random.randint(0, kspace.shape[1])
                kspace_slice = kspace[frame_idx, slice_idx]
            else:
                # Handle other shapes as needed
                kspace_slice = kspace
        
        # Load mask if provided
        mask = None
        if mask_path:
            with h5py.File(mask_path, 'r') as hf:
                mask_key = 'mask' if 'mask' in hf else list(hf.keys())[0]
                mask = hf[mask_key][()]
                
                # Handle different mask shapes
                if len(mask.shape) == 3:  # (nframe, ny, nx)
                    mask = mask[frame_idx]
        
        # Apply mask to create undersampled data
        if mask is not None:
            # Expand mask to match kspace dimensions
            expanded_mask = np.expand_dims(mask, axis=0)
            expanded_mask = np.tile(expanded_mask, (kspace_slice.shape[0], 1, 1))
            kspace_under = kspace_slice * expanded_mask
        else:
            kspace_under = kspace_slice
            expanded_mask = np.ones_like(kspace_slice)
        
        # Convert to tensors
        kspace_tensor = T.to_tensor(kspace_slice)
        kspace_under_tensor = T.to_tensor(kspace_under)
        mask_tensor = T.to_tensor(expanded_mask)
        
        # Apply transform if provided
        if self.transform:
            kspace_tensor, kspace_under_tensor, mask_tensor = self.transform(
                kspace_tensor, kspace_under_tensor, mask_tensor
            )
        
        # Create sample dictionary
        sample = {
            "kspace_full": kspace_tensor,
            "kspace_under": kspace_under_tensor,
            "mask": mask_tensor,
            "metadata": metadata
        }
        
        return sample


class SSIMLoss(nn.Module):
    """SSIM loss for image comparison."""
    
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1 = k1
        self.k2 = k2
    
    def forward(self, x, y):
        """
        Args:
            x, y: Images to compare
        """
        # Convert to magnitude images if complex
        if x.shape[-1] == 2:  # Complex data
            x = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)
            y = torch.sqrt(y[..., 0]**2 + y[..., 1]**2)
        
        # Use PyTorch's built-in SSIM implementation if available
        # or implement a custom SSIM function
        return 1 - ssim(x, y, win_size=self.win_size, k1=self.k1, k2=self.k2)


class ComplexMSELoss(nn.Module):
    """MSE loss for complex data."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: Complex tensors with shape [..., 2]
        """
        return torch.mean((pred[..., 0] - target[..., 0])**2 + (pred[..., 1] - target[..., 1])**2)


class NMSELoss(nn.Module):
    """Normalized MSE loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: Tensors
        """
        if pred.shape[-1] == 2:  # Complex data
            # Convert to magnitude
            pred = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
            
        error = torch.sum((pred - target)**2, dim=(1, 2, 3))
        norm = torch.sum(target**2, dim=(1, 2, 3))
        return torch.mean(error / norm)


class PSNRLoss(nn.Module):
    """PSNR loss."""
    
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: Tensors
        """
        if pred.shape[-1] == 2:  # Complex data
            # Convert to magnitude
            pred = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
            
        mse = torch.mean((pred - target)**2)
        return -10 * torch.log10(mse + 1e-8)


class Trainer:
    """Trainer for CMR reconstruction models."""
    
    def __init__(self, model, train_loader, val_loader, optimizer, 
                 criterion, device, checkpoint_dir="checkpoints", 
                 log_dir="logs"):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics = {
            'ssim': [],
            'psnr': [],
            'nmse': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for batch in pbar:
                # Move data to device
                kspace_full = batch["kspace_full"].to(self.device)
                kspace_under = batch["kspace_under"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(kspace_under, mask)
                
                # Calculate loss
                loss = self.criterion(output, kspace_full)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        epoch_loss = 0
        
        # Metric calculators
        ssim_metric = SSIMLoss()
        nmse_metric = NMSELoss()
        psnr_metric = PSNRLoss()
        
        epoch_metrics = {
            'ssim': 0,
            'psnr': 0,
            'nmse': 0
        }
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]") as pbar:
                for batch in pbar:
                    # Move data to device
                    kspace_full = batch["kspace_full"].to(self.device)
                    kspace_under = batch["kspace_under"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    
                    # Forward pass
                    output = self.model(kspace_under, mask)
                    
                    # Calculate loss
                    loss = self.criterion(output, kspace_full)
                    
                    # Calculate metrics
                    # Convert to image domain for SSIM and PSNR
                    image_full = ifft2c(kspace_full)
                    image_output = ifft2c(output)
                    
                    ssim_val = 1 - ssim_metric(image_output, image_full)
                    psnr_val = psnr_metric(image_output, image_full)
                    nmse_val = nmse_metric(image_output, image_full)
                    
                    epoch_metrics['ssim'] += ssim_val.item()
                    epoch_metrics['psnr'] += psnr_val.item()
                    epoch_metrics['nmse'] += nmse_val.item()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.val_loader)
            self.metrics[k].append(epoch_metrics[k])
        
        # Print metrics
        print(f"Validation metrics: SSIM={epoch_metrics['ssim']:.4f}, "
              f"PSNR={epoch_metrics['psnr']:.2f}, "
              f"NMSE={epoch_metrics['nmse']:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': self.metrics
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            int: Epoch number
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.metrics = checkpoint['metrics']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs, resume_from=None):
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        # Resume training if checkpoint provided
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Plot and save losses
            self.plot_losses()
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} completed in {elapsed:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Best Val Loss: {self.best_val_loss:.4f}")
    
    def plot_losses(self):
        """Plot and save loss curves."""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot metrics
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['ssim'], label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['psnr'], label='PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png")
        plt.close()


def train_model(data_root, output_dir, model_type="dual_domain", modality="Cine", 
                batch_size=4, num_epochs=50, learning_rate=1e-4, device=None):
    """Train a model on CMR data.
    
    Args:
        data_root: Root directory of the dataset
        output_dir: Directory to save outputs
        model_type: Type of model to train
        modality: Modality to train on
        batch_size: Batch size
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to use for training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Get file paths
    # In practice, you would use your dataset explorer to get these paths
    train_data_paths = []  # List of paths to training data
    train_mask_paths = []  # List of paths to training masks
    val_data_paths = []    # List of paths to validation data
    val_mask_paths = []    # List of paths to validation masks
    
    # Create datasets
    train_dataset = CMRReconDataset(
        train_data_paths, train_mask_paths, modality=modality
    )
    
    val_dataset = CMRReconDataset(
        val_data_paths, val_mask_paths, modality=modality
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    if model_type == "dual_domain":
        model = DualDomainCMRRecon(in_channels=1, out_channels=1, num_filters=32)
    elif model_type == "multi_modality":
        model = MultiModalityModel(modalities=[modality])
    elif model_type == "vendor_adaptive":
        base_model = DualDomainCMRRecon(in_channels=1, out_channels=1, num_filters=32)
        model = VendorAdaptiveModel(base_model, num_vendors=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function
    criterion = ComplexMSELoss()
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, criterion, device,
        checkpoint_dir=checkpoint_dir, log_dir=log_dir
    )
    
    # Train model
    trainer.train(num_epochs)
    
    print(f"Training completed. Model checkpoints saved to {checkpoint_dir}")


# Utility functions
def ifft2c(x):
    """Centered 2D inverse Fourier transform."""
    # This is a placeholder. You would need to implement this properly.
    return x

def ssim(x, y, win_size=7, k1=0.01, k2=0.03):
    """Structural Similarity Index Measure."""
    # This is a placeholder. You would need to implement this properly.
    return torch.tensor(0.5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CMR reconstruction model')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--model-type', type=str, default='dual_domain', 
                        choices=['dual_domain', 'multi_modality', 'vendor_adaptive'],
                        help='Type of model to train')
    parser.add_argument('--modality', type=str, default='Cine', help='Modality to train on')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    train_model(
        args.data_root, args.output_dir, args.model_type, args.modality,
        args.batch_size, args.num_epochs, args.lr, device
    )