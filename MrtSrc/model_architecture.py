import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """Complex-valued convolutional layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # x shape: [batch, channels, height, width, 2] (real and imaginary parts)
        real = x[..., 0]
        imag = x[..., 1]
        
        real_real = self.conv_real(real)
        real_imag = self.conv_imag(real)
        imag_real = self.conv_real(imag)
        imag_imag = self.conv_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        # Stack real and imaginary parts
        return torch.stack([real_out, imag_out], dim=-1)


class ComplexConvBlock(nn.Module):
    """Complex convolutional block with residual connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels)
        self.conv2 = ComplexConv2d(channels, channels)
        self.norm = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        # Split real and imaginary parts for normalization
        real = x[..., 0]
        imag = x[..., 1]
        
        # Normalize real and imaginary parts separately
        real = self.norm(real)
        imag = self.norm(imag)
        
        # Recombine and pass through convolutions
        x_norm = torch.stack([real, imag], dim=-1)
        
        residual = x
        out = self.conv1(x_norm)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        
        return out + residual


class DataConsistencyLayer(nn.Module):
    """Data consistency layer for k-space reconstruction."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x_rec, x_under, mask):
        """
        Args:
            x_rec: Reconstructed k-space
            x_under: Undersampled k-space
            mask: Sampling mask
        """
        # Apply mask to get data consistent reconstruction
        return x_under * mask + x_rec * (1 - mask)


class KSpaceUNet(nn.Module):
    """U-Net architecture for k-space reconstruction."""
    
    def __init__(self, in_channels=1, out_channels=1, num_filters=32, num_pool_layers=4):
        super().__init__()
        
        # Encoding path
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # First block
        self.first_block = ComplexConvBlock(in_channels)
        ch = in_channels
        
        # Encoder blocks with downsampling
        for i in range(num_pool_layers):
            out_ch = ch * 2
            self.encoder_blocks.append(ComplexConvBlock(out_ch))
            self.pool_layers.append(nn.MaxPool2d(2))
            ch = out_ch
        
        # Middle block
        self.middle_block = ComplexConvBlock(ch)
        
        # Decoding path
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        # Decoder blocks with upsampling
        for i in range(num_pool_layers):
            out_ch = ch // 2
            self.upconv_layers.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.decoder_blocks.append(ComplexConvBlock(out_ch))
            ch = out_ch
        
        # Final layer
        self.final_conv = ComplexConv2d(ch, out_channels)
        
        # Data consistency layer
        self.dc_layer = DataConsistencyLayer()
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input k-space data [batch, channels, height, width, 2]
            mask: Sampling mask [batch, 1, height, width, 1]
        """
        # Initial features
        x_under = x  # Store undersampled input for data consistency
        
        # Encoder
        x = self.first_block(x)
        encoder_outputs = [x]
        
        for i, (encoder, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            # Split for pooling (can't pool complex data directly)
            real = pool(x[..., 0])
            imag = pool(x[..., 1])
            x = torch.stack([real, imag], dim=-1)
            
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Middle
        x = self.middle_block(x)
        
        # Decoder
        for i, (decoder, upconv) in enumerate(zip(self.decoder_blocks, self.upconv_layers)):
            # Split for upsampling (can't upsample complex data directly)
            real = upconv(x[..., 0])
            imag = upconv(x[..., 1])
            x = torch.stack([real, imag], dim=-1)
            
            # Skip connection
            x = torch.cat([x, encoder_outputs[-(i+2)]], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Data consistency
        if mask is not None:
            x = self.dc_layer(x, x_under, mask)
        
        return x


class DualDomainCMRRecon(nn.Module):
    """Dual-domain model for CMR reconstruction.
    
    This model operates in both k-space and image domain.
    """
    
    def __init__(self, in_channels=1, out_channels=1, num_filters=32, num_cascades=5):
        super().__init__()
        
        self.num_cascades = num_cascades
        
        # K-space network
        self.kspace_net = KSpaceUNet(in_channels, out_channels, num_filters)
        
        # Image domain refinement
        self.image_net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
        
        # Data consistency layers
        self.dc_layers = nn.ModuleList([DataConsistencyLayer() for _ in range(num_cascades)])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input k-space data [batch, channels, height, width, 2]
            mask: Sampling mask [batch, 1, height, width, 1]
        """
        x_original = x
        
        for i in range(self.num_cascades):
            # K-space reconstruction
            x = self.kspace_net(x, mask)
            
            # Convert to image domain
            x_image = ifft2c(x)  # This would need to be implemented properly
            
            # Image domain refinement (needs reshape for Conv2d)
            batch, chans, h, w, comp = x_image.shape
            x_image_reshaped = x_image.reshape(batch, chans*comp, h, w)
            x_image_refined = self.image_net(x_image_reshaped)
            x_image_refined = x_image_refined.reshape(batch, chans, h, w, comp)
            
            # Convert back to k-space
            x = fft2c(x_image_refined)  # This would need to be implemented properly
            
            # Data consistency
            if mask is not None:
                x = self.dc_layers[i](x, x_original, mask)
        
        return x


class MultiModalityModel(nn.Module):
    """Model that handles multiple MRI modalities."""
    
    def __init__(self, modalities=['Cine', 'LGE', 'T1map', 'T2map']):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Modality-specific decoders
        self.decoders = nn.ModuleDict()
        for modality in modalities:
            self.decoders[modality] = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 2, kernel_size=3, padding=1)
            )
        
        # Main reconstruction model
        self.recon_model = DualDomainCMRRecon()
    
    def forward(self, x, mask=None, modality='Cine'):
        """
        Args:
            x: Input k-space data
            mask: Sampling mask
            modality: Current modality being processed
        """
        # Main reconstruction
        x_recon = self.recon_model(x, mask)
        
        # Convert to image domain for refinement
        x_image = ifft2c(x_recon)  # This would need to be implemented properly
        
        # Reshape for 2D convs
        batch, chans, h, w, comp = x_image.shape
        x_image_reshaped = x_image.reshape(batch, chans*comp, h, w)
        
        # Shared encoding
        features = self.shared_encoder(x_image_reshaped)
        
        # Modality-specific decoding
        refined = self.decoders[modality](features)
        refined = refined.reshape(batch, chans, h, w, comp)
        
        # Convert back to k-space
        x_refined = fft2c(refined)  # This would need to be implemented properly
        
        return x_refined


class VendorAdaptiveModel(nn.Module):
    """Model with vendor-adaptive components for better generalization."""
    
    def __init__(self, base_model, num_vendors=5):
        super().__init__()
        
        # Base reconstruction model
        self.base_model = base_model
        
        # Vendor-specific adapters
        self.vendor_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_vendors)
        ])
        
        # Vendor classification branch
        self.vendor_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_vendors),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, mask=None, vendor_id=None):
        """
        Args:
            x: Input k-space data
            mask: Sampling mask
            vendor_id: ID of the vendor (if known, otherwise predicted)
        """
        # Initial reconstruction
        x_recon = self.base_model(x, mask)
        
        # Extract features
        # This would need a feature extraction implementation
        features = extract_features(x_recon)
        
        # Predict vendor if not provided
        if vendor_id is None:
            vendor_probs = self.vendor_classifier(features)
            # Use soft adaptation with weighted average
            adapted_features = sum(prob * adapter(features) for prob, adapter in zip(vendor_probs, self.vendor_adapters))
        else:
            # Use specific vendor adapter
            adapted_features = self.vendor_adapters[vendor_id](features)
        
        # Final reconstruction with adapted features
        # This would need to be implemented based on the feature integration method
        x_final = integrate_features(x_recon, adapted_features)
        
        return x_final


# Utility functions that would need to be implemented
def fft2c(x):
    """Centered 2D Fourier transform."""
    pass

def ifft2c(x):
    """Centered 2D inverse Fourier transform."""
    pass

def extract_features(x):
    """Extract features from reconstruction."""
    pass

def integrate_features(x, features):
    """Integrate features into reconstruction."""
    pass