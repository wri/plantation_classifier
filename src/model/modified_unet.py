import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import os

class ConvBlock(nn.Module):
    """
    A convolutional block that consists of:
    - Conv2D layer
    - Batch Normalization
    - ReLU Activation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=3, 
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    """
    U-Net model modified for 4-class land use classification.
    """
    def __init__(self, in_channels=94, num_classes=4):
        super().__init__()
        
        # Encoding Path (Downsampling)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        # Bottleneck Layer
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoding Path (Upsampling)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Output Layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoding
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final classification layer
        output = self.final(d1)
        return output

class TreeDataset(Dataset):
    """
    Custom PyTorch dataset for loading Sentinel and tree feature data.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.files[idx])
        sample = np.load(data_path)
        
        # First 94 bands are features
        x = sample[:, :, :94]
        x = np.transpose(x, (2, 0, 1))  # Convert to (C, H, W) for PyTorch
        
        # Last band is the label
        y = sample[:, :, 94]
        y = torch.tensor(y, dtype=torch.long)
        
        return torch.tensor(x, dtype=torch.float32), y
