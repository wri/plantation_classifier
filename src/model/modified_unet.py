import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import os
import yaml
from utils.logs import get_logger
from features import create_xy, PlantationsData

class ConvBlock(nn.Module):
    """
    One deep convolutional layer capturing abstract spatial features.
    Consists of:
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

class Dataset(Dataset):
    '''
    Creates a custom PyTorch dataset for loading Sentinel and texture features.
    this needs to load in a single plot of size (1, 29, 14, 14)
    '''
    def __init__(self, dataset_path,):
        self.dataset_path = dataset_path
        self.datapoints = os.listdir(self.dataset_path + 'train-pytorch/')
        self.datapoints = [x for x in self.datapoints if x[-4:] == '.hkl']
        # how do i load in the y values?
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, 'train-pytorch/')
        target_path = os.path.join(self.dataset_path, 'train-labels/')
        input = hkl.load(img_path + self.datapoints[idx])
        output = np.load(target_path + self.datapoints[idx][:-4] + ".npy")
        # confirm if these steps are necessary?
        x = self.X[idx]
        x = np.transpose(x, (2, 0, 1))  # (channels, height, width) for CNN
        y_label = self.y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)



  # this process needs to import each plot in the img_path "data/train-pytorch", 
  # which is a (14,14,29) array and import each corresponding plot in the target_path
  # "data/train-labels" which is a (14, 14) array, using __getitem__.
  # Normalization needs to happen to center the training dataset around 0
  # the data should be loaded as individual tiles and then batched and fed to the model
  # in batch_size

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest = 'param_path', type = str)
    args = parser.parse_args()

    with open(args.param_path) as file:
        params = yaml.safe_load(file)

    dataset = Dataset("../../data/") # what does Dataset take as input?
    dataloader = DataLoader(dataset, 
                            batch_size=params['pytorch']['batch_size'], 
                            shuffle=True)


