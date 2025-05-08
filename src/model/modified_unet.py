import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data import random_split
import numpy as np
import os
import yaml
import hickle as hkl
import random

class ConvBlock(nn.Module):
    """
    One deep convolutional layer capturing abstract spatial features.
    Consists of:
    - Conv2D layer
    - Batch Normalization
    - ReLU Activation
    """
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=3, 
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    """
    U-Net model for input size 28x28 and output size 14x14.

    Encoder path:
    28x28 → 14x14 → 12x12 → 6x6 → 4x4 (bottleneck)

    Decoder path:
    4x4 → 8x8 → 16x16 → 14x14 (output)

    Input:
    - Tensor of shape (batch_size, in_channels, 28, 28)
    Output:
    - Tensor of shape (batch_size, num_classes, 14, 14)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Encoder layers
        self.enc1 = ConvBlock(in_channels, 64, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14

        self.enc2 = ConvBlock(64, 128, padding=1)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  # 14x14 -> 12x12

        self.enc3 = ConvBlock(128, 256, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 12x12 -> 6x6

        self.bottleneck = ConvBlock(256, 512, padding=1)
        self.reduce = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)  # 6x6 -> 4x4

        # Decoder layers
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 4x4 -> 8x8
        self.dec1 = ConvBlock(256, 256, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec2 = ConvBlock(128, 128, padding=1)

        # Instead of ConvTranspose2d here, we use Conv2d to go 16x16 → 14x14
        self.up3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)  # 16x16 → 14x14
        self.dec3 = ConvBlock(64, 64, padding=1)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)
        b = self.reduce(b)

        # Decoder
        d1 = self.up1(b)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = self.dec3(d3)

        output = self.final(d3)
        return output

class Dataset(Dataset):
    '''
    Creates a custom PyTorch dataset for loading Sentinel and texture features.
    __getitem__ loads a single sample (img and corresponding target) one at a time
    '''
    def __init__(self, dataset_path):
        self.img_path = os.path.join(dataset_path, 'train-pytorch/')
        self.target_path = os.path.join(dataset_path, 'train-labels/')
        self.datapoints = [f for f in os.listdir(self.img_path) if f.endswith('.hkl')]
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        img_file = self.datapoints[idx]
        img = hkl.load(os.path.join(self.img_path, img_file))  # (14,14,29)
        img = (img - np.mean(img)) / np.std(img)  # need to norm per band not img L440
        img = np.transpose(img, (2,0,1))  # (29,14,14)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        label_file = img_file.replace('.hkl', '.npy')
        label = np.load(os.path.join(self.target_path, label_file))  # (14,14)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest='param_path', type=str)
    args = parser.parse_args()

    with open(args.param_path) as file:
        params = yaml.safe_load(file)

    # uses same random_state and split sizes from params.yaml
    random_state = params['base']['random_state']
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    test_fraction = params['data_condition']['test_split'] / 100
    train_fraction = params['data_condition']['train_split'] / 100
    dataset = Dataset("data/")
    val_size = int(len(dataset) * test_fraction)
    train_size = len(dataset) - val_size
    print(f"validation size: {val_size}, train size: {train_size}")

    # Perform split using random_state as the generator seed
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state)
    )

    train_loader = DataLoader(train_dataset, 
                              batch_size=params['pytorch']['batch_size'], # batch size is for model training not split
                              shuffle=True
                              )
    val_loader = DataLoader(val_dataset, 
                            batch_size=params['pytorch']['batch_size'], 
                            shuffle=False
                            )

    #Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=params['pytorch']['in_channels'], 
                 num_classes=4
                 )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, params['pytorch']['epochs'] + 1):
        # Training phase
        model.train()
        train_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()              # reset gradients
            output = model(data)               # forward pass, output shape is (16 x 4 x 14 x 14)
            loss = criterion(output, target)   # compute loss
            loss.backward()                    # backward pass (compute gradients)
            optimizer.step()                   # update weights

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                correct_pixels += (preds == target).sum().item()
                total_pixels += target.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_pixels / total_pixels

        print(f"Epoch [{epoch}/{params['pytorch']['epochs']}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}")
