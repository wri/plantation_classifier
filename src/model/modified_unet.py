import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data import random_split
from sklearn.metrics import balanced_accuracy_score
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
    Creates a custom PyTorch dataset for loading Sentinel features.
    __getitem__ loads a single sample (img and corresponding target) at a time.
    Mins and maxs are calculated across all sentinel input tiles using the
    code in src/utils/min_max.py
    '''
    def __init__(self, plot_ids, input_dir, label_dir):
        self.plot_ids = plot_ids
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.min_all = [0.01725032366812229, 0.029564354568719864, 0.01933318004012108, 
                   0.06290531903505325, 0.03508812189102173, 0.05046158283948898, 
                   0.0642404854297638, 0.05921263247728348, 0.01604486256837845, 
                   0.0052948808297514915, 0.0, 0.0, 0.0]
        self.max_all = [0.4679408073425293, 0.4629587233066559, 0.41253527998924255, 
                   0.5527504682540894, 0.47520411014556885, 0.464446485042572, 
                   0.5933089256286621, 0.6391470432281494, 0.5431296229362488, 
                   0.4426642060279846, 0.49999237060546875, 0.9672541618347168, 
                   0.890066385269165]

    def __len__(self):
        return len(self.plot_ids)

    def __getitem__(self, idx):
        plot_id = self.plot_ids[idx]
        x_path = os.path.join(self.input_dir, f"{plot_id}.hkl")
        y_path = os.path.join(self.label_dir, f"{plot_id}.npy")

        x = hkl.load(x_path).astype(np.float32)   # (14, 14, 16)
        x = np.transpose(x, (2, 0, 1))            # (16, 14, 14)
        for band in range(x.shape[0]):  # iterate over bands (channels)
            mins = self.min_all[band]
            maxs = self.max_all[band]
            x[band] = np.clip(x[band], mins, maxs)
            midrange = (maxs + mins) / 2
            rng = maxs - mins
            x[band] = (x[band] - midrange) / (rng / 2)
        x = torch.tensor(x, dtype=torch.float32)

        y = np.load(y_path).astype(np.int64)      # (14, 14)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest='param_path', type=str)
    args = parser.parse_args()

    with open(args.param_path) as file:
        params = yaml.safe_load(file)
    
    local_dir = params['data_load']['local_prefix']

    with open(f"{local_dir}train_params/train_ids.txt") as f:
        train_ids = [line.strip() for line in f]
    with open(f"{local_dir}train_params/val_ids.txt") as f:
        val_ids = [line.strip() for line in f]

    train_dataset = Dataset(train_ids, "data/train-pytorch", "data/train-labels")
    val_dataset = Dataset(val_ids, "data/train-pytorch", "data/train-labels")

    train_loader = DataLoader(train_dataset, batch_size=params['pytorch']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['pytorch']['batch_size'], shuffle=False)

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
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                correct_pixels += (preds == target).sum().item()
                total_pixels += target.numel()
                # For balanced accuracy
                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_labels.extend(target.cpu().numpy().reshape(-1))

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_pixels / total_pixels
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)


        print(f"Epoch [{epoch}/{params['pytorch']['epochs']}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f},"
            f"Balanced Accuracy: {val_bal_acc:.4f}")

