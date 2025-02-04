# Lamar Undergrad Research
# Name: Austin Robertson  
# Date: 10/01/2024
# Design and train a CNN to detect future deforestation hotspots

import os
import torch
import numpy as np
import tifffile as tiff
import DatasetClass as ds
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# # Prepare your data transformations
csv_file_path = 'E:\labels.csv'
transform = None
dataset = ds.DeforestationDataset(csv_file=csv_file_path, transform=transform)

# Check to see if torch.cuda or torhc.backends.mps are available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Test convertion to tensor
for i in range(len(dataset)):
    before_image, after_image, label = dataset[i]
    print("Before image shape:", before_image.shape)
    print("After image shape:", after_image.shape)
    print("Label:", label)


# # Initialize the model, loss function, and optimizer

# # Training loop






