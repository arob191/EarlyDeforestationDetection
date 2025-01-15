import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile as tiff
import torch
import os

# Function to load TIFF image and process it
def load_tiff_to_tensor(image_path):
    # Open the TIFF image
    img_array = tiff.imread(image_path)

    # Handle NaNs
    img_array = np.nan_to_num(img_array, nan=0.0)

    # Normalize and clip values to the 0-255 range
    img_array = np.clip(img_array, 0, 255)

    # Convert the array to uint8
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    # Convert the Numpy array to a PyTorch tensor
    img_tensor = torch.from_numpy(img_array)
    
    return img_tensor

# Sanitize file paths
def sanitize_path(path):
    return os.path.normpath(path.strip())

# Custom dataset class
class DeforestationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        print(self.data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        before_img_path = sanitize_path(self.data.iloc[idx, 0])
        after_img_path = sanitize_path(self.data.iloc[idx, 1])
        print(f'Before image path: {before_img_path}')
        print(f'After image path: {after_img_path}')
        
        before_image = load_tiff_to_tensor(before_img_path)
        after_image = load_tiff_to_tensor(after_img_path)
        label = self.data.iloc[idx, 2] if 'label' in self.data.columns else None

        if self.transform:
            before_image = self.transform(before_image)
            after_image = self.transform(after_image)

        return before_image, after_image, label

# Example usage
# dataset = DeforestationDataset(csv_file='deforestation_dataset.csv', transform=your_transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
