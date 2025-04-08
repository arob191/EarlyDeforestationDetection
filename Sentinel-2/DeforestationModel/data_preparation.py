import os
import pandas as pd
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for multi-task deforestation prediction.
    
    For each sample, this dataset loads:
      • an NDVI difference image (converted to a 3-channel RGB image for ResNet50),
      • a segmentation mask (ternary mask, remapped from [-1, 0, 1] to [0, 1, 2]),
      • a regression label: overall deforestation percentage (computed as percentage of pixels originally -1),
      • a distance map (normalized separately, as a single-channel image).
    
    The deforestation CSV is expected to include:
      "tile_id", "forest", "time_period_comparison", 
      "ndvi_diff_path", "ternary_mask_path", "distance_map_path"
    """
    def __init__(self, csv_file, transform_input=None, transform_target=None, transform_distance=None):
        """
        Args:
            csv_file (str): Path to the deforestation CSV.
            transform_input (callable, optional): Transformations for the RGB NDVI difference image.
            transform_target (callable, optional): Transformations for the segmentation mask.
            transform_distance (callable, optional): Transformations for the distance map.
        """
        self.df = pd.read_csv(csv_file)
        self.transform_input = transform_input
        self.transform_target = transform_target
        # Default transform for distance map: resize to 224x224 and convert to tensor.
        self.transform_distance = transform_distance if transform_distance else transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ndvi_diff_path = row['ndvi_diff_path']
        ternary_mask_path = row['ternary_mask_path']
        distance_map_path = row['distance_map_path']

        # -------------------------
        # Load NDVI difference image
        # -------------------------
        with rasterio.open(ndvi_diff_path) as src:
            ndvi_diff = src.read(1).astype('float32')  # shape (H, W)
        
        # Normalize NDVI difference (using min-max scaling) and convert to uint8
        ndvi_min, ndvi_max = np.min(ndvi_diff), np.max(ndvi_diff)
        # Normalize ndvi_diff to the 0-255 range.
        ndvi_norm = ((ndvi_diff - ndvi_min) / (ndvi_max - ndvi_min + 1e-6)) * 255.0
        # Replace NaNs with zeros.
        ndvi_norm = np.nan_to_num(ndvi_norm, nan=0.0)
        ndvi_norm = ndvi_norm.astype(np.uint8)
        # Convert to a PIL image then to RGB (3 channels)
        pil_ndvi = Image.fromarray(ndvi_norm, mode='L').convert('RGB')
        if self.transform_input:
            input_tensor = self.transform_input(pil_ndvi)
        else:
            input_tensor = transforms.ToTensor()(pil_ndvi)
        
        # -------------------------
        # Load ternary mask and remap labels
        # -------------------------
        with rasterio.open(ternary_mask_path) as src:
            mask = src.read(1).astype('int16')
        # Remap: -1 -> 0, 0 -> 1, 1 -> 2
        mask_remapped = np.copy(mask)
        mask_remapped[mask == -1] = 0
        mask_remapped[mask == 0] = 1
        mask_remapped[mask == 1] = 2
        pil_mask = Image.fromarray(mask_remapped.astype(np.uint8), mode='L')
        if self.transform_target:
            target_tensor = self.transform_target(pil_mask)
        else:
            target_tensor = transforms.ToTensor()(pil_mask).squeeze(0).long()  # shape (H, W)
        
        # -------------------------
        # Compute regression target: % of pixels originally -1 (deforestation)
        # -------------------------
        defo_pixels = (mask == -1).sum()
        total_pixels = mask.size
        defo_pct = (defo_pixels / total_pixels) * 100.0
        regression_label = torch.tensor(defo_pct, dtype=torch.float32)
        
        # -------------------------
        # Load and process the distance map
        # -------------------------
        with rasterio.open(distance_map_path) as src:
            distance_map = src.read(1).astype('float32')  # shape (H, W)
        # Normalize the distance map (min-max scaling to 0-255)
        d_min, d_max = np.min(distance_map), np.max(distance_map)
        distance_norm = ((distance_map - d_min) / (d_max - d_min + 1e-6)) * 255.0
        distance_norm = distance_norm.astype(np.uint8)
        # Create a PIL image from the normalized distance map (leave as single channel)
        pil_distance = Image.fromarray(distance_norm, mode='L')
        if self.transform_distance:
            distance_tensor = self.transform_distance(pil_distance)  # shape [1, 224, 224]
        else:
            distance_tensor = transforms.ToTensor()(pil_distance)
        
        sample = {
            "input": input_tensor,               # NDVI diff in 3 channels, shape: [3, 224, 224]
            "segmentation": target_tensor,         # Segmentation mask, shape: [224, 224]
            "regression": regression_label,        # Deforestation percentage (scalar tensor)
            "distance": distance_tensor,           # Distance map tensor, shape: [1, 224, 224]
            "tile_id": row['tile_id'],
            "forest": row['forest'],
            "time_period_comparison": row['time_period_comparison']
        }
        return sample

def main():
    # Path to the deforestation CSV (should include a "distance_map_path" column)
    csv_file = r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_All_Pairs.csv"

    # Define transformations:
    input_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    target_trans = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST)
        # Conversion to tensor happens by default in __getitem__
    ])
    # For distance maps we use the default transform defined in the dataset

    dataset = DeforestationDataset(csv_file,
                                   transform_input=input_trans,
                                   transform_target=target_trans)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Get one batch to check shapes:
    for batch in dataloader:
        print("Batch input shape:", batch["input"].shape)             # [batch, 3, 224, 224]
        print("Batch segmentation shape:", batch["segmentation"].shape) # [batch, 224, 224]
        print("Batch distance shape:", batch["distance"].shape)         # [batch, 1, 224, 224]
        print("Regression labels:", batch["regression"])
        print("Tile IDs:", batch["tile_id"])
        break

if __name__ == "__main__":
    main()