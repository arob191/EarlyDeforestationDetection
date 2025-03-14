# dataset_class.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from torchvision.transforms import functional as F

class GeoTIFFChangeDetectionDataset(Dataset):
    # Read in our data from our CSV files
    def __init__(self, csv_file_early, csv_file_late, resize=(224, 224), normalize=True):
        self.data_frame_early = pd.read_csv(csv_file_early)
        self.data_frame_late = pd.read_csv(csv_file_late)
        self.resize = resize
        self.normalize = normalize
        
        self.data = pd.merge(self.data_frame_early, self.data_frame_late, on='Location_ID', suffixes=('_early', '_late'))
        
        valid_data = []
        for _, row in self.data.iterrows():
            if os.path.exists(row['Path_early']) and os.path.exists(row['Path_late']):
                valid_data.append(row)
            else:
                if not os.path.exists(row['Path_early']):
                    print(f'Warning: File {row["Path_early"]} not found. Skipping...')
                if not os.path.exists(row['Path_late']):
                    print(f'Warning: File {row["Path_late"]} not found. Skipping...')
        
        self.data = pd.DataFrame(valid_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path_early = row['Path_early']
        img_path_late = row['Path_late']
        
        with rasterio.open(img_path_early) as src:
            img_early = src.read().transpose((1, 2, 0))
        
        with rasterio.open(img_path_late) as src:
            img_late = src.read().transpose((1, 2, 0))
        
        img_early = self.transform_image(img_early)
        img_late = self.transform_image(img_late)
        
        label_early = row['Primary_Label_early']
        label_late = row['Primary_Label_late']
        
        label = 1.0 if label_early == 'Intact Forest' and label_late == 'Deforestation' else 0.0
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return (img_early, img_late), label_tensor

    def transform_image(self, img):
        # Convert to torch.Tensor
        img = torch.tensor(img, dtype=torch.float32)

        # Resize
        img = F.resize(img.permute(2, 0, 1), self.resize)  # Permute to (C, H, W) format

        if self.normalize:
            num_channels = img.shape[0]
            mean = torch.tensor([0.485] * num_channels)
            std = torch.tensor([0.229] * num_channels)
            img = (img - mean[:, None, None]) / std[:, None, None]

        return img

