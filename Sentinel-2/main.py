# main.py

import torch
from torch.utils.data import DataLoader
from dataset_class import GeoTIFFChangeDetectionDataset, custom_collate_fn  # Import the modified dataset class
from custom_transform import CustomTransform  # Import the custom transform class

custom_transform = CustomTransform(augment=False)  # Disable augmentation for simplicity

csv_file_path_early = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2015_2016.csv'
csv_file_path_late = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2017_2018.csv'

dataset = GeoTIFFChangeDetectionDataset(csv_file_early=csv_file_path_early, csv_file_late=csv_file_path_late, transform=custom_transform)

if len(dataset) == 0:
    print("No valid samples found in the dataset. Please check your file paths and CSV files.")
else:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)  # Batch size set to 1 for simplicity

    for idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        (images_early, images_late), labels = batch
        print(f'Batch {idx+1}:')
        print('Early Image Tensor:', images_early)
        print('Late Image Tensor:', images_late)
        print('Label Tensor:', labels)
        break  # Print only the first batch for demonstration purposes

    print('Data transformation complete.')

