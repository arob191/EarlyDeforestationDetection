# main.py

import torch
from torch.utils.data import DataLoader
from dataset_class import GeoTIFFChangeDetectionDataset  # Import the modified dataset class

# Define the full paths to your CSV files
csv_file_path_early = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2015_2016.csv'
csv_file_path_late = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2017_2018.csv'

# Create the dataset
dataset = GeoTIFFChangeDetectionDataset(csv_file_early=csv_file_path_early, csv_file_late=csv_file_path_late, resize=(224, 224), normalize=True)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size set to 1 for simplicity

# Iterate through the DataLoader and print the tensors
for idx, batch in enumerate(dataloader):
    (images_early, images_late), ground_truth = batch
    print(f'Batch {idx+1}:')
    print('Early Image Tensor:', images_early)
    print('Late Image Tensor:', images_late)
    break  # Print only the first batch for demonstration purposes

print('Data transformation complete.')



