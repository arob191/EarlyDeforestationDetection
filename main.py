# Lamar Undergrad Research
# Name: Austin Robertson  
# Date: 10/01/2024
# Design and train a CNN to detect future deforestation hotspots

import torch
import numpy as np
import tifffile as tiff
import DatasetClass as ds

# # Prepare your data transformations

# # Define your CNN model

# # Initialize the model, loss function, and optimizer

# # Training loop


# def load_tiff_to_numpy(image_path):
#     # Open the TIFF image
#     img_array = tiff.imread(image_path)

#     # Handle NaNs
#     img_array = np.nan_to_num(img_array, nan=0.0)

#     # Normalize and clip values to the 0-255 rang
#     img_array = np.clip(img_array, 0, 255)

#     # Convert the array to uint8
#     if img_array.dtype != np.uint8:
#         img_array = img_array.astype(np.uint8)
    
#     return img_array

# # Import image
# image_path = r'E:\DataSet\\not_deforested\\Landsat8_SamHouston_2013-6-14.tif'
# np_array = load_tiff_to_numpy(image_path)
# print("Numpy Array Shape:", np_array.shape)

csv_file_path = 'E:\labels.csv'
transform = None
dataset = ds.DeforestationDataset(csv_file=csv_file_path, transform=transform)


# Test convertion to tensor
for i in range(len(dataset)):
    before_image, after_image, label = dataset[i]
    print("Before image shape:", before_image.shape)
    print("After image shape:", after_image.shape)
    print("Label:", label)





