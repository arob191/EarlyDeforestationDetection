# Lamar Undergrad Research
# Name: Austin Robertson  
# Date: 10/01/2024
# Design and train a CNN on forest that have been deforested and fores that haven't

import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from skimage import io, img_as_ubyte
from PIL import Image

# # Custom Dataset to handle .tif images
# class CustomImageDataset(datasets.ImageFolder):
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         # Use skimage to read the image
#         img = io.imread(path)
        
#         # Normalize the float image data to the range [0, 1]
#         if img.dtype == np.float32 or img.dtype == np.float64:
#             img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
#         # Convert to uint8
#         img = img_as_ubyte(img)
#         img = Image.fromarray(img)
        
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target

# # Prepare your data transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load your dataset
# train_dataset = CustomImageDataset(r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\DataSet', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Define your CNN model
# class DeforestationModel(nn.Module):
#     def __init__(self):
#         super(DeforestationModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 56 * 56, 512)
#         self.fc2 = nn.Linear(512, 2)  # Assuming binary classification: deforested or not

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 56 * 56)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Initialize the model, loss function, and optimizer
# model = DeforestationModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# print('Training complete.')

import numpy as np
import tifffile as tiff
import torch

def load_tiff_to_numpy(image_path):
    # Open the TIFF image
    img_array = tiff.imread(image_path)

    # Handle NaNs
    img_array = np.nan_to_num(img_array, nan=0.0)

    # Normalize and clip values to the 0-255 rang
    img_array = np.clip(img_array, 0, 255)

    # Convert the array to uint8
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    return img_array

def numpy_to_tensor(np_array):
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(np_array)
    
    return tensor

# Import image
image_path = r'E:\DataSet\not_deforested\Landsat8_SamHouston_2013-6-14.tif'
np_array = load_tiff_to_numpy(image_path)
print("Numpy Array Shape:", np_array.shape)

# Test convertion to tensor
tensor = numpy_to_tensor(np_array)
print("Tensor Shape:", tensor.shape)





