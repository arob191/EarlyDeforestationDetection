import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.models import resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet-101 model with pixel-wise output
def get_resnet101_model():
    """
    Constructs and customizes a ResNet-101 model for pixel-wise deforestation predictions.

    Returns:
    torch.nn.Module: The customized ResNet-101 model.
    """
    resnet101_model = resnet101()
    original_conv1 = resnet101_model.conv1
    resnet101_model.conv1 = torch.nn.Conv2d(
        in_channels=2,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )
    torch.nn.init.kaiming_normal_(resnet101_model.conv1.weight, mode="fan_out", nonlinearity="relu")

    resnet101_model.fc = torch.nn.Conv2d(
        in_channels=resnet101_model.fc.in_features,
        out_channels=1,  # Single output channel (deforestation risk)
        kernel_size=1
    )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Identity layer
        x = self.fc(x)  # 1x1 convolution
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # Resize output to match target
        return x


    resnet101_model._forward_impl = _forward_impl.__get__(resnet101_model, type(resnet101_model))
    return resnet101_model

def load_model(model_path):
    """
    Loads a trained ResNet-101 model from the specified path.

    Parameters:
    model_path (str): Path to the model weights file.

    Returns:
    torch.nn.Module: The loaded model.
    """
    model = get_resnet101_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Avoid FutureWarning
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def resize_tensor(image_tensor, target_size):
    """
    Resizes a tensor using PyTorch's interpolation.
    """
    return F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)

def predict_deforestation(model, image_path, target_size=(256, 256)):
    """
    Generates a heat map of deforestation predictions for a given image.

    Parameters:
    model (torch.nn.Module): The trained model.
    image_path (str): Path to the input image.
    target_size (tuple): Target size to resize the image.

    Returns:
    np.ndarray: Heat map of deforestation risk.
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # Read all bands
    b4, b8 = image[0, :, :], image[3, :, :]  # Extract B4 (Red) and B8 (NIR)
    image_2channel = np.stack([b4, b8], axis=0).astype("float32") / 10000.0
    image_tensor = torch.tensor(image_2channel).unsqueeze(0).to(device)  # Add batch dimension
    image_tensor = resize_tensor(image_tensor, target_size)
    
    print(f"Image tensor shape: {image_tensor.shape}")  # Debug shape
    
    with torch.no_grad():
        predictions = model(image_tensor)
        print(f"Raw predictions shape: {predictions.shape}")
        predictions = predictions.squeeze().cpu().numpy()  # Remove batch dimension
        print(f"Processed predictions shape: {predictions.shape}")
    return predictions

def plot_heatmap(heat_map, image_path):
    """
    Plots a heat map overlaid on the original image.

    Parameters:
    heat_map (np.ndarray): Heat map of deforestation risk.
    image_path (str): Path to the original image.

    Returns:
    None
    """
    if len(heat_map.shape) != 2:
        raise ValueError(f"Expected a 2D array for heat_map, but got shape {heat_map.shape}")

    # Load the original satellite image
    with rasterio.open(image_path) as src:
        image = src.read(1)  # Use the first band for visualization
        image_height, image_width = image.shape

    print(f"Original image shape: {image.shape}")  # Debug
    print(f"Heat map shape before resizing: {heat_map.shape}")  # Debug

    # Resize heat map to match the satellite image dimensions
    heat_map_resized = torch.nn.functional.interpolate(
        torch.tensor(heat_map).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=(image_height, image_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0).numpy()  # Remove extra dimensions

    print(f"Heat map shape after resizing: {heat_map_resized.shape}")  # Debug

    # Plot the original satellite image with the heat map overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray", interpolation="none")  # Display the satellite image
    plt.imshow(heat_map_resized, cmap="hot", alpha=0.5, interpolation="none")  # Overlay the resized heat map
    plt.colorbar(label="Deforestation Risk")
    plt.title("Deforestation Heat Map")
    plt.axis("off")
    plt.show()

# Example usage
image_path = "E:/Sentinelv3/Fazenda Forest/Fazenda_Manna_2015_2016/Fazenda_Manna_2015_2016_Tile_027.tif"
model_path = "deforestation_model_resnet101.pth"

model = load_model(model_path)
heat_map = predict_deforestation(model, image_path, target_size=(256, 256))
plot_heatmap(heat_map, image_path)

