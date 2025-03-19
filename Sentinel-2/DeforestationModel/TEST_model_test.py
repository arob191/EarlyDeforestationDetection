import torch
from model_definition import get_resnet101_model

# Create the model
model = get_resnet101_model()

# Test on a dummy input
dummy_input = torch.randn(1, 2, 256, 256)  # Batch size 1, 2 channels (B4, B8), 256x256 pixels
output = model(dummy_input)
print(output.shape)  # Output shape should be [1, 1, 256, 256] (binary prediction per pixel)