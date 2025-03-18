import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

def get_resnet101_model(pretrained_path=None):
    """
    Constructs and customizes a ResNet-101 model for scalar deforestation predictions.
    """
    # Load the ResNet-101 model with pretrained weights
    resnet101_model = resnet101(weights=ResNet101_Weights.DEFAULT)

    # Modify the first convolutional layer to accept 2 input channels (B4, B8)
    original_conv1 = resnet101_model.conv1
    resnet101_model.conv1 = nn.Conv2d(
        in_channels=2,  # Use 2 channels (B4 and B8)
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )
    torch.nn.init.kaiming_normal_(resnet101_model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Keep the global average pooling layer for scalar output predictions
    # This reduces the spatial dimensions to 1x1 before passing to the fully connected layer.
    resnet101_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Replace the fully connected (fc) layer with a Linear layer for scalar predictions
    num_features = resnet101_model.fc.in_features
    resnet101_model.fc = nn.Linear(num_features, 1)  # Single scalar output

    # Load pretrained weights, excluding the fc layer (if provided)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        resnet101_model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded from {pretrained_path}")

    return resnet101_model


