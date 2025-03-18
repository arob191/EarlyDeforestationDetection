import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn.functional as F

def get_resnet101_model(pretrained_path=None):
    """
    Constructs and customizes a ResNet-101 model for pixel-wise deforestation predictions.
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

    # Replace the global average pooling layer with an identity layer to preserve spatial dimensions
    resnet101_model.avgpool = nn.Identity()

    # Replace the fully connected (fc) layer with a 1x1 convolution for pixel-wise predictions
    resnet101_model.fc = nn.Conv2d(
        in_channels=resnet101_model.fc.in_features,
        out_channels=1,  # Single output channel (deforestation risk)
        kernel_size=1
    )

    # Overwrite the _forward_impl method to prevent flattening
    def _forward_impl(self, x):
        # Standard forward pass up to the final fc layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Identity layer; no pooling happens here
        x = self.fc(x)  # Pass through the 1x1 convolution

        # Upsample the output to match the target size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x


    # Replace the default _forward_impl with the custom one
    resnet101_model._forward_impl = _forward_impl.__get__(resnet101_model, type(resnet101_model))

    # Load pretrained weights, excluding the fc layer (if provided)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        resnet101_model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded from {pretrained_path}")

    return resnet101_model


