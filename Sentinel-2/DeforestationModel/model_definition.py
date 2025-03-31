import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ResNet34MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the ResNet-34 backbone with pretrained weights
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 2 input channels (B4, B8)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,  # 2 input channels
            out_channels=64,  # Default number of filters in ResNet-34
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        torch.nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Replace the global average pooling layer to preserve spatial dimensions
        self.backbone.avgpool = nn.Identity()

        # Shared intermediate layer for efficiency
        self.shared_head = nn.Conv2d(
            in_channels=512,  # Final ResNet-34 output
            out_channels=256,  # Shared representation
            kernel_size=1
        )
        torch.nn.init.kaiming_normal_(self.shared_head.weight, mode="fan_out", nonlinearity="relu")

        # Classification head for forest loss/growth prediction
        self.classification_head = nn.Conv2d(
            in_channels=256,  # From shared head
            out_channels=3,   # Ternary classification (3 channels for loss, stable, gain)
            kernel_size=1
        )
        torch.nn.init.kaiming_normal_(self.classification_head.weight, mode="fan_out", nonlinearity="relu")

        # Regression head for NDVI change prediction
        self.regression_head = nn.Conv2d(
            in_channels=256,  # From shared head
            out_channels=1,  # Output 1 continuous value (NDVI difference) per pixel
            kernel_size=1
        )
        torch.nn.init.kaiming_normal_(self.regression_head.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # Store original input size for dynamic upsampling
        original_size = (x.size(2), x.size(3))  # Height and width of the input image

        # ResNet-34 forward pass (shared backbone)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Shared feature representation
        shared_features = self.shared_head(x)

        # Classification output (softmax for ternary classification probabilities)
        classification_output = self.classification_head(shared_features)
        classification_output = F.softmax(classification_output, dim=1)  # Predict probabilities for 3 classes

        # Regression output (NDVI differences)
        regression_output = self.regression_head(shared_features)

        # Dynamically upsample outputs to match input size
        classification_output = F.interpolate(classification_output, size=original_size, mode='bilinear', align_corners=True)
        regression_output = F.interpolate(regression_output, size=original_size, mode='bilinear', align_corners=True)

        return classification_output, regression_output




