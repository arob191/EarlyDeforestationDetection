import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50MultiTask(nn.Module):
    def __init__(self, in_channels=4):  # Adjusted for additional input channels (e.g., B4, B8, Distance Map)
        super().__init__()
        # Load the ResNet-50 backbone with pretrained weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to accept custom input channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,  # Input channels (e.g., B4, B8, and distance map)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        torch.nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Replace the global average pooling layer to preserve spatial dimensions
        self.backbone.avgpool = nn.Identity()

        # Deeper shared intermediate layer with residual connections
        self.shared_head = ResidualSharedHead(2048, 1024, 512)  # Increase mid_channels and set out_channels to 512

        # Classification head for forest loss/growth prediction
        self.classification_head = nn.Conv2d(
        in_channels=512,  # Matches the shared head output
        out_channels=3,   # Ternary classification (loss, stable, gain)
        kernel_size=1
        )
        torch.nn.init.kaiming_normal_(self.classification_head.weight, mode="fan_out", nonlinearity="relu")

        # Regression head for NDVI change prediction
        self.regression_head = nn.Conv2d(
            in_channels=512,  # Matches the shared head output
            out_channels=1,  # Single NDVI regression output
            kernel_size=1
        ) 
        torch.nn.init.kaiming_normal_(self.regression_head.weight, mode="fan_out", nonlinearity="relu")

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Store the original input size for dynamic upsampling
        original_size = (x.size(2), x.size(3))  # Height and width of the input image

        # ResNet-50 forward pass
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Shared feature representation with deeper residual head
        shared_features = self.shared_head(x)
        shared_features = self.dropout(shared_features)  # Apply dropout

        # Classification output (log-softmax for stability)
        classification_output = self.classification_head(shared_features)
        classification_output = F.log_softmax(classification_output, dim=1)

        # Regression output (tanh for normalized NDVI differences)
        regression_output = torch.tanh(self.regression_head(shared_features))

        # Dynamically upsample outputs to match input size
        classification_output = F.interpolate(classification_output, size=original_size, mode='bilinear', align_corners=True)
        regression_output = F.interpolate(regression_output, size=original_size, mode='bilinear', align_corners=True)

        return classification_output, regression_output

class ResidualSharedHead(nn.Module):
    """
    Shared intermediate layer with deeper residual connections.
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        # First residual block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # Second residual block
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # Projection for residual connections
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.project(x)  # Project identity to match out_channels

        # Apply the first residual block
        out = self.block1(x)
        out += identity  # Residual connection
        out = self.relu(out)

        # Apply the second residual block
        identity = out  # Store intermediate output as new identity
        out = self.block2(out)
        out += identity  # Residual connection
        return self.relu(out)