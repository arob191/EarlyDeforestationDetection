import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class DeforestationResNet(nn.Module):
    def __init__(self, num_seg_classes=3, pretrained=True, use_distance=True):
        """
        Args:
            num_seg_classes (int): Number of segmentation classes.
            pretrained (bool): If True, load ImageNet pretrained weights using the new API.
            use_distance (bool): Whether to expect an extra 1-channel distance map (resulting in a 4-channel input).
        """
        super(DeforestationResNet, self).__init__()
        self.use_distance = use_distance
        input_channels = 4 if self.use_distance else 3

        # Use the new weights API.
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        
        # If using an additional distance channel, adjust the first convolutional layer.
        if input_channels != 3:
            old_conv = resnet.conv1  # Original conv layer: (64, 3, 7, 7)
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Initialize new_conv weights: copy existing weights for the first 3 channels and replicate for extra channels.
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight[:, :1].clone()
            resnet.conv1 = new_conv

        # Remove the final fully-connected layer.
        resnet.fc = nn.Identity()
        self.resnet = resnet

        # Build the segmentation head.
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_seg_classes, kernel_size=1)
        )

        # Build the regression head.
        self.reg_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, ndvi_input, distance=None):
        """
        Args:
            ndvi_input (Tensor): Tensor of shape [N, 3, H, W] - the NDVI difference image.
            distance (Tensor): Tensor of shape [N, 1, H, W] - the distance map (required if use_distance is True).
        Returns:
            seg_output (Tensor): Segmentation output of shape [N, num_seg_classes, H, W].
            reg_output (Tensor): Regression output of shape [N, 1].
        """
        if self.use_distance:
            if distance is None:
                raise ValueError("Model expects a distance map, but got None.")
            # Concatenate the NDVI difference image and distance map along the channel dimension.
            x = torch.cat([ndvi_input, distance], dim=1)  # Shape: [N, 4, H, W]
        else:
            x = ndvi_input

        # Forward through the ResNet backbone.
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        features = self.resnet.layer4(x)  # [N, 2048, H/32, W/32]

        # Segmentation branch.
        seg_logits = self.seg_head(features)  # [N, num_seg_classes, H/32, W/32]
        seg_output = F.interpolate(seg_logits, scale_factor=32, mode="bilinear", align_corners=False)

        # Regression branch.
        pooled = self.resnet.avgpool(features)  # [N, 2048, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [N, 2048]
        reg_output = self.reg_head(pooled)         # [N, 1]

        return seg_output, reg_output

# Simple test for the model.
if __name__ == '__main__':
    # Create dummy inputs: NDVI input (3 channels) and distance map (1 channel).
    ndvi_input = torch.randn(2, 3, 224, 224)
    distance = torch.randn(2, 1, 224, 224)
    
    model = DeforestationResNet(num_seg_classes=3, pretrained=False, use_distance=True)
    seg_out, reg_out = model(ndvi_input, distance)
    print("Segmentation output shape:", seg_out.shape)  # Expected: [2, 3, 224, 224]
    print("Regression output shape:", reg_out.shape)      # Expected: [2, 1]