# custom_transform.py

from torchvision.transforms import functional as F
import torch
import numpy as np

class CustomTransform:
    def __init__(self, resize=(224, 224), normalize=True, augment=False):
        self.resize = resize
        self.normalize = normalize
        self.augment = augment

    def __call__(self, img_pair):
        img_early, img_late = img_pair

        # Ensure consistent augmentation
        if self.augment:
            seed = np.random.randint(2147483647)
            img_early = self.transform_image(img_early, seed)
            img_late = self.transform_image(img_late, seed)
        else:
            img_early = self.transform_image(img_early)
            img_late = self.transform_image(img_late)

        return img_early, img_late

    def transform_image(self, img, seed=None):
        # Data augmentation
        if self.augment and seed is not None:
            np.random.seed(seed)
            if np.random.random() > 0.5:
                img = img[:, :, ::-1]  # Horizontal flip for multi-channel
            if np.random.random() > 0.5:
                img = img[:, ::-1, :]  # Vertical flip for multi-channel
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                img = self.rotate_image(img, angle)  # Rotate for multi-channel

        img = torch.tensor(img, dtype=torch.float32)
        img = F.resize(img, self.resize)

        if self.normalize:
            num_channels = img.shape[0]
            mean = torch.tensor([0.485] * num_channels)
            std = torch.tensor([0.229] * num_channels)
            img = (img - mean[:, None, None]) / std[:, None, None]

        return img

    def rotate_image(self, img, angle):
        # Helper function to rotate multi-channel image
        rotated = torch.tensor(np.empty_like(img))
        for i in range(img.shape[2]):
            rotated[:, :, i] = F.rotate(torch.tensor(img[:, :, i]).unsqueeze(0), angle)
        return rotated.numpy()


