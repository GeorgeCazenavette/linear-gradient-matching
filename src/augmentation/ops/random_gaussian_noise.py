import torch
import torch.nn as nn


class RandomGaussianNoise(nn.Module):
    """GPU-accelerated differentiable Gaussian noise augmentation

    Adds Gaussian noise to images with per-sample application probability.
    The noise generation and application are both differentiable.

    Args:
        mean: Mean of the Gaussian distribution (default: 0.0)
        std: Standard deviation of the Gaussian distribution (default: 1.0)
        p: Probability of applying noise to each image (default: 0.5)
    """

    def __init__(self, mean=0.0, std=1.0, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, images):
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Generate application decisions on GPU
        with torch.no_grad():
            apply_noise = torch.rand(B, device=device) < self.p

        # Only generate noise if at least one image needs it
        if apply_noise.any():
            # Generate Gaussian noise for all images (differentiable)
            noise = torch.randn(B, C, H, W, device=device, dtype=dtype)
            noise = noise * self.std + self.mean

            # Create mask for broadcasting [B, 1, 1, 1]
            mask = apply_noise.float().view(B, 1, 1, 1)

            # Apply noise only to selected images (differentiable)
            images = images + mask * noise

        return images
