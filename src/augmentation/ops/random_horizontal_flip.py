import torch
import torch.nn as nn


class RandomHorizontalFlip(nn.Module):
    """GPU-accelerated differentiable horizontal flip

    Uses per-sample binary decisions (not blending).
    Preserves gradients through selective indexing and cloning.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        B = images.shape[0]
        device = images.device

        # Generate flip decisions on GPU
        with torch.no_grad():
            flip_decisions = torch.rand(B, device=device) < self.p

        # Only flip if at least one image needs flipping (optimization)
        if flip_decisions.any():
            # Clone to avoid in-place modification issues with autograd
            images = images.clone()
            # Flip only the selected images
            images[flip_decisions] = torch.flip(images[flip_decisions], dims=[3])

        return images
