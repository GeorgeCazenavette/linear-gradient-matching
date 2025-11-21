import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomResizedCrop(nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def forward(self, images):
        B, C, H, W = images.shape
        device = images.device

        with torch.no_grad():
            rand_vals = torch.rand(B, 4, device=device)

            # Sample scale uniformly in [scale[0], scale[1]]
            scales = rand_vals[:, 0] * (self.scale[1] - self.scale[0]) + self.scale[0]

            # Sample log ratio uniformly, then exp
            ratios = torch.exp(
                rand_vals[:, 1] * (self.log_ratio[1] - self.log_ratio[0])
                + self.log_ratio[0]
            )

            # Compute crop dimensions as fractions of original image
            scale_w = torch.sqrt(scales * ratios)
            scale_h = torch.sqrt(scales / ratios)

            # Sample top-left corner uniformly in valid range (independent x and y)
            x1 = rand_vals[:, 2] * (1 - scale_w)
            y1 = rand_vals[:, 3] * (1 - scale_h)

            # Convert to affine transform offsets (center of crop in [-1, 1] coords)
            offset_x = (x1 + scale_w / 2 - 0.5) * 2
            offset_y = (y1 + scale_h / 2 - 0.5) * 2

        # Build affine matrix
        theta = torch.zeros(B, 2, 3, dtype=images.dtype, device=device)
        theta[:, 0, 0] = scale_w
        theta[:, 1, 1] = scale_h
        theta[:, 0, 2] = offset_x
        theta[:, 1, 2] = offset_y

        grid = F.affine_grid(theta, (B, C, *self.size), align_corners=False)
        return F.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
