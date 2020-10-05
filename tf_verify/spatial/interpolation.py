"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


from typing import List

import torch
import torch.nn.functional as F


def interpolate(images: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
    """
    Computes the composition of the images with the vector fields by
    means of bilinear interpolation
    :param images: shape (B, C, H, W) containing pixel values
    :param flows: shape (B, H, W, 2) containing flows in x-/y-direction
    :return: shape (B, C, H, W) containing deformed pixel values
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size, channels, height, width = images.shape

    h_range = torch.arange(height).to(device)
    w_range = torch.arange(width).to(device)
    grid_x = w_range.repeat(height, 1).unsqueeze(2)
    grid_y = h_range.view(-1, 1).repeat(1, width).unsqueeze(2)
    grid = torch.cat((grid_x, grid_y), 2).double()

    # Grid normalized to [-1,1]^2
    scale = torch.div(
        torch.full(size=[2], fill_value=2.),
        torch.tensor(data=[width - 1., height - 1.], dtype=torch.float)
    ).to(device)
    grid = scale * grid - 1
    deformed_grid = torch.add(
        grid.repeat(batch_size, 1, 1, 1), torch.mul(scale, flows)
    )

    # avoid segmentation faults
    assert not torch.isnan(deformed_grid).any()

    return F.grid_sample(
        images, deformed_grid, padding_mode='border', align_corners=True
    )


def compute_grid(shape: List[int]) -> torch.Tensor:
    """
    Computes the normalized and centered grid coordinates for an image
    :param shape: contains batch_size, height and width of grid
    :return: shape (B, 2, H, W) containing grid coordinates
    """
    batch_size, height, width = shape

    h_range = torch.arange(height)
    w_range = torch.arange(width)
    grid_x = w_range.repeat(height, 1).unsqueeze(0)
    grid_y = h_range.view(-1, 1).repeat(1, width).unsqueeze(0)

    return torch.cat(
        (2 * grid_x - width + 1, height - 2 * grid_y - 1), 0
    ).repeat(batch_size, 1, 1, 1)
