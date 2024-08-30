import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VFELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VFELayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # Max pooling along the feature dimension
        x_repeat = x_max.repeat(1, x.size(1), 1)  # Repeat max value to match the original dimensions
        x = torch.cat([x, x_repeat], dim=2)  # Concatenate along the feature dimension
        return x

class VFE(nn.Module):
    def __init__(self, input_dim=5, num_vfe_layers=2, vfe_dim=64):
        super(VFE, self).__init__()
        self.vfe_layers = nn.ModuleList(
            [VFELayer(input_dim, vfe_dim)] +  # First layer
            [VFELayer(vfe_dim * 2, vfe_dim) for _ in range(num_vfe_layers - 1)]  # Subsequent layers
        )
        self.fc = nn.Linear(vfe_dim * 2, vfe_dim)  # Final linear layer
        self.bn = nn.BatchNorm1d(vfe_dim)  # Batch normalization

    def forward(self, voxel_features):
        mask = (voxel_features[:, :, 0] != 0)  # Mask to exclude empty voxels
        x = voxel_features

        for vfe_layer in self.vfe_layers:
            x = vfe_layer(x)

        x = F.relu(self.bn(self.fc(x)))  # Apply final linear layer and batch normalization

        # Apply mask and reduce
        x = x * mask.unsqueeze(2).float()  # Apply mask to the feature tensor
        x_max = torch.max(x, dim=1)[0]  # Max pooling to reduce to (batch_size, output_dim)
        return x_max

def voxelize_and_encode(data, voxel_size=0.5, max_points_per_voxel=35):
    """
    Voxelize the input point cloud and prepare features for VoxelNet.

    Args:
    - points (np.ndarray): Input point cloud of shape (N, 4) where N is the number of points.
    - voxel_size (float): Size of each voxel.
    - max_points_per_voxel (int): Maximum number of points allowed per voxel.

    Returns:
    - voxel_features (np.ndarray): Feature vector for each voxel.
    - voxel_coords (np.ndarray): The grid coordinates of each voxel.
    """
    voxel_grid = {}

    # Extract features: x, y, z, intensity
    for point in data:
        x, y, z, intensity = point
        voxel_coord = tuple((point[:3] / voxel_size).astype(int))
        if voxel_coord not in voxel_grid:
            voxel_grid[voxel_coord] = []
        voxel_grid[voxel_coord].append(point)

    voxel_features = []
    voxel_coords = []

    for coord, pts in voxel_grid.items():
        if len(pts) > max_points_per_voxel:
            pts = pts[:max_points_per_voxel]

        pts = np.array(pts)

        # Calculate distance to sensor
        distances = np.linalg.norm(pts[:, :3], axis=1)

        # Compute mean x, y, z, intensity, and distance
        mean_x, mean_y, mean_z = np.mean(pts[:, :3], axis=0)
        mean_intensity = np.mean(pts[:, 3])
        mean_distance = np.mean(distances)

        feature_vector = np.array([mean_x, mean_y, mean_z, mean_intensity, mean_distance])
        voxel_features.append(feature_vector)
        voxel_coords.append(coord)

    return np.array(voxel_features), np.array(voxel_coords)
