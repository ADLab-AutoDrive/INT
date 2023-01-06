import torch
import torch.nn as nn
from torch.nn import functional as F
from ...ops.voxel import Voxelization
from ..registry import VOXELIZATION


@VOXELIZATION.register_module
class BasicVoxelization(nn.Module):

    def __init__(self, voxel_size, pc_range, max_points_in_voxel, max_voxel_num, ):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size, pc_range, max_points_in_voxel, max_voxel_num)
        self.voxel_size = voxel_size
        self.grid_size = self.voxel_layer.grid_size.cpu().detach().numpy()
        # self.pcd_shape = self.voxel_layer.pcd_shape

    def forward(self, points, batch_size):
        """Apply hard voxelization to points."""
        bs_idx, points = points[:, 0], points[:, 1:]
        voxels, coors, num_points = [], [], []
        for bs_cnt in range(batch_size):
            one_point = points[bs_idx == bs_cnt]
            res_voxels, res_coors, res_num_points = self.voxel_layer(one_point)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        return voxels, num_points, coors_batch, self.grid_size
