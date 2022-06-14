# This file is derived from [NeuralRecon](https://github.com/zju3dv/NeuralRecon).
# Originating Author: Yiming Xie
# Modified for [PlanarRecon](https://github.com/neu-vi/PlanarRecon) by Yiming Xie.

# Original header:
# Copyright SenseTime. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchsparse import PointTensor
from utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GRUFusion(nn.Module):
    """
    GRU Fusion module as in the NeuralRecon (https://zju3dv.github.io/neuralrecon/). 
    Update hidden state features with ConvGRU.
    """

    def __init__(self, cfg, ch_in=None):
        super(GRUFusion, self).__init__()
        self.cfg = cfg

        # features
        self.ch_in = ch_in
        self.feat_init = 0

        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None, None, None]
        self.global_origin = [None, None, None]
        self.global_volume = [None, None, None]
        self.target_label_volume = [None, None, None]

        self.fusion_nets = nn.ModuleList()
        for i, ch in enumerate(ch_in):
            self.fusion_nets.append(ConvGRU(hidden_dim=ch,
                                            input_dim=ch,
                                            pres=1,
                                            vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_label_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()

    def convert2dense(self, current_coords, current_values, coords_target_global, label_target, relative_origin,
                      scale):
        '''
        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth label.

        :param current_coords: (Tensor), current coordinates, (N, 3)
        :param current_values: (Tensor), current features/label, (N, C)
        :param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        :param label_target: (Tensor), plane label ground truth, (N',)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        :return: current_volume: (Tensor), current dense feature/label volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: global_volume: (Tensor), global dense feature/label volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: target_volume: (Tensor), dense target label volume, (DIM_X, DIM_Y, DIM_Z, 1)
        :return: valid: mask: 1 represent in current FBV (N,)
        :return: valid_target: gt mask: 1 represent in current FBV (N,)
        '''
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_label_target = self.target_label_volume[scale].F
        global_coords_target = self.target_label_volume[scale].C

        dim = (torch.tensor(self.cfg.N_VOX, device=current_coords.device) // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        # sparse to dense
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[scale],
                                                self.feat_init, global_value.device)

        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[scale],
                                                 self.feat_init, global_value.device)

        # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
        updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))

        # fuse ground truth
        if label_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current label and global label
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]

            label_target = torch.cat([global_label_target[valid_target], label_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, label_target, dim_list, 1, -1,
                                                    label_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target

    def update_map(self, value, coords, target_volume, valid, valid_target,
                   relative_origin, scale):
        '''
        Replace Hidden state/label in global Hidden state/label volume by direct substitute corresponding voxels
        :param value: (Tensor) fused feature (N, C)
        :param coords: (Tensor) updated coords (N, 3)
        :param target_volume: (Tensor) label volume (DIM_X, DIM_Y, DIM_Z, 1)
        :param valid: (Tensor) mask: 1 represent in current FBV (N,)
        :param valid_target: (Tensor) gt mask: 1 represent in current FBV (N,)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return:
        '''
        # pred
        self.global_volume[scale].F = torch.cat(
            [self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_label_volume[scale].F = torch.cat(
                [self.target_label_volume[scale].F[valid_target == False],
                 target_volume[target_volume != -1].unsqueeze(-1)])

            target_coords = torch.nonzero(target_volume != -1) + relative_origin

            self.target_label_volume[scale].C = torch.cat(
                [self.target_label_volume[scale].C[valid_target == False], target_coords])

    def forward(self, coords, values_in, inputs, scale=2, outputs=None):
        '''
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param values_in: (Tensor), features/label, (N, C)
        :param inputs: dict: meta data from dataloader
        :param scale:
        :param outputs:
        :return: updated_coords_all: (Tensor), updated coordinates, (N', 4) (4 : Batch ind, x, y, z)
        :return: updated_r_coords_all: (Tensor), updated r coordinates, (N', 4) (4 : Batch ind, x, y, z)
        :return: values_all: (Tensor), features after gru fusion, (N', C)
        :return: label_target_all: (Tensor), label ground truth, (N', 1)
        :return: occ_target_all: (Tensor), occupancy ground truth, (N', 1)
        '''
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)

        label_target_all = None
        occ_target_all = None
        values_all = None
        updated_coords_all = None
        updated_r_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
            scene = inputs['scene'][i]  # scene name
            global_origin = inputs['vol_origin'][i]  # origin of global volume
            origin = inputs['vol_origin_partial'][i]  # origin of part volume

            # if this fragment is from new scene, we reinitialize backend map
            if self.scene_name[scale] is None or scene != self.scene_name[scale]:
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.cfg.VOXEL_SIZE * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.long()

            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                continue
            coords_b = coords[batch_ind, 1:].long() // interval
            values = values_in[batch_ind]

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.cfg.N_LAYER - scale - 1][i]
                label_target = inputs['label_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]
                coords_target = torch.nonzero(occ_target)
            else:
                coords_target = label_target = None

            # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
            # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target = self.convert2dense(
                coords_b,
                values,
                coords_target,
                label_target,
                relative_origin,
                scale)

            # dense to sparse: get features using new feature coordinates (updated_coords)
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            # get fused gt
            if target_volume is not None:
                label_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                occ_target = label_target >= 0
            else:
                label_target = occ_target = None

            # convert to aligned camera coordinate
            r_coords = updated_coords.detach().clone().float()
            r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
            r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
            r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
            r_coords = torch.cat([r_coords, torch.zeros((1, r_coords.shape[-1]), device=r_coords.device)])
            r_coords = r_coords.permute(1, 0).contiguous()

            h = PointTensor(global_values, r_coords)
            x = PointTensor(values, r_coords)

            values = self.fusion_nets[scale](h, x)
            r_coords[:, 3] = i

            # feed back to global volume (direct substitute)
            self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale)

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                                dim=1)
                updated_r_coords_all = r_coords
                values_all = values
                label_target_all = label_target
                occ_target_all = occ_target
            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                            dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                    
                values_all = torch.cat([values_all, values])
                if r_coords is not None:
                    updated_r_coords_all = torch.cat([updated_r_coords_all, r_coords])

                if label_target_all is not None:
                    label_target_all = torch.cat([label_target_all, label_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

        return updated_coords_all, updated_r_coords_all, values_all, label_target_all, occ_target_all
