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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from tools.bin_mean_shift import Bin_Mean_Shift


class FragNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''
 
    def __init__(self, cfg):
        super(FragNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 1 + 9, 48 + 24 * alpha + 1 + 9, 24 + 24 + 1 + 9]
        channels = [96, 48, 24]

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion: follow the network design in NeuralRecon
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict occupancy and plane.
        self.occ_preds = nn.ModuleList()
        self.plane_class = nn.ModuleList()
        self.plane_residual = nn.ModuleList()
        self.plane_distance = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.occ_preds.append(nn.Linear(channels[i], 1))
            self.plane_class.append(nn.Linear(channels[i], 7))
            self.plane_distance.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i], bias=True),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(),
                    nn.Linear(channels[i], 1, bias=True)
                )
            )
            self.plane_residual.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i], bias=True),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(),
                    nn.Linear(channels[i], 7 * 3, bias=True)
                ))

        self.offset_center_preds = nn.Sequential(
            nn.Linear(channels[self.n_scales], channels[self.n_scales], bias=True),
            nn.BatchNorm1d(channels[self.n_scales]),
            nn.ReLU(),
            nn.Linear(channels[self.n_scales], 3, bias=True)
        )

        self.normal_anchors = torch.from_numpy(np.load(self.cfg.NORMAL_ANCHOR_PATH)).float()
        self.normal_anchors = torch.nn.parameter.Parameter(self.normal_anchors, requires_grad=False)
        self.mean_shift = Bin_Mean_Shift(device='cuda')

    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['label_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: label_target: (Tensor), label ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            label_target = inputs['label_list'][scale]
            occ_target = inputs['occ_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            label_target = label_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

            return label_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, features, inputs, outputs):
        '''

        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords_':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'feat':                     (Tensor), voxel features,
                                    (number of voxels, 24)
            'embedding':                (Tensor), planes normal (, 3), center_points (, 3), planes dis (, 1), pre_occ (, 1)
                                    (number of voxels, 8)
            others: target info.
        }
        :return: loss_dict: dict: {
            'multi_level_loss_X':         (Tensor), multi level loss
        }
        '''
        bs = features[0][0].shape[0]
        if "plane_anchors" in inputs.keys():
            anchors_gt = inputs['plane_anchors']
            residual_gt = inputs['residual']
            planes_gt = inputs['planes_trans']
            mean_xyz_gt = inputs['mean_xyz']
        else:
            anchors_gt = residual_gt = planes_gt = mean_xyz_gt = None

        pre_feat = None
        pre_coords = None
        outputs['label_target'] = []
        outputs['occ_target'] = []
        outputs['coords_'] = []
        loss_dict = {}
        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i

            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.cfg.N_VOX, interval)[0]
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)
            # ----back project----
            feats = torch.stack([feat[scale] for feat in features])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                                         KRcam)
            grid_mask = count > 1
            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.cfg.FUSION.FUSION_ON:
                label_target, occ_target = self.get_target(up_coords,
                                                            inputs,
                                                            scale)

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b, as_tuple=False).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)
            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                up_coords, r_coords, feat, label_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                grid_mask = torch.ones_like(feat[:, 0]).bool()
                if label_target is not None:
                    label_target = label_target.squeeze(1).long()
                    occ_target = occ_target.squeeze(1)
            # -----get occupancy and offset(for instance segmentation)----
            occ = self.occ_preds[i](feat)
            # class and regress plane
            class_logits = self.plane_class[i](feat)
            residuals = self.plane_residual[i](feat)
            residuals = residuals.view(-1, 7, 3)
            distance = self.plane_distance[i](feat)
            if i == self.n_scales:
                off_center = self.offset_center_preds(feat)
            else:
                off_center = None
            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]
            occupancy[grid_mask == False] = False

            num = int(occupancy.sum().data.cpu())

            # -------compute loss-------
            if anchors_gt is not None and self.training:
                loss = self.compute_loss(occ, class_logits, residuals, distance, off_center, occ_target, label_target,
                                        anchors_gt, residual_gt, planes_gt, mean_xyz_gt, r_coords,
                                        mask=grid_mask,
                                        pos_weight=self.cfg.POS_WEIGHT)
            else:
                loss = torch.Tensor([0])[0]
            loss_dict.update({f'multi_level_loss_{i}': loss})
            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                logger.info('larger points: scale {}'.format(i))
                choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy, as_tuple=False)
                occupancy[ind[choice]] = False

            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b, as_tuple=False).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict

            feat = feat[occupancy]
            pre_occ = occ[occupancy]
            pre_class = class_logits[occupancy]
            pre_distance = distance[occupancy]

            pre_feat = torch.cat([feat, pre_occ, pre_class, pre_distance], dim=1)

            output_coord_ = pre_coords.detach().clone()
            output_coord_[:, 1:] = output_coord_[:, 1:] // 2 ** scale
            outputs['coords_'].append(output_coord_.long())
            if anchors_gt is not None:
                outputs['label_target'].append(label_target[occupancy])
                outputs['occ_target'].append(occ_target[occupancy])
            if i == self.cfg.N_LAYER - 1:
                # ----convert class, residuals to normals -----
                class_probs = F.softmax(class_logits[occupancy], dim=-1)
                class_ids = class_probs.argmax(-1)
                idx = torch.arange(class_ids.shape[0], device=class_ids.device).long()
                residuals_pred = residuals[occupancy][idx, class_ids]
                normals = self.normal_anchors[class_ids] + residuals_pred
                offset_points = r_coords[occupancy, :3] + 0.12 * distance[occupancy] * normals / torch.norm(normals, dim=1,
                                                                                                     keepdim=True)
                D = -(offset_points.unsqueeze(1) @ normals.unsqueeze(2)).squeeze(1)
                planes = torch.cat([normals, D], dim=1)
                
                center_points = r_coords[occupancy, :3] + off_center[occupancy]
                center_points = torch.cat((center_points, torch.ones_like(center_points[:, :1])), dim=1)
                for b in range(bs):
                    # convert coordinate
                    ind = torch.nonzero(pre_coords[:, 0] == b, as_tuple=False).squeeze(1)
                    planes[ind] = (inputs['world_to_aligned_camera'][b].transpose(0,
                                                                            1) @ planes[ind].transpose(
                        0, 1)).transpose(0, 1)
                    if anchors_gt is not None:
                        planes_gt[b] = (inputs['world_to_aligned_camera'][b].transpose(0,
                                                                                    1) @ planes_gt[b].transpose(
                            0, 1)).transpose(0, 1)
                    
                    center_points[ind] = center_points[ind] @ torch.inverse(inputs['world_to_aligned_camera'])[b].permute(1, 0).contiguous()
                center_points = center_points[:, :3]

                # A,B,C,D,X,Y,Z,OCC for vote
                # planes = planes[:, :3] # / planes[:, 3:]
                embedding = torch.cat([planes[:, :3], center_points, planes[:, 3:], pre_occ], dim=1)

                outputs['embedding'] = embedding
                outputs['planes_gt'] = planes_gt
                outputs['feat'] = feat
                
        return outputs, loss_dict

    def compute_offset_loss(self, offset, gt_offsets):
        pt_diff = offset - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        offset_norm_loss = torch.sum(pt_dist) / (pt_dist.shape[0] + 1e-6)
        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(offset, p=2, dim=1)
        pt_offsets_ = offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff) / (direction_diff.shape[0] + 1e-6)

        return offset_norm_loss + offset_dir_loss

    def compute_loss(self, occ, class_logits, residuals, distance, off_center, occ_target, label_target,
                     anchors_gt, residual_gt, planes_gt, mean_xyz, r_coords,
                     lw=(1, 1, 1, 1, 1),
                     mask=None, pos_weight=1.0):
        '''
        :return: loss: (Tensor)
        '''
        # compute occupancy loss (bce)
        occ = occ.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            occ = occ[mask]
            occ_target = occ_target[mask]
            label_target = label_target[mask]
            class_logits = class_logits[mask]
            residuals = residuals[mask]
            distance = distance[mask]
            r_coords = r_coords[mask]
            if off_center is not None:
                off_center = off_center[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            if off_center is not None:
                return torch.Tensor([0.0]).cuda()[
                           0] * occ.sum() * off_center.sum() * residuals.sum() * class_logits.sum() * distance.sum()
            else:
                return torch.Tensor([0.0]).cuda()[0] * occ.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # plane loss            
        class_logits = class_logits[occ_target]
        residuals = residuals[occ_target]
        label_target = label_target[occ_target]
        r_coords = r_coords[occ_target]
        distance = distance[occ_target]
        if off_center is not None:
            off_center = off_center[occ_target]

        valid = torch.nonzero(label_target >= 0, as_tuple=False).squeeze(1)
        if len(valid) != 0:
            label_target = label_target[valid]
            class_logits = class_logits[valid]
            residuals = residuals[valid]
            r_coords = r_coords[valid]
            distance = distance[valid]
            if off_center is not None:
                off_center = off_center[valid]

            # extract gt for planes
            bs = len(anchors_gt)
            anchors_target = torch.zeros([label_target.shape[0]], device=label_target.device).long()
            residual_target = torch.zeros([label_target.shape[0], 3], device=label_target.device)
            planes_target = torch.zeros([label_target.shape[0], 4], device=label_target.device)
            for b in range(bs):
                batch_ind = torch.nonzero(r_coords[:, -1] == b, as_tuple=False).squeeze(1)
                anchors_target[batch_ind] = anchors_gt[b][label_target[batch_ind]]
                residual_target[batch_ind] = residual_gt[b][label_target[batch_ind]]
                planes_target[batch_ind] = planes_gt[b][label_target[batch_ind]]


            class_loss = F.cross_entropy(class_logits, anchors_target)
            idx = torch.arange(residuals.shape[0], device=residuals.device).long()
            residuals_roi = residuals[idx, anchors_target]
            residual_loss = F.smooth_l1_loss(residuals_roi * 20, residual_target * 20)

            # ----distance loss-----
            coords = torch.cat([r_coords[:, :3], torch.ones_like(r_coords[:, :1])], dim=1)
            planes_target = planes_target / (- planes_target[:, 3:4])
            distance_gt = - (coords.unsqueeze(1) @ planes_target.unsqueeze(-1)).squeeze() / torch.norm(
               planes_target[:, :3],
               dim=1)

            distance = apply_log_transform(distance.squeeze(1))
            distance_gt = apply_log_transform(distance_gt / 0.12)
            distance_loss = torch.mean(torch.abs(distance - distance_gt))

            if off_center is not None:
                # compute offset loss
                # pt_offsets: (N, 3), float, cuda
                # coords: (N, 3), float32
                mean_xyz_target = torch.zeros([label_target.shape[0], 3], device=label_target.device)
                for b in range(bs):
                    batch_ind = torch.nonzero(r_coords[:, -1] == b, as_tuple=False).squeeze(1)
                    unique_ins = torch.unique(label_target[batch_ind])
                    for ins in unique_ins:
                        batch_ins_ind = torch.nonzero(label_target[batch_ind] == ins, as_tuple=False).squeeze(1)
                        mean_xyz_target[batch_ind[batch_ins_ind]] = r_coords[batch_ind[batch_ins_ind], :3].mean(0)

                # mean_xyz_target = torch.zeros([label_target.shape[0], 3], device=label_target.device)
                # for b in range(bs):
                #     batch_ind = torch.nonzero(r_coords[:, -1] == b, as_tuple=False).squeeze(1)
                #     mean_xyz_target[batch_ind] = mean_xyz[b][label_target[batch_ind]]
                
                r_coords = r_coords[:, :3]

                gt_offsets_center = mean_xyz_target - r_coords  # (N, 3)
                off_loss = self.compute_offset_loss(off_center, gt_offsets_center)

            else:
                off_loss = 0
        else:
            class_loss = residual_loss = off_loss = 0

        # compute final loss
        loss = lw[0] * occ_loss + lw[1] * class_loss + lw[2] * residual_loss + lw[3] * distance_loss + lw[4] * off_loss
        return loss
