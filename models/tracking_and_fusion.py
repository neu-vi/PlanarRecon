import io
import numpy as np
import torch
from torch._C import default_generator
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import PointTensor
from models.modules import MLPGRU
from tools.bin_mean_shift import Bin_Mean_Shift
from loguru import logger
from utils import sparse_to_dense_long
from .matching import *


class TrackFuse(nn.Module):
    '''
    Tracking and Fusion module
    '''
    default_config = {
        'descriptor_dim': 64,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.01,
    }
    def __init__(self, cfg):
        super(TrackFuse, self).__init__()
        MAX_NUM = 3000
        self.cfg = cfg
        self.max_num = 4000
        self.iou_weight = 1.0
        self.scene_name = None
        self.plane_map = torch.zeros(MAX_NUM, 4)
        self.feature_map = torch.zeros(MAX_NUM, 64)
        self.score_map = torch.zeros(MAX_NUM)
        self.target_label = torch.zeros(MAX_NUM)
        self.label_volume = PointTensor(torch.Tensor([]).long(), torch.Tensor([]).view(0, 3).long())

        self.mean_shift = Bin_Mean_Shift(device='cuda')

        self.iou_threshold = 0.01
        self.config = self.default_config
        self.encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
        self.gru = MLPGRU(64, 68)
        self.plane_updation = nn.Linear(64, 1)
    
    def reset(self,):
        self.label_volume = PointTensor(torch.Tensor([]).long(), torch.Tensor([]).view(0, 3).long()).cuda()
        self.plane_map = self.plane_map.zero_().cuda()
        self.feature_map = self.feature_map.zero_().cuda()
        self.score_map = self.score_map.zero_().cuda()
        self.target_label = self.target_label.zero_().cuda().long()

    def fragment_planes(self, relative_origin):
        '''
        '''
        # ------update plane points, planes, features------
        global_coords = self.label_volume.C
        labels = self.label_volume.F
        dim = (torch.Tensor(self.cfg.MODEL.N_VOX).cuda()).int()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        fragment_valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        fragment_label = labels[fragment_valid]
        pre_id = torch.unique(fragment_label)

        if len(pre_id) != 0:
            pre_features = self.feature_map[pre_id]
            pre_planes = self.plane_map[pre_id]
            pre_scores = self.score_map[pre_id]
            pre_labels = self.target_label[pre_id]
        else:
            pre_planes = pre_features = pre_scores = pre_labels = None
        
        return pre_features, pre_planes, pre_scores, pre_labels, pre_id

    def matching(self, plane_clusters, plane_features, plane_occ, plane_labels, iou, angle_simi, dis_simi, relative_origin):
        desc1, kpts1, scores1, pre_labels, pre_id = self.fragment_planes(relative_origin)
        plane_features = self.encoder(plane_features)
        plane_features = torch.nn.functional.normalize(plane_features, p=2, dim=1)

        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0 = plane_features.permute(1, 0).contiguous().unsqueeze(0)
        kpts0 = plane_clusters.unsqueeze(0)
        scores0 = plane_occ.unsqueeze(0)
            
        iou = iou[:, pre_id]
        angle_simi = angle_simi[:, pre_id]
        dis_simi = dis_simi[:, pre_id]

        n = plane_occ.shape[0]
        ids = torch.zeros(n).cuda().long()
        if desc1 is not None:
            desc1 = desc1.permute(1, 0).contiguous().unsqueeze(0)
            kpts1 = kpts1.unsqueeze(0)
            scores1 = scores1.unsqueeze(0)

            # Keypoint MLP encoder.
            desc0 = desc0 + self.kenc(kpts0, scores0)
            desc1 = desc1 + self.kenc(kpts1, scores1)

            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1)

            # Final MLP projection. 
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / self.config['descriptor_dim']**.5
            # Run the optimal transport.
            scores = log_optimal_transport(
                    scores, self.bin_score,
                    iters=self.config['sinkhorn_iterations'])
            if not self.training:
                combine_scores = iou + self.iou_weight * scores[0, :-1, :-1].exp()
                _, indices0 = combine_scores.max(dim=1)
                    # desk reject
                ind = torch.arange(len(indices0)).cuda()
                valid1 = iou[ind, indices0] >= self.iou_threshold
                valid2 = angle_simi[ind, indices0] >= 0.866
                valid3 = scores[0, ind, indices0].exp() >= self.config['match_threshold']
                valid1 = valid1 & valid2 & valid3
                ids[valid1] = pre_id[indices0[valid1]]
            else:

                # Get the matches with score above "match_threshold".
                max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
                indices0, indices1 = max0.indices, max1.indices
                mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
                mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
                zero = scores.new_tensor(0)
                mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
                valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
                valid1 = mutual1 & valid0.gather(1, indices1)
                indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
                indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

                if self.training:
                    assign_matrix_gt = plane_labels.unsqueeze(1) == pre_labels.unsqueeze(0)
                    indices = torch.nonzero(assign_matrix_gt, as_tuple=False)
                    ids[indices[:, 0]] = pre_id[indices[:, 1]]
                else:
                    valid = indices0 > -1
                    ids[valid.squeeze(0)] = pre_id[indices0[valid]]
        else:
            scores = None
        # ----------------generate new id-----------------------------
        new_ids_ind = torch.nonzero(ids == 0).squeeze(1).long()
        if len(new_ids_ind) != 0:
            if self.label_volume.F.shape[0] != 0:
                max_id = self.label_volume.F.max()
            else:
                max_id = 0
            new_first_id = max_id + 1
            ids[new_ids_ind] = torch.arange(len(new_ids_ind)).cuda() + new_first_id
        
        # --------update feature score-------
        self.score_map[ids] = plane_occ
        self.target_label[ids] = plane_labels
        return ids, scores, pre_labels, plane_features

    def fusion(self, current_coords, current_ids, ids, plane_clusters, plane_features, plane_weight, relative_origin):
        pre_features = self.feature_map[ids]
        pre_planes = self.plane_map[ids]
        
        valid = (pre_planes != 0).any(1)
        updated_planes = plane_clusters

        if valid.sum() > 0:
            features = torch.cat([plane_features, plane_clusters], dim=-1)
            plane_features = self.gru(pre_features, features)
            weight = self.plane_updation(plane_features[valid])
            weight = F.sigmoid(weight)
            updated_planes[valid] = (plane_clusters[valid] + weight * pre_planes[valid]) / (1 + weight)
            
        # ------update label------
        global_coords = self.label_volume.C
        labels = self.label_volume.F
        dim = (torch.Tensor(self.cfg.MODEL.N_VOX).cuda()).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        fragment_valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        fragment_label = labels[fragment_valid]
        # sparse to dense
        global_volume = sparse_to_dense_long(global_coords[fragment_valid], fragment_label, dim_list,
                                              0, fragment_label.device)

        current_volume = sparse_to_dense_long(current_coords, current_ids, dim_list,
                                               0, current_ids.device)
        current_volume[global_volume != 0] = global_volume[global_volume != 0]
        updated_coords = torch.nonzero(current_volume, as_tuple=False).squeeze(1)
        updated_ids = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]

        self.label_volume.F = torch.cat(
            [self.label_volume.F[fragment_valid == False], updated_ids])
        updated_coords = updated_coords + relative_origin
        self.label_volume.C = torch.cat([self.label_volume.C[fragment_valid == False], updated_coords])
        
        # -------------
        ids = ids.long()
        self.feature_map[ids] = plane_features
        self.plane_map[ids] = updated_planes

        return updated_planes

    def compute_iou(self, plane_param, plane_points, relative_origin):
        global_coords = self.label_volume.C
        labels = self.label_volume.F
        dim = (torch.Tensor(self.cfg.MODEL.N_VOX).cuda()).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        fragment_valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        fragment_label = labels[fragment_valid]
        # sparse to dense
        global_volume = sparse_to_dense_long(global_coords[fragment_valid], fragment_label, dim_list,
                                              0, fragment_label.device)

        n = len(plane_points)
        m = torch.unique(global_volume).max()
        iou = torch.zeros((n, m + 1)).cuda()
        angle_simi = torch.zeros((n, m + 1)).cuda()
        dis_simi = torch.zeros((n, m + 1)).cuda()
        # m = 0 means id = -1
        for i in range(n):
            points = plane_points[i]
            instances = global_volume[points[:, 0], points[:, 1], points[:, 2]]
            instances_unique = torch.unique(instances)
            for ins in instances_unique:
                if ins != 0:
                    a = torch.nonzero(self.label_volume.F == ins).shape[0]
                    b = points.shape[0]
                    c = torch.nonzero(instances == ins).shape[0]
                    iou[i, ins] = c / min((a + b - c), self.max_num)
                    plane_a = self.plane_map[ins] / torch.norm(self.plane_map[ins][:3])
                    plane_b = plane_param[i] / torch.norm(plane_param[i][:3])
                    angle_simi[i, ins] = torch.dot(plane_a[:3], plane_b[:3]).abs()
                    dis_simi[i, ins] = (plane_a[3] - plane_b[3]).abs()
        return iou, angle_simi, dis_simi

    def forward(self, inputs, outputs, loss_dict):
        # delete computational graph to save memory
        self.feature_map = self.feature_map.detach()
        self.plane_map = self.plane_map.detach()
        self.score_map = self.score_map.detach()

        batch_size = len(inputs['fragment'])
        total_loss = 0
        outputs['label_volume'] = []
        outputs['plane_map'] = []
        for i in range(batch_size):
            scene = inputs['scene'][i]
            scene = scene.replace('/', '-')
            global_origin = inputs['vol_origin'][i]  # origin of global volume
            origin = inputs['vol_origin_partial'][i]  # origin of part volume
            if self.scene_name is None or scene != self.scene_name:
                if not self.training:
                    outputs['label_volume'] = deepcopy(self.label_volume)
                    outputs['plane_map'] = deepcopy(self.plane_map)
                self.reset()
                self.scene_name = scene

            if 'embedding' not in outputs.keys():
                continue
            batch_ind = torch.nonzero(outputs['coords_'][-1][:, 0] == i, as_tuple=False).squeeze(1)
            mask0 = outputs['embedding'][batch_ind, :3].abs() < 2
            mask0 = mask0.all(-1)
            batch_ind = batch_ind[mask0]
            embedding, distance, occ = outputs['embedding'][batch_ind, :6], outputs['embedding'][batch_ind, 6:7], outputs['embedding'][batch_ind, 7:]
            feat = outputs['feat'][batch_ind]
            if self.training:
                plane_gt = outputs['planes_gt'][i]
                labels = outputs['label_target'][2][batch_ind]


            # --------clustering------------
            scale = embedding.max(dim=0)[0] - embedding.min(dim=0)[0]
            embedding = embedding / scale
            segmentation, plane_clusters, invalid, _ = self.mean_shift(occ, embedding)

            if segmentation is not None:
                # pixel-level -> instance-level
                segmentation = segmentation.argmax(-1)
                segmentation[invalid] = -1
                plane_clusters = (plane_clusters * scale)[:, :4]
                # plane_clusters[:, 3] = 1

                plane_occ = []
                plane_weight = []
                plane_features = []
                plane_labels = []
                plane_param = []
                plane_points = []
                coords = outputs['coords_'][-1][batch_ind, 1:]
                # relative origin in global volume
                relative_origin = (origin - global_origin) / self.cfg.MODEL.VOXEL_SIZE
                relative_origin = relative_origin.cuda().long()

                count = 0
                # To avoid repeating, only pick the first one
                # Parse plane instance
                for i in range(plane_clusters.shape[0]):
                    if self.training:
                        # Generate matching ground truth
                        plane_label = labels[segmentation == i]
                        plane_label = plane_label[plane_label != -1]
                        if plane_label.shape[0] != 0:
                            bincount = torch.bincount(plane_label)
                            label_ins = bincount.argmax()
                            ratio = bincount[label_ins].float() / plane_label.shape[0]
                            if ratio > 0.5 and label_ins not in plane_labels:
                                plane_labels.append(label_ins)
                                plane_points.append(coords[segmentation == i])
                                plane_occ.append(occ[segmentation == i].mean())
                                plane_weight.append(occ[segmentation == i].sum())
                                plane_clusters[i, 3] = (distance[segmentation == i].mean() * occ[segmentation == i]).sum(0) / (occ[segmentation == i].sum() + 1e-4)
                                plane_param.append(plane_clusters[i])

                                # -----3D pooling-----
                                plane_features.append((feat[segmentation == i] * occ[segmentation == i]).sum(0) / (occ[segmentation == i].sum() + 1e-4))
                                    
                                segmentation[segmentation == i] = count
                                count += 1
                            else:
                                segmentation[segmentation == i] = -1           
                        else:
                            segmentation[segmentation == i] = -1

                    else:
                        plane_labels.append(occ.sum().long() * 0)
                        plane_points.append(coords[segmentation == i])
                        plane_occ.append(occ[segmentation == i].mean())
                        plane_weight.append(occ[segmentation == i].sum())
                        plane_clusters[i, 3] = (distance[segmentation == i].mean() * occ[segmentation == i]).sum(0) / (occ[segmentation == i].sum() + 1e-4)
                        plane_param.append(plane_clusters[i])
                        # -----3D average pooling-----
                        plane_features.append((feat[segmentation == i] * occ[segmentation == i]).sum(0) / (occ[segmentation == i].sum() + 1e-4))

                # avoid batch size = 1; TODO
                desc1, _, _, _, _ = self.fragment_planes(relative_origin)
                if desc1 is None:
                    pre_true = True
                else:
                    pre_true = desc1.shape[0] != 1
                # ----------------------
                
                if (self.training and len(plane_param) > 1 and pre_true) or (not self.training and len(plane_param) != 0):
                    plane_param = torch.stack(plane_param)
                    plane_features = torch.stack(plane_features)
                    plane_occ = torch.stack(plane_occ)
                    plane_weight = torch.stack(plane_weight)
                    # gt
                    plane_labels = torch.stack(plane_labels)
                    # -------matching-------
                    iou, angle_simi, dis_simi = self.compute_iou(plane_param, plane_points, relative_origin)
                    ids, scores, pre_labels, plane_features= self.matching(plane_param, plane_features, plane_occ, plane_labels, iou, angle_simi, dis_simi, relative_origin)
                    current_ids = ids[segmentation[segmentation != -1]]
                    coords = coords[segmentation != -1]
                    updated_planes = self.fusion(coords, current_ids, ids, plane_param, plane_features, plane_weight, relative_origin)

                    if self.training:
                        plane_gt = plane_gt[plane_labels]
                        loss = self.compute_loss(updated_planes, plane_gt, scores, pre_labels, plane_labels)
                    else:
                        loss = torch.Tensor([0])[0]
                    total_loss = total_loss + loss

        if total_loss != 0:
            if (not torch.isnan(total_loss)):
                loss_dict.update({f'match_fuse_loss': total_loss / batch_size})
            elif scores is not None:
                loss_dict.update({f'match_fuse_loss': 0 * updated_planes.sum() * scores.sum()})
            else:
                loss_dict.update({f'match_fuse_loss': 0 * updated_planes.sum()})
        return outputs, loss_dict

    def compute_loss(self, updated_planes, plane_gt, assign_matrix, pre_labels, plane_labels):
        alpha = .25
        gamma = 2

        if assign_matrix is not None:
            assign_matrix = assign_matrix[0, :-1, :-1].exp()
            # generate assign matrix gt
            assign_matrix_gt = plane_labels.unsqueeze(1) == pre_labels.unsqueeze(0)
            pos_mask, neg_mask = assign_matrix_gt == 1, assign_matrix_gt == 0

            # corner case: empty mask, assign a wrong gt
            if not pos_mask.any():
                pos_mask[0, 0] = True
            assign_matrix = torch.clamp(assign_matrix, 1e-6, 1-1e-6)

            match_loss = - torch.log(assign_matrix[pos_mask]).mean()
        else:
            match_loss = 0
        
        # plane residual loss -- l1
        plane_loss = torch.mean(torch.abs(updated_planes - plane_gt)) * 2
        return match_loss + plane_loss