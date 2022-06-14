import torch
import torch.nn as nn

from .backbone import MnasMulti
from .planarrecon_network import FragNet
from .gru_fusion import GRUFusion
from .tracking_and_fusion import TrackFuse
from utils import tocuda


class PlanarRecon(nn.Module):
    '''
    PlanarRecon main class.
    '''

    def __init__(self, cfg):
        super(PlanarRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.fragment_net = FragNet(cfg.MODEL)
        self.track_fuse = TrackFuse(cfg)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'plane_anchors':           (List), gt plane_anchors_idx
                                    [(number of planes,)]
            'residual':                (List), gt plane residules
                                    [(number of planes, 3)]
            'mean_xyz':                (List), gt plane center coordinate
                                    [(number of planes, 3)]
            'label_list'               (List), gt plane labels for each level
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results: 
            'label_volume':            (PointTensor), predicted plane labels,
                                    'C: coordinates, F: label'
            'plane_map':               (Tensor), predicted plane parameters,
                                    'max_planes, 4'
        }
                 loss_dict: dict: {
            'multi_level_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.fragment_net(features, inputs, outputs)
        
        if self.cfg.TRACKING:
            outputs, loss_dict = self.track_fuse(inputs, outputs, loss_dict)
        # gather loss.
        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
