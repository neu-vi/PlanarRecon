import numpy as np
import cv2
from scipy.io import loadmat
from glob import glob
import os.path as osp

# code developed based on
# https://github.com/fuy34/planerecover/blob/master/eval/compute_sc.m
# https://github.com/fuy34/planerecover/blob/master/eval/compare_segmentations.m

def compact_segm(seg_in):
    seg = seg_in.copy()
    uniq_id = np.unique(seg)
    cnt = 1
    for id in sorted(uniq_id):
        if id == 0:
            continue
        seg[seg==id] = cnt
        cnt += 1

    # every id (include non-plane should not be 0 for the later process in match_seg
    seg = seg + 1
    return seg

def match_seg(pred_in, gt_in):
     assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

     pred, gt = compact_segm(pred_in), compact_segm(gt_in)
     n_gt = gt.max() + 1
     n_pred = pred.max() + 1


     # this will offer the overlap between gt and pred
     # if gt == 1, we will later have conf[1, j] = gt(1) + pred(j) * n_gt
     # essential, we encode conf_mat[i, j] to overlap, and when we decode it we let row as gt, and col for pred
     # then assume we have 13 gt label, 6 pred label --> gt 1 will correspond to 14, 1+2*13 ... 1 + 6*13
     overlap =  gt + n_gt * pred
     freq, bin_val = np.histogram(overlap, np.arange(0, n_gt * n_pred+1)) # hist given bins [1, 2, 3] --> return [1, 2), [2, 3)
     conf_mat = freq.reshape([ n_gt, n_pred], order='F') # column first reshape, like matlab

     acc = np.zeros([n_gt, n_pred])
     for i in range(n_gt):
         for j in range(n_pred):
             gt_i = conf_mat[i].sum()
             pred_j = conf_mat[:, j].sum()
             gt_pred = conf_mat[i, j]
             acc[i,j] = gt_pred / (gt_i + pred_j - gt_pred) if (gt_i + pred_j - gt_pred) != 0 else 0
     return acc[1:, 1:], pred, gt

def compute_sc(gt_in, pred_in):
    # to be consistent with skimage sklearn input arrangment

    assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

    acc, pred, gt  = match_seg(pred_in, gt_in) # n_gt * n_pred

    bestmatch_gt2pred = acc.max(axis=1)
    bestmatch_pred2gt = acc.max(axis=0)

    pred_id, pred_cnt = np.unique(pred, return_counts=True)
    gt_id, gt_cnt = np.unique(gt, return_counts=True)

    cnt_pred, sum_pred = 0, 0
    for i, p_id in enumerate(pred_id):
        cnt_pred += bestmatch_pred2gt[i] * pred_cnt[i]
        sum_pred += pred_cnt[i]

    cnt_gt, sum_gt = 0, 0
    for i, g_id in enumerate(gt_id):
        cnt_gt += bestmatch_gt2pred[i] * gt_cnt[i]
        sum_gt += gt_cnt[i]

    sc = (cnt_pred / sum_pred + cnt_gt / sum_gt) / 2

    return sc

# code for test
# if __name__ == '__main__':
#     pred_pth = '/data/PlaneRecon_CVPR_2022/eval/labels/eval_masks/'
#     gt_pth = '/data/PlaneRecon_CVPR_2022/eval/labels/SYN_GT_sgmt/'
#
#     pred_lst = sorted(glob(pred_pth + '*.png'))
#     for p_pth in pred_lst:
#         name = osp.basename(p_pth)[:-4]
#         pred = cv2.imread(p_pth, -1)
#         pred = cv2.resize(pred, dsize =(640, 380), interpolation=cv2.INTER_NEAREST)
#
#         _gt = loadmat(gt_pth + '/{}-planes.mat'.format(name))
#         gt = _gt['assignment']
#         # match_seg(pred.reshape(-1), gt.reshape(-1))
#         sc = compute_sc(pred.reshape(-1), gt.reshape(-1))
#         print(name, sc)
