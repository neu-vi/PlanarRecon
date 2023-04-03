import numpy as np
import sys
from numpy.lib.arraysetops import unique

sys.path.append('.')
from sklearn.metrics import rand_score, mutual_info_score
from skimage.metrics import variation_of_information
from tools.segm_cover_utils import compute_sc
import argparse
import ray
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='/home/xie.yim/repo/PlanarRecon/results/scene_scannet_bottom_up_fixd_fusion_tsdf_scratch_41_scratch/plane_ins',
                        help='path to directory of predicted .txt files')
    parser.add_argument('--gt_path', default='/work/vig/Datasets/PlanarRecon/planes_tsdf_9/instance',
                        help='path to directory of gt .txt files')
    parser.add_argument('--output_file', default='',
                        help='output file [default: pred_path/semantic_instance_evaluation.txt]')
    parser.add_argument("--scan_list", default='/work/vig/Datasets/ScanNet/ScanNet/Tasks/Benchmark/scannetv2_val.txt', help="which scene(s) to run on")

    # ray config
    parser.add_argument('--n_proc', type=int, default=8, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=1)
    return parser.parse_args()

args = parse_args()

def eval(opt, scene):

    pred_pth = os.path.join(opt.pred_path, '{}.txt'.format(scene))
    gt_pth = os.path.join(opt.gt_path, '{}.txt'.format(scene))

    pred_ins = np.loadtxt(pred_pth).astype(np.int)
    gt_ins = np.loadtxt(gt_pth).astype(np.int)

    # mask_value_less than 
    # unique_value = np.unique(gt_ins)
    # mask = []
    # for v in unique_value:
    #     ind = np.where(gt_ins == v)[0]
    #     if len(ind) > 1000:
    #         mask.append(ind)
    # mask = np.concatenate(mask)
    # gt_ins = gt_ins[mask]
    # pred_ins = pred_ins[mask]

    ri =  rand_score(gt_ins, pred_ins)
    h1, h2 = variation_of_information(gt_ins, pred_ins) # this metric does not minuse the mutual info
    # mi = mutual_info_score(gt_ins, pred_ins)
    voi = h1 + h2 # - 2*mi
    
    sc = compute_sc(gt_ins, pred_ins)

    print('RI',ri) # ours 0.83, ransac 0.88 --> higher better
    print('VOI', voi) # ours 6.40, ransac 4.61 --> lower better
    print('SC', sc)

    return ri, voi, sc

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(info_files):
    metrics = {'ri':[],
    'voi':[],
    'sc':[]}
    for i, info_file in enumerate(info_files):
        # if info_file == 'scene0701_02':
        #     continue
        ri, voi, sc = eval(args, info_file)
        metrics['ri'].append(ri)
        metrics['voi'].append(voi)
        metrics['sc'].append(sc)
    return metrics

def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


if __name__ == '__main__':
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    # convert to abs path for convenience
    args.pred_path = os.path.abspath(args.pred_path)
    args.gt_path = os.path.abspath(args.gt_path)

    if args.output_file == '':
        args.output_file = os.path.join(args.pred_path, 'semantic_instance_evaluation.txt')

    with open(args.scan_list) as f:
        info_files = [line.strip() for line in f]

    info_files = split_list(info_files, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(info_files[w_idx]))

    results = ray.get(ray_worker_ids)

    # process_with_single_worker(info_files)

    met = {}
    for value in results:
        if len(met) == 0:
            for key2, value2 in value.items():
                met[key2] = value2
        else:
            for key2, value2 in value.items():
                met[key2] += value2
    ri = np.array(met['ri']).mean()
    voi = np.array(met['voi']).mean()
    sc = np.array(met['sc']).mean()
    print('RI_mean', ri)
    print('VOI_mean', voi)
    print('SC mean', sc)
