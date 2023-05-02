'''
Generate instance groundtruth .txt files (for evaluation)
Code is partial developed based on scannet export_train_mesh_for_evaluation.py in their github repo

original author: Fengting Yang
Modified by Yiming Xie for PlanarRecon project
'''

import torch
import numpy as np 
from tsdf import TSDF
from util import planarize, get_planeIns_RANSAC_mesh, get_unique_colors, break_faces, extract_plane_surface
import trimesh
import os
import os.path as osp
import ray
import argparse
from tqdm import tqdm
from eval_time import EvalTime
from chamferdist import ChamferDistance


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(2021)
np.random.seed(2021)
ET = EvalTime(disable=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate meshes from GT planes')
    parser.add_argument("--model", metavar="DIR",
                        # help="path to dataset with GT planes", default='/home/xie.yim/repo/NeuralRecon/results/scene_demo_checkpoints_fusion_eval_47')
                        help="path to dataset with GT planes", default='/home/xie.yim/repo/NeuralRecon/results/scene_scannet_checkpoints_fusion_eval_47')

    parser.add_argument('--save_path', metavar="DIR", default='/home/xie.yim/repo/seq_ransac_refac/RANSAC_output_double')

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=2, help='#processes launched to process scenes.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()

args = parse_args()


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def seq_ransac_plane(args, scenes):
    for scene in tqdm(scenes):
        # if not os.path.exists(os.path.join(args.model, '{}_trim.ply'.format(scene))):
            # continue
        print(scene)
        # PATH
        tsdf_file = os.path.join(args.model, '{}.npz'.format(scene))
        mesh_file = os.path.join(args.model, '{}.ply'.format(scene))
        save_pth = args.save_path
        if not osp.isdir(osp.abspath(save_pth)):
            os.makedirs(save_pth)

        scene = osp.basename(tsdf_file)[:-4]

        # hyper-param
        ransac_iter = 1000
        angle_thres=30 # degree
        dist_thres = 0.25 # pnt to plane dist thres, meter
        init_verts_thres= 200 # min verts number for a proposal
        connect_kernel = 5 # maxpooling kernel sz for connection check, pick from 3, 5, 7, etc
        connect_verts_thres = 4 # min_verts number for a final plane instance, once fail, the RANSAC will stop

        tsdf = TSDF.load(tsdf_file)
        tsdf_origin, voxel_sz = tsdf.origin, tsdf.voxel_size

        # mesh_tsdf = tsdf.get_mesh()
        # mesh_tsdf.export('tst.ply')

        pred = trimesh.load(mesh_file)

        ET('generate_begin')
        verts, faces, norms = pred.vertices, pred.faces, pred.vertex_normals
        verts_torch = torch.from_numpy(verts.view(np.ndarray)).float().T.to(device) # 3*n
        norm_torch =  torch.from_numpy(norms.view(np.ndarray)).float().T.to(device)

        plane_ins, plane_params = get_planeIns_RANSAC_mesh(verts_torch, norm_torch, tsdf.tsdf_vol, voxel_sz, tsdf_origin, connect_kernel,
                                                        angle_thres=angle_thres, dist_thres=dist_thres,
                                                        init_verts_thres=init_verts_thres, connect_verts_thres = connect_verts_thres,
                                                        n_iter = ransac_iter, device=device)

        n_ins = plane_ins.max().cpu().int().item() + 1
        planar_verts = planarize(verts_torch, plane_ins, plane_params, n_ins)

        colors = get_unique_colors(ret_n = n_ins)
        plane_ins_np = plane_ins.int().cpu().numpy()
        plane_colors = colors[plane_ins_np]
        
        # -- this is for visualization: make sure the plane color for the same instance from all methods is the same, e.g. ground plane is always green for all methods
        consistent = False
        if consistent is True:
            gt_points = list(
                            np.load(os.path.join('/work/vig/Datasets/PlanarRecon/planes_tsdf_9/', scene, 'annotation', 'plane_points.npy'), allow_pickle=True))
            colorMap_vis_gt = list(
                            np.load(os.path.join('/work/vig/Datasets/PlanarRecon/planes_tsdf_9/', scene, 'annotation', 'colorMap_vis.npy'), allow_pickle=True))
            ind_list = []
            unique = np.unique(plane_ins_np)
            for i in unique:
                points_ins = verts[plane_ins_np == i]
                dist_list = []
                points_ins_cuda = torch.from_numpy(points_ins).cuda().unsqueeze(0).float()
                for gt_points_ins in gt_points:
                    chamferDist = ChamferDistance()
                    gt_points_ins_cuda = torch.from_numpy(gt_points_ins).cuda().unsqueeze(0).float()
                    dist1 = chamferDist(points_ins_cuda, gt_points_ins_cuda, reduction='None')
                    dist_list.append(dist1.mean().data.cpu().numpy())
                ind = np.array(dist_list).argmin()
                if ind not in ind_list:
                    color = colorMap_vis_gt[ind]
                    ind_list.append(ind)
                plane_colors[plane_ins_np == i] = np.ones_like(points_ins) * color
        # -- this is for visualization: make sure the plane color for the same instance from all methods is the same, e.g. ground plane is always green for all methods
        
        ET('generate_over')
        # mesh where color indicates instances
        res = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=plane_colors, process=False)
        res.export('{}/{}_ransac_ori_mesh.ply'.format(save_pth, scene))

        # plane project results -- this one will used for eval
        plane_verts_np = planar_verts.cpu().numpy().T
        res = trimesh.Trimesh(vertices = plane_verts_np, faces=faces,
                            vertex_colors=plane_colors, process=False)
        res.export('{}/{}_ransac_proj_mesh.ply'.format(save_pth,scene))
        np.save('{}/{}_planeIns'.format(save_pth, scene), {'plane_ins':plane_ins_np, 'conf':None})

        # plane_proj results with face break ---- just for viz
        plane_face = break_faces(faces, plane_ins_np)
        piecewise_mesh = trimesh.Trimesh(vertices=plane_verts_np, faces=plane_face,
                            vertex_colors=plane_colors, process=False)
        piecewise_mesh.export('{}/{}_ransac_piecewise.ply'.format(save_pth, scene))


        # plane surface only
        plane_only_mesh = extract_plane_surface(plane_ins_np, piecewise_mesh)
        plane_only_mesh.export('{}/{}_plane_only.ply'.format(save_pth, scene))

# res.show()
def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


if __name__ == "__main__":
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    val_file = open('./scannetv2_val.txt', 'r')
    # val_file = open('/home/xie.yim/repo/PlanarReconstruction/test_scenes.txt', 'r')

    files = sorted(val_file.read().splitlines())
    files = split_list(files, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(seq_ransac_plane.remote(args, files[w_idx]))

    results = ray.get(ray_worker_ids)
    # seq_ransac_plane(args, files)