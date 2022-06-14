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

import sys

import numpy as np

sys.path.append('.')

import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from tools.simple_loader import *
from tools.generate_planes import generate_planes
from tools.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils import coordinates

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ground truth plane')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to dataset", default='./scannet')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='planes_9/')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--voxel_size', default=0.04, type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=8, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose, max_depth=3.0):
    """Get corners of 3D camera view frustum of depth image
    """
    if depth_im is not None:
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]
        max_depth = np.max(depth_im)
    else:
        im_h = 480
        im_w = 640
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def compute_global_volume(args, cam_intr, cam_pose_list):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(cam_pose_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(cam_pose_list.keys()))[ind]
    else:
        image_id = cam_pose_list.keys()
    for id in image_id:
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(None, cam_intr, cam_pose, max_depth=args.max_depth)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # Adjust volume bounds and ensure C-order contiguous
    vol_dim = np.round((vol_bnds[:, 1] - vol_bnds[:, 0]) / args.voxel_size).copy(
            order='C').astype(int)
    vol_bnds[:, 1] = vol_bnds[:, 0] + vol_dim * args.voxel_size
    vol_origin = vol_bnds[:, 0].copy(order='C').astype(np.float32)

    return vol_dim, vol_origin


def save_label_full(args, scene, vol_dim, vol_origin, planes, plane_points):
    planes = np.concatenate([planes, - np.ones_like(planes[:, :1])], axis=-1)
    
    # ========================generate indicator gt========================
    planes = torch.from_numpy(planes).cuda()
    coords = coordinates(vol_dim, device=planes.device)
    coords = coords.type(torch.float) * args.voxel_size + torch.from_numpy(vol_origin).view(3, 1).cuda()
    coords = coords.permute(1, 0).contiguous()

    min_dist = None
    indices = None
    for i, points in enumerate(plane_points[:]):
        points = torch.from_numpy(points).cuda()
        chamLoss = dist_chamfer_3D.chamfer_3DDist()
        dist1, _, _, _ = chamLoss(coords.unsqueeze(0), points.unsqueeze(0))
        if min_dist is None:
            min_dist = dist1
            indices = torch.zeros_like(dist1)
        else:
            current_id = torch.ones_like(indices) * i
            indices = torch.where(dist1 < min_dist, current_id, indices)
            min_dist = torch.where(dist1 < min_dist, dist1, min_dist)
    # remove too far points which may not have a plane
    current_id = torch.ones_like(indices) * -1
    indices = torch.where(0.36 ** 2 < min_dist, current_id, indices)
    indices = indices.view(vol_dim.tolist()).data.cpu().numpy()

    np.savez_compressed(os.path.join(args.save_path, scene, 'indices'), indices)
    # ==============================================================================================


def save_fragment_pkl(args, scene, cam_pose_list, vol_dim, vol_origin):
    # view selection
    fragments = []
    print('segment: process scene {}'.format(scene))

    all_ids = []
    ids = []
    count = 0
    last_pose = None
    for id in cam_pose_list.keys():
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    ids = []
                    count = 0

    # save fragments
    for i, ids in enumerate(all_ids):
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': ids,
            'vol_origin': vol_origin,
            'vol_dim': vol_dim,
            'voxel_size': args.voxel_size,
        })

    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

    return


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(args, scannet_files):
    planes_all = []
    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        cam_pose_all = {}

        n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
        intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
        cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
        dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)

        planes, plane_points = generate_planes(args, scene, save_mesh=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.loader_num_workers)

        for id, (cam_pose, _, _) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))

            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            cam_pose_all.update({id: cam_pose})

        vol_dim, vol_origin = compute_global_volume(args, cam_intr, cam_pose_all)
        save_label_full(args, scene, vol_dim, vol_origin, planes, plane_points)
        save_fragment_pkl(args, scene, cam_pose_all, vol_dim, vol_origin)

    planes_center = np.array([
        [0, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [-1, 0, -1],
        [0, 0, -1],
        [1, 0, -1],
        [1, 0, 1],
    ])
    planes_center = planes_center / np.linalg.norm(planes_center, axis=1)[..., np.newaxis]

    np.save(os.path.join(args.save_path, 'normal_anchors.npy'), planes_center)


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.save_path))
    # todo: fix for both train/val
    splits = ['train', 'val']
    for split in splits:
        fragments = []
        with open(os.path.join(args.data_path[:-6],'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


if __name__ == "__main__":
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    if args.dataset == 'scannet':
        args.data_raw_path = os.path.join(args.data_path, 'scans_raw/')
        args.data_path = os.path.join(args.data_path, 'scans')
        files = sorted(os.listdir(args.data_path))
    else:
        raise NameError('error!')

    files = split_list(files, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))

    results = ray.get(ray_worker_ids)

    # process_with_single_worker(args, files)

    if args.dataset == 'scannet':
        generate_pkl(args)
