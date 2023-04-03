'''
Generate instance groundtruth .txt files (for evaluation)
Code is partial developed based on scannet export_train_mesh_for_evaluation.py in their github repo

last modified: Fengting Yang
2021-10-31
'''

import numpy as np
import glob
import torch
import os

import trimesh
import sys
import argparse


# val_list = '/work/vig/Datasets/ScanNet/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
# plane_mesh_path = os.path.abspath('/work/vig/Datasets/PlanarRecon/planes_tsdf_9')
# dump_path = os.path.abspath('/work/vig/Datasets/PlanarRecon/planes_tsdf_9/instance')
# plane_mesh_path = os.path.abspath('/home/xie.yim/repo/PlanarRecon/results/scene_scannet_bottom_up_fixd_fusion_tsdf_scratch_41_tsdf_scratch')
# dump_path = os.path.abspath('/home/xie.yim/repo/PlanarRecon/results/scene_scannet_bottom_up_fixd_fusion_tsdf_scratch_41_tsdf_scratch/instance')

invalid_pln_id = 16777216 // 100 - 1 # invalid plane id #(255,255,255)


# todo: modify the map_planes in data_prep_util to be re-usable here
def get_planeIns(mesh_file):
    mesh_plane = trimesh.load(mesh_file, process=False)
    # plane_verts = mesh_plane.vertices.view(np.ndarray)
    colors = mesh_plane.visual.vertex_colors.view(np.ndarray)

    chan_0 = colors[:, 2]
    chan_1 = colors[:, 1]
    chan_2 = colors[:, 0]
    plane_id = (chan_2 * 256 ** 2 + chan_1 * 256 + chan_0) // 100 - 1  # there is no (0,0,0) color in fitting mesh
    unique_id = np.unique(plane_id) #ascending order

    plane_ins = np.zeros_like(plane_id, dtype=np.uint32)
    for k, id in enumerate(unique_id):
        if id == invalid_pln_id: continue
        plane_ins[plane_id == id] = (k + 1)

    return plane_ins

def export(mesh_file, output_file):

    plane_ins = get_planeIns(mesh_file)
    assert plane_ins.max() > 1, "No plane in the mesh!"

    # we do not consider semantic meaning of plane, so put all sem_id == 1
    plane_ins_label = np.zeros(plane_ins.shape, dtype=np.int32)
    sem_id = 1

    for ins_id in np.unique(plane_ins):
        if ins_id == 0: continue # ignore none-plane
        instance_mask = plane_ins == ins_id
        plane_ins_label[instance_mask] = sem_id * 1000 + ins_id

    np.savetxt(output_file, plane_ins_label, fmt='%d')


if __name__ == '__main__':
    # split = 'val'
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', default='/work/vig/Datasets/ScanNet/ScanNet/Tasks/Benchmark/scannetv2_val.txt',
                        help='scannet validation set txt')
    parser.add_argument('--plane_mesh_path', default='/work/vig/Datasets/PlanarRecon/planes_tsdf_9',
                        help='path to directory of gt plane mesh files')

    args = parser.parse_args()
    
    val_list = args.val_list
    plane_mesh_path = args.plane_mesh_path
    # path to directory of gt .txt files
    dump_path = os.path.join(plane_mesh_path, "instance")

    if not os.path.isdir(dump_path):
        os.makedirs(dump_path)

    with open(val_list) as f:
        scenes = [line.strip() for line in f]

    n_scenes = len(scenes)
    for i, scan_name in enumerate(sorted(scenes)):
        sys.stdout.write("\rprocessing {}/{}, {}".format(i, n_scenes, scan_name))
        sys.stdout.flush()

        mesh_file = os.path.join(plane_mesh_path, scan_name, 'annotation', 'planes_mesh.ply')
        # mesh_file = os.path.join(plane_mesh_path, scan_name, 'planes_mesh.ply')
        if os.path.isfile(mesh_file):
            if not os.path.isdir(dump_path):
                os.makedirs(dump_path)
            output_file = os.path.join(dump_path, scan_name + '.txt')
            export(mesh_file,  output_file)
