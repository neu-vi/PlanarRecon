# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [PlanarRecon](https://github.com/neu-vi/PlanarRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

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
import open3d as o3d

import os, sys
import json
import scipy.stats as stats


def project_to_mesh(from_mesh, to_mesh, attribute, attr_name, color_mesh=None, dist_thresh=None):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attr_name] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = attribute.copy()
    pred_colors = from_mesh.visual.vertex_colors  if color_mesh is None else color_mesh.visual.vertex_colors

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    mesh.vertex_attributes[attr_name] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh


def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02, error_map=True):
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    if error_map:
        # repeat but without downsampling
        mesh_pred = o3d.io.read_triangle_mesh(file_pred)
        mesh_trgt = o3d.io.read_triangle_mesh(file_trgt)
        verts_pred = np.asarray(mesh_pred.vertices)
        verts_trgt = np.asarray(mesh_trgt.vertices)
        _, dist1 = nn_correspondance(verts_pred, verts_trgt)
        _, dist2 = nn_correspondance(verts_trgt, verts_pred)
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        # recall_err_viz
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        # cmap = cm.get_cmap('brg')
        dist1_n = dist1 / 0.3
        color = cmap(dist1_n)
        # recal_mask = (dist1 < threshold)        
        # color = np.array([[1., 0., 0]]).repeat(verts_trgt.shape[0], axis=0)
        # color[recal_mask] = np.array([[0, 1., 0]]).repeat((recal_mask.sum()).astype(np.int), axis=0)
        mesh_trgt.vertex_colors = o3d.utility.Vector3dVector(color[:, :3])

        # precision_err_viz
        dist2_n = dist2 / 0.4
        color = cmap(dist2_n)
        # prec_mask = dist2 < threshold
        # color = np.array([[1., 0., 0]]).repeat(verts_pred.shape[0], axis=0)
        # color[prec_mask] = np.array([[0, 1., 0]]).repeat((prec_mask.sum()).astype(np.int), axis=0)
        mesh_pred.vertex_colors = o3d.utility.Vector3dVector(color[:, :3])
    else:
        mesh_pred = mesh_trgt = None
    return metrics, mesh_pred, mesh_trgt


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

