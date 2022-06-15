import os
import torch
from torch.serialization import save
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
import cv2
from tools.bin_mean_shift import Bin_Mean_Shift
from tools.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from scipy.spatial import Delaunay
from tools.generate_planes import furthest_point_sampling, project2plane, writePointCloudFace
from tools.random_color import random_color
import trimesh


# print arguments
def print_args(args):
    logger.info("################################  args  ################################")
    for k, v in args.__dict__.items():
        logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    logger.info("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], float(default_val), device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_long(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], default_val, dtype=torch.long, device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense



class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        log_dir = cfg.LOGDIR.split('/')[-1]
        self.log_dir = os.path.join('results', 'scene_' + cfg.DATASET + '_' + log_dir)
        self.scene_name = None
        self.global_origin = None
        self.color_vis = random_color()

        self.keyframe_id = None
        
        # intersection parameters
        self.distance = 0.05
        self.min_points = 1
        self.min_angle = 0.5
        self.sample_points = 1000
        self.filter_max = 500
        
    def reset(self):
        self.keyframe_id = 0

    @staticmethod
    def generate_mesh(planes, points, faces):
        points_plane = []
        points_plane_idx = []
        faces_plane = []
        total_points = 0
        color_vis = random_color()
        for i in range(len(planes)):
            plane, plane_points, face = planes[i], points[i], faces[i]
            if (plane != 0).any() and not np.isnan(plane).any():
                if plane.shape[0] == 4:
                    plane /= -plane[3]
                    plane = plane[:3]
                t = (np.matmul(plane_points, plane) - 1) / (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
                plane_points = plane_points - plane[np.newaxis, :3] * t[:, np.newaxis]

                face = face + total_points
                points_idx = np.ones(plane_points.shape[0]).astype(np.int) * i

                points_plane.append(plane_points)
                points_plane_idx.append(points_idx)

                faces_plane.append(face)
                total_points += plane_points.shape[0]
        points_plane_idx = np.concatenate(points_plane_idx, axis=0)
        faces_plane = np.concatenate(faces_plane)

        n_ins = points_plane_idx.max() + 1
        segmentationColor = (np.arange(n_ins + 1) + 1) * 100
        colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256],
                            axis=1)
        colorMap[-1] = 0
        plane_colors = colorMap[points_plane_idx]
     
        colorMap_vis = color_vis(n_ins)
        plane_colors_vis = colorMap_vis[points_plane_idx]

        points_plane = np.concatenate(points_plane, axis=0)
        
        mesh = trimesh.Trimesh(vertices=points_plane, vertex_colors=plane_colors.astype(np.int32), faces=faces_plane, process=False)
        mesh_vis = trimesh.Trimesh(vertices=points_plane, vertex_colors=plane_colors_vis.astype(np.int32), faces=faces_plane, process=False)

        sample_points, _ = trimesh.sample.sample_surface_even(mesh, mesh.vertices.shape[0] * 2)
        vertices_eval = trimesh.Trimesh(vertices=sample_points, process=False)
        
        return mesh, mesh_vis, vertices_eval, points_plane_idx

    def filter(self, plane, coords, p1, p2, threshold=1000):
        normals_ins = - plane[:3] / plane[3:4]
        M = project2plane(normals_ins, coords)
        plane_points_3d = np.concatenate([coords, np.ones_like(coords[:, :1])], axis=-1)
        plane_points_2d = np.matmul(M, plane_points_3d.transpose()).transpose()[:, :2]
        
        p12 = np.concatenate([p1, p2])
        p12 = np.concatenate([p12, np.ones_like(p12[:, :1])], axis=-1)
        p12_2d = np.matmul(M, p12.transpose()).transpose()[:, :2]
        
        v1 = p12_2d[1:] - p12_2d[:1]
        v2 = p12_2d[1:] - plane_points_2d
        xp = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # Cross product
        mask1 = xp > 0
        mask2 = xp < 0
        if (mask1.sum() < mask2.sum()) and mask1.sum() < self.filter_max:
            mask = mask1
        elif (mask1.sum() > mask2.sum()) and mask2.sum() < self.filter_max:
            mask = mask2
        else:
            mask = None
        return mask
    
    def detect_intersection_line(self, coords_list, project_list, planes_list, threshold=0.08):
        num_planes = len(coords_list)
        set_list = []
        for i in range(num_planes):
            for j in range(num_planes):
                if i == j:
                    continue
                if set([i, j]) in set_list:
                    continue
                else:
                    set_list.append(set([i, j]))
                coords_a = coords_list[i]
                coords_b = coords_list[j]
                coords_p_a = project_list[i]
                coords_p_b = project_list[j]
                plane_a = planes_list[i]
                plane_b = planes_list[j]
                plane_a = plane_a / np.linalg.norm(plane_a[:3])
                plane_b = plane_b / np.linalg.norm(plane_b[:3])
                angle_simi = np.absolute(np.dot(plane_a[:3], plane_b[:3]))
                # chamferDist = ChamferDistance()
                coords_p_a_cuda = torch.from_numpy(coords_p_a).cuda().unsqueeze(0).float()
                coords_p_b_cuda = torch.from_numpy(coords_p_b).cuda().unsqueeze(0).float()
                chamLoss = dist_chamfer_3D.chamfer_3DDist()
                dist1, _, _, _ = chamLoss(coords_p_b_cuda, coords_p_a_cuda)
                dist2, _, _, _ = chamLoss(coords_p_a_cuda, coords_p_b_cuda)
                # dist1_ = chamferDist(coords_p_b_cuda, coords_p_a_cuda, reduction='None')
                # dist2_ = chamferDist(coords_p_a_cuda, coords_p_b_cuda, reduction='None')
                dist1 = dist1.data.cpu().numpy()[0]
                dist2 = dist2.data.cpu().numpy()[0]
                mask1 = dist1 < self.distance
                mask2 = dist2 < self.distance
                if mask1.sum() > self.min_points and angle_simi < self.min_angle:
                    """
                    a, b   4-tuples/lists
                        Ax + By +Cz + D = 0
                        A,B,C,D in order  

                    output: 2 points on line of intersection, np.arrays, shape (3,)
                    """
                    a_vec, b_vec = np.array(plane_a[:3]), np.array(plane_b[:3])

                    aXb_vec = np.cross(a_vec, b_vec)

                    A = np.array([a_vec, b_vec, aXb_vec])
                    d = np.array([-plane_a[3], -plane_b[3], 0.]).reshape(3,1)

                    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

                    p_inter = np.linalg.solve(A, d).T
                    p1, p2 = p_inter, (p_inter + aXb_vec)
                    '''end''' 
                    
                    #The line extending the segment is parameterized as p1 + t (p2 - p1).
                    #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
                    #distance between p1 and p2
                    l2 = np.sum((p1 - p2) ** 2)
                    # project
                    coords_choice_b = coords_b[mask1]
                    coords_choice_a = coords_a[mask2]
                    if coords_choice_a.shape[0] > self.sample_points:
                        choice = np.random.choice(coords_choice_a.shape[0], self.sample_points)
                        coords_choice_a = coords_choice_a[choice]
                        
                    if coords_choice_b.shape[0] > self.sample_points:
                        choice = np.random.choice(coords_choice_b.shape[0], self.sample_points)
                        coords_choice_b = coords_choice_b[choice]
                    
                    t_a = np.sum((coords_choice_a - p1) * (p2 - p1), axis=1) / l2
                    t_b = np.sum((coords_choice_b - p1) * (p2 - p1), axis=1) / l2
                    
                    mask_a = self.filter(plane_a, coords_p_a, p1, p2)
                    mask_b = self.filter(plane_b, coords_p_b, p1, p2)
                    if mask_a is not None:
                        coords_a = coords_a[~mask_a]
                        coords_p_a = coords_p_a[~mask_a]
                        # mask2 = mask2 | mask_a
                    if mask_b is not None:
                        coords_b = coords_b[~mask_b]
                        coords_p_b = coords_p_b[~mask_b]
                        # mask1 = mask1 | mask_b

                    projection_a = p1 + t_a[..., np.newaxis] * (p2 - p1)
                    projection_b = p1 + t_b[..., np.newaxis] * (p2 - p1)
                    coords_list[i] = np.concatenate([coords_a, projection_a])
                    coords_list[j] = np.concatenate([coords_b, projection_b])
                    project_list[i] = np.concatenate([coords_p_a, projection_a])
                    project_list[j] = np.concatenate([coords_p_b, projection_b])
                    # coords_list[i] = np.concatenate([coords_a, projection])
                    # coords_list[j] = np.concatenate([coords_b, projection])
                    # project_list[i] = np.concatenate([coords_p_a, projection])
                    # project_list[j] = np.concatenate([coords_p_b, projection])
        return coords_list, project_list
        
    def save_scene_eval(self, epoch, outputs, inputs, batch_idx=0):
        global_origin = inputs['vol_origin'][batch_idx].cuda()
        label_volume = outputs['label_volume']
        plane_map = outputs['plane_map']
        coords = label_volume.C
        labels = label_volume.F
        labels_unique = torch.unique(labels)
        coords_list = []
        planes_list = []
        project_list = []
        coords_2d_list = []
        faces_list = []
        valid_id = []
        for i, label in enumerate(labels_unique):
            ind = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
            if len(ind) > 10:
                coords_ins = coords[ind]
                coords_ins = coords_ins * self.cfg.MODEL.VOXEL_SIZE + global_origin
                coords_ins = coords_ins.data.cpu().numpy()
                planes_ins = plane_map[label].data.cpu().numpy()
                normals_ins = - planes_ins[:3] / planes_ins[3:4]
                t = (np.matmul(coords_ins, normals_ins) - 1) / (
                        normals_ins[0] ** 2 + normals_ins[1] ** 2 + normals_ins[2] ** 2)
                project_points = coords_ins - normals_ins[np.newaxis, :3] * t[:, np.newaxis]
                coords_list.append(coords_ins)
                planes_list.append(planes_ins)
                project_list.append(project_points)
        
        coords_list, project_list = self.detect_intersection_line(coords_list, project_list, planes_list)
        
        # triangulation
        points_list = []
        planes_final = []
        for coords_ins, planes_ins, project_points in zip(coords_list, planes_list, project_list):
            if coords_ins.shape[0] > 2:
                normals_ins = - planes_ins[:3] / planes_ins[3:4]
                M = project2plane(normals_ins, project_points)
                plane_points_3d = np.concatenate([project_points, np.ones_like(project_points[:, :1])], axis=-1)
                plane_points_2d = np.matmul(M, plane_points_3d.transpose()).transpose()[:, :2]
                
                if not np.isnan(plane_points_2d).any():
                    try:
                        tri = Delaunay(plane_points_2d)

                        face = tri.simplices
                        faces_list.append(face)
                        points_list.append(coords_ins)
                        planes_final.append(planes_ins)
                    except:
                        pass
                            
        planes =  np.stack(planes_final)
        
        mesh, mesh_vis, vertices_eval, points_plane_idx = self.generate_mesh(planes, points_list, faces_list)
        
        save_path = self.log_dir + '_' + str(epoch) + '/' + self.scene_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        mesh.export(os.path.join(save_path, 'planes_mesh.ply'))
        mesh_vis.export(os.path.join(save_path, 'planes_mesh_vis.ply'))
        vertices_eval.export(os.path.join(save_path, 'planes_mesh_eval.ply'))
        np.save(os.path.join(save_path, 'indices'), points_plane_idx)
        
    def __call__(self, outputs, inputs, epoch_idx):
        batch_size = len(inputs['fragment'])
        for i in range(batch_size):
            scene = inputs['scene'][i]
            scene = scene.replace('/', '-')

            if scene != self.scene_name and self.scene_name is not None and self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, inputs, i)

            if scene != self.scene_name or self.scene_name is None:
                self.scene_name = scene
                self.reset()
            else:
                self.keyframe_id += 1
            