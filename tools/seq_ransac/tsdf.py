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

#  Originating Author: Zak Murez (zak.murez.com)

from matplotlib.cm import get_cmap as colormap
import numpy as np
from skimage import measure
import torch
import trimesh

# TODO: move to another file and support other colormaps
NYU40_COLORMAP = [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


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


def depth_to_world(projection, depth):
    """ backprojects depth maps to point clouds
    Args:
        projection: 3x4 projection matrix
        depth: hxw depth map

    Returns:
        tensor of 3d points 3x(h*w)
    """

    # add row to projection 3x4 -> 4x4
    eye_row = torch.tensor([[0,0,0,1]]).type_as(depth)
    projection = torch.cat((projection, eye_row))

    # pixel grid
    py, px = torch.meshgrid(torch.arange(depth.size(-2)).type_as(depth),
                            torch.arange(depth.size(-1)).type_as(depth))
    pz = torch.ones_like(px)
    p = torch.cat((px.unsqueeze(0), py.unsqueeze(0), pz.unsqueeze(0), 
                   1/depth.unsqueeze(0)))

    # backproject
    P = (projection.inverse() @ p.view(4,-1)).view(p.size())
    P = P[:3]/P[3:]
    return P


class TSDF():
    """ class to hold a truncated signed distance function (TSDF)

    Holds the TSDF volume along with meta data like voxel size and origin
    required to interpret the tsdf tensor.
    Also implements basic opperations on a TSDF like extracting a mesh.

    """

    def __init__(self, voxel_size, origin, tsdf_vol, attribute_vols=None, 
                 attributes=None):
        """
        Args:
            voxel_size: metric size of voxels (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            tsdf_vol: tensor of size hxwxd containing the TSDF values
            attribute_vols: dict of additional voxel volume data
                example: {'semseg':semseg} can be used to store a 
                    semantic class id for each voxel
            attributes: dict of additional non voxel volume data (ex: instance
                labels, instance centers, ...)
        """

        self.voxel_size = voxel_size
        self.origin = origin
        self.tsdf_vol = tsdf_vol
        if attribute_vols is not None:
            self.attribute_vols = attribute_vols
        else:
            self.attribute_vols = {}
        if attributes is not None:
            self.attributes = attributes 
        else:
            self.attributes = {}
        self.device = tsdf_vol.device

    def save(self, fname):
        data = {'origin': self.origin.cpu().numpy(),
                'voxel_size': self.voxel_size,
                'tsdf': self.tsdf_vol.detach().cpu().numpy()}
        for key, value in self.attribute_vols.items():
            data[key] = value.detach().cpu().numpy()
        for key, value in self.attributes.items():
            data[key] = value.cpu().numpy()
        np.savez_compressed(fname, **data)

    @classmethod
    def load(cls, fname, voxel_types=None):
        """ Load a tsdf from disk (stored as npz).

        Args:
            fname: path to archive
            voxel_types: list of strings specifying which volumes to load 
                ex ['tsdf', 'color']. tsdf is loaded regardless.
                to load all volumes in archive use None (default)

        Returns:
            TSDF
        """
        
        with np.load(fname) as data:
            voxel_size = data['voxel_size'].item()
            origin = torch.as_tensor(data['origin']).view(1,3)
            tsdf_vol = torch.as_tensor(data['tsdf'])
            attribute_vols = {}
            attributes     = {}
            if 'color' in data and (voxel_types is None or 'color' in voxel_types):
                attribute_vols['color'] = torch.as_tensor(data['color'])
            if ('instance' in data and (voxel_types is None or 
                                        'instance' in voxel_types or 
                                        'semseg' in voxel_types)):
                attribute_vols['instance'] = torch.as_tensor(data['instance'])
            ret = cls(voxel_size, origin, tsdf_vol, attribute_vols, attributes)
        return ret

    def to(self, device):
        """ Move tensors to a device"""

        self.origin = self.origin.to(device)
        self.tsdf_vol = self.tsdf_vol.to(device)
        self.attribute_vols = {key:value.to(device) 
                               for key, value in self.attribute_vols.items()}
        self.attributes = {key:value.to(device) 
                           for key, value in self.attributes.items()}
        self.device = device
        return self

    def get_mesh(self, attribute='color', cmap='nyu40'):
        """ Extract a mesh from the TSDF using marching cubes

        If TSDF also has atribute_vols, these are extracted as
        vertex_attributes. The mesh is also colored using the cmap 

        Args:
            attribute: which tsdf attribute is used to color the mesh
            cmap: colormap for converting the attribute to a color

        Returns:
            trimesh.Trimesh
        """

        tsdf_vol = self.tsdf_vol.detach().clone()

        # measure.marching_cubes() likes positive 
        # values in front of surface
        tsdf_vol = -tsdf_vol

        # don't close surfaces using unknown-empty boundry
        tsdf_vol[tsdf_vol==-1]=1

        tsdf_vol = tsdf_vol.clamp(-1,1).cpu().numpy()

        if tsdf_vol.min()>=0 or tsdf_vol.max()<=0:
            return trimesh.Trimesh(vertices=np.zeros((0,3)))

        verts, faces, _, _ = measure.marching_cubes(tsdf_vol, level=0)

        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + self.origin.cpu().numpy()

        vertex_attributes = {}
        # get vertex attributes
        if 'semseg' in self.attribute_vols:
            semseg_vol = self.attribute_vols['semseg'].detach().cpu().numpy()
            semseg = semseg_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]] # the semantic label is copy from the voxel label whre the verts located
            vertex_attributes['semseg'] = semseg

        if 'instance' in self.attribute_vols:
            instance_vol = self.attribute_vols['instance']
            instance_vol = instance_vol.detach().cpu().numpy()
            instance = instance_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
            vertex_attributes['instance'] = instance

        # add Fengting 11/06 for plane ins estimation
        if 'plane_ins' in self.attribute_vols:
            planeIns_vol = self.attribute_vols['plane_ins'].detach().cpu().numpy()
            planeIns_vol = planeIns_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            vertex_attributes['plane_ins'] = planeIns_vol

        # color mesh
        if attribute=='color' and 'color' in self.attribute_vols:
            color_vol = self.attribute_vols['color']
            color_vol = color_vol.detach().clamp(0,255).byte().cpu().numpy()
            colors = color_vol[:, verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]].T
        elif attribute=='instance':
            label_viz = instance+1
            n=label_viz.max()
            cmap = (colormap('jet')(np.linspace(0,1,n))[:,:3]*255).astype(np.uint8)
            cmap = cmap[np.random.permutation(n),:]
            cmap = np.insert(cmap,0,[0,0,0],0)
            colors = cmap[label_viz,:]
        elif attribute=='semseg':
            if cmap=='nyu40':
                cmap = np.array(NYU40_COLORMAP) # FIXME: support more general colormaps
            else:
                raise NotImplementedError('colormap %s'%cmap)
            label_viz = semseg.copy()
            label_viz[(label_viz<0) | (label_viz>=len(cmap))]=0
            colors = cmap[label_viz,:]
        elif 'plane_ins'  in attribute and ('plane_ins' in self.attribute_vols):
            n_plane = (planeIns_vol.max() + 1).type(torch.int) # is not equal to the real pred plane num, but the largest plane idx
            # if no center is detected
            if n_plane == 0:
                return trimesh.Trimesh(vertices=np.zeros((1, 3)))
            else:
                plane_color = self.get_planeColor_map(n_plane)
                colors = plane_color[planeIns_vol,:]
                return   trimesh.Trimesh(
                            vertices=verts, faces=faces, vertex_colors=colors, process=False)
        else:
            colors = None

        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces, vertex_colors=colors,
            vertex_attributes=vertex_attributes, process=False)
        return mesh

    def get_planeColor_map(self, n_plane):
        # get color for plane instance Fengting 2020/11/07
        _cmap = np.array(NYU40_COLORMAP)
        if n_plane - 41 > 0:
            cmap = (colormap('jet')(np.linspace(0, 1, n_plane - 41))[:, :3] * 255).astype(np.uint8)
            cmap = cmap[np.random.permutation(n_plane - 41), :]
            plane_color = np.concatenate([_cmap, cmap], axis=0)
        else:
            plane_color = _cmap

        return plane_color

    def transform(self, transform=None, voxel_dim=None, origin=None,
                  align_corners=False):
        """ Applies a 3x4 linear transformation to the TSDF.

        Each voxel is moved according to the transformation and a new volume
        is constructed with the result.

        Args:
            transform: 3x4 linear transform
            voxel_dim: size of output voxel volume to construct (nx,ny,nz)
                default (None) is the same size as the input
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
                default (None) is the same as the input
        
        Returns:
            A new TSDF with the transformed coordinates
        """

        device = self.tsdf_vol.device

        old_voxel_dim = list(self.tsdf_vol.size())
        old_origin = self.origin

        if transform is None:
            transform = torch.eye(4, device=device)
        if voxel_dim is None:
            voxel_dim = old_voxel_dim
        if origin is None:
            origin = old_origin
        else:
            origin = torch.tensor(origin, dtype=torch.float, device=device).view(1,3)
        # print('transform', origin)
        coords = coordinates(voxel_dim, device) #coordinate of out size
        world = coords.type(torch.float) * self.voxel_size + origin.T
        world = torch.cat((world, torch.ones_like(world[:1]) ), dim=0)
        world = transform[:3,:] @ world
        coords = (world - old_origin.T) / self.voxel_size

        # grid sample expects coords in [-1,1]
        coords = 2*coords/(torch.tensor(old_voxel_dim, device=device)-1).view(3,1)-1 # move the origin to the center
        coords = coords[[2,1,0]].T.view([1]+voxel_dim+[3]) # (z, y, x) for each voxel

        # bilinear interpolation near surface,
        # no interpolation along -1,1 boundry
        tsdf_vol = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim),
            coords, mode='nearest', align_corners=align_corners
        ).squeeze()
        tsdf_vol_bilin = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim), coords, mode='bilinear',
            align_corners=align_corners
        ).squeeze()
        mask = tsdf_vol.abs()<1
        tsdf_vol[mask] = tsdf_vol_bilin[mask]

        # padding_mode='ones' does not exist for grid_sample so replace 
        # elements that were on the boarder with 1.
        # voxels beyond full volume (prior to croping) should be marked as empty
        mask = (coords.abs()>=1).squeeze(0).any(3)
        tsdf_vol[mask] = 1

        # transform attribute_vols
        attribute_vols={}
        for key, value in self.attribute_vols.items():
            dtype = value.dtype
            if len(value.size())==3:
                channels=1
            else:
                channels=value.size(0)
            value = value.view([1,channels]+old_voxel_dim).float()
            mode = 'bilinear' if dtype==torch.float else 'nearest'
            attribute_vols[key] = torch.nn.functional.grid_sample(
                value, coords, mode=mode, align_corners=align_corners
            ).squeeze().type(dtype)

            if key=='mask_outside':
                attribute_vols[key][mask] = True
            elif key=='semseg':
                attribute_vols[key][mask] = -1

        # TODO: transform attributes
        attributes = self.attributes

        return TSDF(self.voxel_size, origin, tsdf_vol, attribute_vols, attributes)



