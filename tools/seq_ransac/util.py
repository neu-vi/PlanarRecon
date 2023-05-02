import numpy as np
import torch
import sys
import torch.nn.functional as F
import trimesh

from tsdf import coordinates

# grouping the 3 sigma region defined in prepare_data
GROUPING_RADIUS =  {16: 1, 8: 2, 4: 3 }

# for inference
PLANE_MIN_N_VOXELS = {16: 25 ,8: 100, 4: 400 } #0.04**3*1600=0.1024 m3

#  window, blind, pillow, mirror, clothes, shower curtain, person, toilet, sink, lamp, bag
NONE_PLANE_ID = [0, -1, 9, 13, 18, 19, 21, 28, 31, 33, 34,  35, 37 ]

# =================== seq RANSAC using voxels ================
def seq_ransac(coords, normals, semLab,  planeIns, valid_mask, mask_surface,
               norm_thres, radius, area_thres, cur_planeID, n_iter=100):
    # sequential one-point plane ransac, the principle is the same as onePoint_ransan in fit_plane/util,
    # but the code is slightly different  because we do not have the instance level label,
    # so we need consider semantic label consistency here

    # sample the seeds in valid_mask
    # in case replacement == False, and n_iter > (prob > 0).sum(), it will return idx whose weight == 0
    # therefore, we should make sure the n_iter is always < (prob>0).sum()
    idxs = torch.multinomial(valid_mask.float(), min(n_iter, (valid_mask).sum()), replacement=False)

    resume = False
    n_inliers = 0
    best_mask = torch.zeros_like(semLab).type(torch.bool)
    for i in idxs:
        sample_pnt = coords[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)
        sample_semg = semLab[i]

        # semantic should be same
        semseg_mask = (semLab == sample_semg)

        # normal should be similiar
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * coords).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= radius

        # only sign once
        available_mask = planeIns == 0

        cluster_mask = semseg_mask & norm_mask & spatial_mask & available_mask & mask_surface
        n =  cluster_mask.sum()
        if n > n_inliers:
            best_mask = cluster_mask.clone()
            n_inliers = n
            center_coord, center_semseg, center_norm = sample_pnt, sample_semg, sample_norm

    # ransac will stop if the best plane_area < area_thres
    if n_inliers >= area_thres:
        planeIns[best_mask] = cur_planeID
        valid_mask[best_mask] = False # seed will only be assigned once as well
        resume = True

        center_coord = torch.mean(coords[:,best_mask].float(), dim=1, keepdim=True)
        # dist = torch.sum((coords[:, best_mask].float() - mean_center).abs(), dim=0) # use l1 dist find nearest inliers to mean_center
        # new_idx = dist.argmin(dim=0)  # acedend
        # center_coord = coords[:,best_mask][:, new_idx].unsqueeze(1)
        normals[:, best_mask] = center_norm
    else:
        center_coord, center_semseg, center_norm = sample_pnt, sample_semg, sample_norm # just return sth, will not be used

    return planeIns, valid_mask, resume, normals,  center_coord, center_semseg, center_norm

def get_planeIns_RANSAC_raw(tsdf, angle=30, device = torch.device('cpu')):
    # init necessary variables
    voxel_size =  tsdf.voxel_size
    tsdf.tsdf_vol = tsdf.tsdf_vol.to(device)
    normals =  get_normal(tsdf.tsdf_vol)
    if 'semseg' in tsdf.attribute_vols:
        semLab = tsdf.attribute_vols['semseg']
    else:
        semLab = torch.ones_like(tsdf.tsdf_vol)

    radius =  GROUPING_RADIUS[int(voxel_size * 100)]
    norm_thres =  np.cos(np.deg2rad(angle))
    area_thres =  PLANE_MIN_N_VOXELS[int(voxel_size * 100)]

    coords = coordinates(semLab.shape,device=device)

    normals, semLab = normals.reshape([3, -1]), semLab.reshape([-1])
    planeIns = torch.zeros_like(semLab)

    # pick valid center
    mask_surface = tsdf.tsdf_vol.abs() < 1
    mask_surface = mask_surface.reshape([-1])

    if mask_surface.sum() == 0:
        return planeIns, normals, [], [], []

    # start sequential RANSAC for each pred_semantic label
    cur_planeId = 1
    centers, center_segms, center_norms = [], [], []
    for semid in torch.unique(semLab):
        if semid.item() in NONE_PLANE_ID: continue # ignore invalid semantic label
        resume_ransac = True
        tmp_valid_mask = mask_surface & (semLab == semid)
        # Start seq_Ransac,
        while resume_ransac:
            sys.stdout.write('\rprocessing: {}'.format(semid.item()))
            sys.stdout.flush()
            if tmp_valid_mask.sum() == 0: #quit if no seeds exist
                resume_ransac = False
            else:
                planeIns, tmp_valid_mask, resume_ransac,  normals, center_coord, center_semseg, center_norm  =\
                    seq_ransac(coords, normals, semLab,  planeIns, tmp_valid_mask , mask_surface,
                           norm_thres, radius,  area_thres, cur_planeId, n_iter=500)

                if resume_ransac:
                    cur_planeId += 1
                    centers.append(center_coord)
                    center_segms.append(center_semseg)
                    center_norms.append(center_norm)


    return planeIns, normals,  centers, center_segms, center_norms

def get_normal( tsdf_vol):
    # refer to https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    # Note the tsdf coordiate are x y z
    # mask = ~torch.logical_or (tsdf_vol == 1, tsdf_vol==-1)
    # replicate usage
    if len(tsdf_vol.shape) == 3:
        tsdf_vol = tsdf_vol.unsqueeze(0).unsqueeze(0)
    pad_vol = F.pad(tsdf_vol, (1, 1, 1, 1, 1, 1),
                    mode="replicate")  # pad each dim 1,1 to compute grad
    nx = (pad_vol[:,:, 2:, :, :] - pad_vol[:,:, :-2, :, :])[:,:, :, 1:-1, 1:-1]
    ny = (pad_vol[:,:, :, 2:, :] - pad_vol[:,:, :, :-2, :])[:,:, 1:-1, :, 1:-1]
    nz = (pad_vol[:,:, :, :, 2:] - pad_vol[:,:, :, :, :-2])[:,:, 1:-1, 1:-1, :]

    normal = torch.cat([nx, ny, nz], dim=1) # concat in channel dim

    normal /= normal.norm(dim=1)
    normal[normal != normal] = 0  # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4 set nan to 0
    return normal.squeeze()



# ============================== RANSAC using Mesh ====================
def find_largest_connected_componenet(occup, connect_kernel_sz = 3):
    h, w, z = occup.shape
    seeds = torch.arange(0, occup.view(-1).shape[0]).reshape([h,w,z]).float().view(1,1,h,w,z).to(occup.device)
    _occup = occup.view(1,1,h,w,z)
    seeds[~_occup] = 0

    pre_mask = seeds.clone()
    candidate_mask = seeds.clone()

    # use 3D max pooling to propgate seed
    for cnt in range(max([h, w, z])):  # longest dist to flood fill equals to the largest dim
        candidate_mask = F.max_pool3d(candidate_mask, kernel_size=connect_kernel_sz, stride=1, padding=connect_kernel_sz//2)
        candidate_mask[~_occup] = 0

        if  (pre_mask == candidate_mask).all():
            break
        pre_mask = candidate_mask.clone()

    print('find largest connection in ', cnt, 'steps')
    # take the most freq value
    freq_val , _ = torch.mode(candidate_mask[candidate_mask > 0].view(-1))

    return (candidate_mask == freq_val).squeeze()


def seq_ransac_mesh(verts, normals, occup,  planeIns, voxel_sz, origin, connect_kernel,
               norm_thres, dist_thres, init_verts_thres, cur_planeID, n_iter=100):
    # sequential one-point plane ransac using mesh as input

    # every verts can at most be assigned once
    valid_mask = planeIns == 0

    voxel_verts = (verts - origin.view(-1, 1)) / voxel_sz
    voxel_verts_ind = torch.round(voxel_verts).long()

    # sample the seeds in valid_mask, we must ensure the sampled number is <= valid_mask points
    idxs = torch.multinomial(valid_mask.float(), min(n_iter, (valid_mask).sum()), replacement=False)

    resume = False
    cur_inlier_vol = 0
    best_mask = torch.zeros_like(valid_mask).type(torch.bool)
    for i in idxs:
        sample_pnt = verts[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)

        # normal should be similiar,
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * verts).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= dist_thres

        cluster_mask = valid_mask & norm_mask & spatial_mask
        # occup_area = cluster_mask.clone()

        # check occupied volume
        fill_volum = torch.zeros_like(occup)

        # convert verts into volume space, and select all voxels which are current inliners and be occupied
        inlier_verts = verts[:, cluster_mask]
        inlier_voxel_verts = (inlier_verts -  origin.view(-1, 1)) / voxel_sz
        inlier_verts_ind =  torch.round(inlier_voxel_verts).long()
        fill_volum[inlier_verts_ind[0],inlier_verts_ind[1],inlier_verts_ind[2]] = True

        _occup_area = torch.logical_and(fill_volum, occup)
        occup_area = find_largest_connected_componenet(_occup_area, connect_kernel)

        # pick the verts
        final_mask = torch.where(occup_area[voxel_verts_ind[0],voxel_verts_ind[1],voxel_verts_ind[2]],
                                 torch.ones_like(voxel_verts_ind[0]), torch.zeros_like(voxel_verts_ind[0])).bool()

        proposal_vol =  occup_area.sum()
        if proposal_vol > cur_inlier_vol:
            best_mask = final_mask.clone()
            cur_inlier_vol = proposal_vol

    # ransac will stop if the best plane_area < area_thres
    if cur_inlier_vol >= init_verts_thres:
        planeIns[best_mask] = cur_planeID
        resume = True
        plane_param , _ = torch.lstsq(torch.ones_like(verts[:1, best_mask].T), verts[:,best_mask].T) # B, A, return X of min |AX - B|
        plane_param = plane_param[:3] # first 3 is the solution
    else:
        plane_param = None # just return sth, will not be used

    return planeIns, resume, plane_param


def seq_ransac_mesh_approx(verts, normals, occup,  planeIns, voxel_sz, origin, connect_kernel,
               norm_thres, dist_thres, init_verts_thres, connect_verts_thres, cur_planeID, n_iter=100):
    # sequential one-point plane ransac using mesh as input
    # the difference with seq_ransac_mesh() is here, we do the connection check as a post-process
    # instead of one step of fitting, as connection_check take lots of times

    # every verts can at most be assigned once
    valid_mask = planeIns == 0

    voxel_verts = (verts - origin.view(-1, 1)) / voxel_sz
    voxel_verts_ind = torch.round(voxel_verts).long()

    # sample the seeds in valid_mask, we must ensure the sampled number is <= valid_mask points
    idxs = torch.multinomial(valid_mask.float(), min(n_iter, (valid_mask).sum()), replacement=False)

    resume = False
    best_inlier_vol = 0
    plane_param = None
    best_mask = torch.zeros_like(valid_mask).type(torch.bool)
    for i in idxs:
        sample_pnt = verts[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)

        # normal should be similiar,
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * verts).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= dist_thres

        cluster_mask = valid_mask & norm_mask & spatial_mask
        # occup_area = cluster_mask.clone()

        proposal_vol =  cluster_mask.sum()
        if proposal_vol > best_inlier_vol:
            best_mask = cluster_mask.clone()
            best_inlier_vol = proposal_vol

    # ransac will stop if the best plane_area < area_thres
    if best_inlier_vol >= init_verts_thres:

        # check occupied volume
        fill_volum = torch.zeros_like(occup)

        # convert verts into volume space, and select all voxels which are current inliners and be occupied
        inlier_verts = verts[:, best_mask]
        inlier_voxel_verts = (inlier_verts - origin.view(-1, 1)) / voxel_sz
        inlier_verts_ind = torch.round(inlier_voxel_verts).long()
        fill_volum[inlier_verts_ind[0], inlier_verts_ind[1], inlier_verts_ind[2]] = True

        _occup_area = torch.logical_and(fill_volum, occup)
        occup_area = find_largest_connected_componenet(_occup_area, connect_kernel)

        # pick the verts within the largest connected componenet
        final_mask = torch.where(occup_area[voxel_verts_ind[0], voxel_verts_ind[1], voxel_verts_ind[2]],
                                 torch.ones_like(voxel_verts_ind[0]), torch.zeros_like(voxel_verts_ind[0])).bool()

        print('instance verts number ', final_mask.sum().item())

        # final lstsq to get the result
        if final_mask.sum() >= connect_verts_thres: # we ask at least 4 verts for plane fittng
            planeIns[final_mask] = cur_planeID
            resume = True
            plane_param, _ = torch.lstsq(torch.ones_like(verts[:1, final_mask].T),
                                         verts[:, final_mask].T)  # B, A, return X of min |AX - B|
            plane_param = plane_param[:3]  # first 3 is the solution

    return planeIns, resume, plane_param # plane_param == None and will not be used if no plane is found



def get_planeIns_RANSAC_mesh(verts, norms, tsdf, voxel_sz, origin, connect_kernel,
                             angle_thres=30, dist_thres = 0.05, init_verts_thres= 200,  connect_verts_thres=4,
                             n_iter= 100, device = torch.device('cpu')):
    # angle thres: for normal angle, degree
    # dist_thres: pnt to plane distance, meter
    # area_thres: number of vertis in a plane proposal

    # move to CUDA if needed
    verts = verts.to(device)
    norms = norms.to(device)
    occupancy = tsdf.to(device).abs() < 1
    origin = origin.to(device)

    # init necessary variables
    norm_thres =  np.cos(np.deg2rad(angle_thres))
    planeIns = torch.zeros_like(verts[0])

    # start sequential RANSAC for each pred_semantic label
    resume_ransac = True
    cur_planeId = 1
    plane_params  = []
    while resume_ransac:
        sys.stdout.write('\rprocessing: {}'.format(cur_planeId))
        sys.stdout.flush()

        planeIns, resume_ransac, plane_param = \
            seq_ransac_mesh_approx(verts, norms, occupancy, planeIns, voxel_sz, origin, connect_kernel,
                       norm_thres, dist_thres, init_verts_thres, connect_verts_thres, cur_planeId, n_iter=n_iter)


        if resume_ransac:
            cur_planeId += 1
            plane_params.append(plane_param)

    return planeIns, plane_params

# ========= planarize ==========
def planarize(verts, plane_ins, plane_params, n_plane):
    new_verts = verts.clone()
    for i in range(1, n_plane): # skip the non_plane
        plane_mask = plane_ins==i
        param = plane_params[i-1] # convert to 0-idx
        _planeD = param.norm()
        planeN, planeD = param/_planeD, 1/_planeD

        plane_verts = new_verts[:, plane_mask]

        # proj
        dist = (planeN.T @ plane_verts - planeD) * planeN
        on_plane_verts = plane_verts - dist
        new_verts[:, plane_mask] = on_plane_verts

    return new_verts

def break_faces(faces, plane_ins):
    new_faces = []
    for face in faces:
        ins_id_1 = plane_ins[face[0]]
        ins_id_2 = plane_ins[face[1]]
        ins_id_3 = plane_ins[face[2]]

        if ins_id_1 == ins_id_2 and ins_id_1 == ins_id_3:
            new_faces.append(face)
    ret = np.stack(new_faces,axis=0)
    return ret

def extract_plane_surface(plane_ins, piecewise_mesh):
    all_color = piecewise_mesh.visual.vertex_colors
    verts = piecewise_mesh.vertices
    plane_verts = verts[plane_ins > 0]

    # deal with face
    face_color = piecewise_mesh.visual.face_colors
    face_mask = face_color[:, :3].sum(axis=1) > 0 # exclude all zeros (non-plane) color
    init_plane_mesh = piecewise_mesh.submesh(np.nonzero(face_mask))[0]
    init_color = init_plane_mesh.visual.vertex_colors

    # some of verts may not include in init_plane_mesh, due to the broken face
    init_verts = init_plane_mesh.vertices
    non_tri_pnts = np.setdiff1d(plane_verts.view(dtype='f8,f8,f8').reshape(plane_verts.shape[0]),
                                init_verts.view(dtype='f8,f8, f8').reshape(init_verts.shape[0]))
    non_tri_pnts = non_tri_pnts.view(dtype='f8').reshape([non_tri_pnts.shape[0], 3])

    # to get non-tri pnts color
    vmask = np.in1d(verts.view(dtype='f8,f8,f8').reshape(verts.shape[0]),
                    non_tri_pnts.view(dtype='f8,f8, f8').reshape(non_tri_pnts.shape[0]))
    non_tri_color = all_color[vmask]

    final_verts = np.concatenate([init_verts, non_tri_pnts], axis=0)
    final_colors = np.concatenate([init_color, non_tri_color], axis=0)

    # get final mesh
    final_plane_mesh = trimesh.Trimesh(vertices=final_verts, faces=init_plane_mesh.faces,
                                       vertex_colors=final_colors, process=False)

    return final_plane_mesh


# ============ for viz =====================

def get_unique_colors(ret_n = 10):
    # all color based on
    #  https://stackoverflow.com/questions/33295120/how-to-generate-gif-256-colors-palette

    colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B", "#1E2324", "#DEC9B2", "#9D4948",
        "#85ABB4", "#342142", "#D09685", "#A4ACAC", "#00FFFF", "#AE9C86", "#742A33", "#0E72C5",
        "#AFD8EC", "#C064B9", "#91028C", "#FEEDBF", "#FFB789", "#9CB8E4", "#AFFFD1", "#2A364C",
        "#4F4A43", "#647095", "#34BBFF", "#807781", "#920003", "#B3A5A7", "#018615", "#F1FFC8",
        "#976F5C", "#FF3BC1", "#FF5F6B", "#077D84", "#F56D93", "#5771DA", "#4E1E2A", "#830055",
        "#02D346", "#BE452D", "#00905E", "#BE0028", "#6E96E3", "#007699", "#FEC96D", "#9C6A7D",
        "#3FA1B8", "#893DE3", "#79B4D6", "#7FD4D9", "#6751BB", "#B28D2D", "#E27A05", "#DD9CB8",
        "#AABC7A", "#980034", "#561A02", "#8F7F00", "#635000", "#CD7DAE", "#8A5E2D", "#FFB3E1",
        "#6B6466", "#C6D300", "#0100E2", "#88EC69", "#8FCCBE", "#21001C", "#511F4D", "#E3F6E3",
        "#FF8EB1", "#6B4F29", "#A37F46", "#6A5950", "#1F2A1A", "#04784D", "#101835", "#E6E0D0",
        "#FF74FE", "#00A45F", "#8F5DF8", "#4B0059", "#412F23", "#D8939E", "#DB9D72", "#604143",
        "#B5BACE", "#989EB7", "#D2C4DB", "#A587AF", "#77D796", "#7F8C94", "#FF9B03", "#555196",
        "#31DDAE", "#74B671", "#802647", "#2A373F", "#014A68", "#696628", "#4C7B6D", "#002C27",
        "#7A4522", "#3B5859", "#E5D381", "#FFF3FF", "#679FA0", "#261300", "#2C5742", "#9131AF",
        "#AF5D88", "#C7706A", "#61AB1F", "#8CF2D4", "#C5D9B8", "#9FFFFB", "#BF45CC", "#493941",
        "#863B60", "#B90076", "#003177", "#C582D2", "#C1B394", "#602B70", "#887868", "#BABFB0",
        "#030012", "#D1ACFE", "#7FDEFE", "#4B5C71", "#A3A097", "#E66D53", "#637B5D", "#92BEA5",
        "#00F8B3", "#BEDDFF", "#3DB5A7", "#DD3248", "#B6E4DE", "#427745", "#598C5A", "#B94C59",
        "#8181D5", "#94888B", "#FED6BD", "#536D31", "#6EFF92", "#E4E8FF", "#20E200", "#FFD0F2",
        "#4C83A1", "#BD7322", "#915C4E", "#8C4787", "#025117", "#A2AA45", "#2D1B21", "#A9DDB0",
        "#FF4F78", "#528500", "#009A2E", "#17FCE4", "#71555A", "#525D82", "#00195A", "#967874",
        "#555558", "#0B212C", "#1E202B", "#EFBFC4", "#6F9755", "#6F7586", "#501D1D", "#372D00",
        "#741D16", "#5EB393", "#B5B400", "#DD4A38", "#363DFF", "#AD6552", "#6635AF", "#836BBA",
        "#98AA7F", "#464836", "#322C3E", "#7CB9BA", "#5B6965", "#707D3D", "#7A001D", "#6E4636",
        "#443A38", "#AE81FF", "#489079", "#897334", "#009087", "#DA713C", "#361618", "#FF6F01",
        "#006679", "#370E77", "#4B3A83", "#C9E2E6", "#C44170", "#FF4526", "#73BE54", "#C4DF72",
        "#ADFF60", "#00447D", "#DCCEC9", "#BD9479", "#656E5B", "#EC5200", "#FF6EC2", "#7A617E",
        "#DDAEA2", "#77837F", "#A53327", "#608EFF", "#B599D7", "#A50149", "#4E0025", "#C9B1A9",
        "#03919A", "#1B2A25", "#E500F1", "#982E0B", "#B67180", "#E05859", "#006039", "#578F9B",
        "#305230", "#CE934C", "#B3C2BE", "#C0BAC0", "#B506D3", "#170C10", "#4C534F", "#224451",
        "#3E4141", "#78726D", "#B6602B", "#200441", "#DDB588", "#497200", "#C5AAB6", "#033C61",
        "#71B2F5", "#A9E088", "#4979B0", "#A2C3DF", "#784149", "#2D2B17", "#3E0E2F", "#57344C",
        "#0091BE", "#E451D1", "#4B4B6A", "#5C011A", "#7C8060", "#FF9491", "#4C325D", "#005C8B",
        "#E5FDA4", "#68D1B6", "#032641", "#140023", "#8683A9", "#CFFF00", "#A72C3E", "#34475A",
        "#B1BB9A", "#B4A04F", "#8D918E", "#A168A6", "#813D3A", "#425218", "#DA8386", "#776133",
        "#563930", "#8498AE", "#90C1D3", "#B5666B", "#9B585E", "#856465", "#AD7C90", "#E2BC00",
        "#E3AAE0", "#B2C2FE", "#FD0039", "#009B75", "#FFF46D", "#E87EAC", "#DFE3E6", "#848590",
        "#AA9297", "#83A193", "#577977", "#3E7158", "#C64289", "#EA0072", "#C4A8CB", "#55C899",
        "#E78FCF", "#004547", "#F6E2E3", "#966716", "#378FDB", "#435E6A", "#DA0004", "#1B000F",
        "#5B9C8F", "#6E2B52", "#011115", "#E3E8C4", "#AE3B85", "#EA1CA9", "#FF9E6B", "#457D8B",
        "#92678B", "#00CDBB", "#9CCC04", "#002E38", "#96C57F", "#CFF6B4", "#492818", "#766E52",
        "#20370E", "#E3D19F", "#2E3C30", "#B2EACE", "#F3BDA4", "#A24E3D", "#976FD9", "#8C9FA8",
        "#7C2B73", "#4E5F37", "#5D5462", "#90956F", "#6AA776", "#DBCBF6", "#DA71FF", "#987C95",
        "#52323C", "#BB3C42", "#584D39", "#4FC15F", "#A2B9C1", "#79DB21", "#1D5958", "#BD744E",
        "#160B00", "#20221A", "#6B8295", "#00E0E4", "#102401", "#1B782A", "#DAA9B5", "#B0415D",
        "#859253", "#97A094", "#06E3C4", "#47688C", "#7C6755", "#075C00", "#7560D5", "#7D9F00",
        "#C36D96", "#4D913E", "#5F4276", "#FCE4C8", "#303052", "#4F381B", "#E5A532", "#706690",
        "#AA9A92", "#237363", "#73013E", "#FF9079", "#A79A74", "#029BDB", "#FF0169", "#C7D2E7",
        "#CA8869", "#80FFCD", "#BB1F69", "#90B0AB", "#7D74A9", "#FCC7DB", "#99375B", "#00AB4D",
        "#ABAED1", "#BE9D91", "#E6E5A7", "#332C22", "#DD587B", "#F5FFF7", "#5D3033", "#6D3800",
        "#FF0020", "#B57BB3", "#D7FFE6", "#C535A9", "#260009", "#6A8781", "#A8ABB4", "#D45262",
        "#794B61", "#4621B2", "#8DA4DB", "#C7C890", "#6FE9AD", "#A243A7", "#B2B081", "#181B00",
        "#286154", "#4CA43B", "#6A9573", "#A8441D", "#5C727B", "#738671", "#D0CFCB", "#897B77",
        "#1F3F22", "#4145A7", "#DA9894", "#A1757A", "#63243C", "#ADAAFF", "#00CDE2", "#DDBC62",
        "#698EB1", "#208462", "#00B7E0", "#614A44", "#9BBB57", "#7A5C54", "#857A50", "#766B7E",
        "#014833", "#FF8347", "#7A8EBA", "#274740", "#946444", "#EBD8E6", "#646241", "#373917",
        "#6AD450", "#81817B", "#D499E3", "#979440", "#011A12", "#526554", "#B5885C", "#A499A5",
        "#03AD89", "#B3008B", "#E3C4B5", "#96531F", "#867175", "#74569E", "#617D9F", "#E70452",
        "#067EAF", "#A697B6", "#B787A8", "#9CFF93", "#311D19", "#3A9459", "#6E746E", "#B0C5AE",
        "#84EDF7", "#ED3488", "#754C78", "#384644", "#C7847B", "#00B6C5", "#7FA670", "#C1AF9E",
        "#2A7FFF", "#72A58C", "#FFC07F", "#9DEBDD", "#D97C8E", "#7E7C93", "#62E674", "#B5639E",
        "#FFA861", "#C2A580", "#8D9C83", "#B70546", "#372B2E", "#0098FF", "#985975", "#20204C",
        "#FF6C60", "#445083", "#8502AA", "#72361F", "#9676A3", "#484449", "#CED6C2", "#3B164A",
        "#CCA763", "#2C7F77", "#02227B", "#A37E6F", "#CDE6DC", "#CDFFFB", "#BE811A", "#F77183",
        "#EDE6E2", "#CDC6B4", "#FFE09E", "#3A7271", "#FF7B59", "#4E4E01", "#4AC684", "#8BC891",
        "#BC8A96", "#CF6353", "#DCDE5C", "#5EAADD", "#F6A0AD", "#E269AA", "#A3DAE4", "#436E83",
        "#002E17", "#ECFBFF", "#A1C2B6", "#50003F", "#71695B", "#67C4BB", "#536EFF", "#5D5A48",
        "#890039", "#969381", "#371521", "#5E4665", "#AA62C3", "#8D6F81", "#2C6135", "#410601",
        "#564620", "#E69034", "#6DA6BD", "#E58E56", "#E3A68B", "#48B176", "#D27D67", "#B5B268",
        "#7F8427", "#FF84E6", "#435740", "#EAE408", "#F4F5FF", "#325800", "#4B6BA5", "#ADCEFF",
        "#9B8ACC", "#885138", "#5875C1", "#7E7311", "#FEA5CA", "#9F8B5B", "#A55B54", "#89006A",
        "#AF756F", "#2A2000", "#576E4A", "#7F9EFF", "#7499A1", "#FFB550", "#00011E", "#D1511C",
        "#688151", "#BC908A", "#78C8EB", "#8502FF", "#483D30", "#C42221", "#5EA7FF", "#785715",
        "#0CEA91", "#FFFAED", "#B3AF9D", "#3E3D52", "#5A9BC2", "#9C2F90", "#8D5700", "#ADD79C",
        "#00768B", "#337D00", "#C59700", "#3156DC", "#944575", "#ECFFDC", "#D24CB2", "#97703C",
        "#4C257F", "#9E0366", "#88FFEC", "#B56481", "#396D2B", "#56735F", "#988376", "#9BB195",
        "#A9795C", "#E4C5D3", "#9F4F67", "#1E2B39", "#664327", "#AFCE78", "#322EDF", "#86B487",
        "#C23000", "#ABE86B", "#96656D", "#250E35", "#A60019", "#0080CF", "#CAEFFF", "#323F61",
        "#A449DC", "#6A9D3B", "#FF5AE4", "#636A01", "#D16CDA", "#736060", "#FFBAAD", "#D369B4",
        "#FFDED6", "#6C6D74", "#927D5E", "#845D70", "#5B62C1", "#2F4A36", "#E45F35", "#FF3B53",
        "#AC84DD", "#762988", "#70EC98", "#408543", "#2C3533", "#2E182D", "#323925", "#19181B",
        "#2F2E2C", "#023C32", "#9B9EE2", "#58AFAD", "#5C424D", "#7AC5A6", "#685D75", "#B9BCBD",
        "#834357", "#1A7B42", "#2E57AA", "#E55199", "#316E47", "#CD00C5", "#6A004D", "#7FBBEC",
        "#F35691", "#D7C54A", "#62ACB7", "#CBA1BC", "#A28A9A", "#6C3F3B", "#FFE47D", "#DCBAE3",
        "#5F816D", "#3A404A", "#7DBF32", "#E6ECDC", "#852C19", "#285366", "#B8CB9C", "#0E0D00",
        "#4B5D56", "#6B543F", "#E27172", "#0568EC", "#2EB500", "#D21656", "#EFAFFF", "#682021",
        "#2D2011", "#DA4CFF", "#70968E", "#FF7B7D", "#4A1930", "#E8C282", "#E7DBBC", "#A68486",
        "#1F263C", "#36574E", "#52CE79", "#ADAAA9", "#8A9F45", "#6542D2", "#00FB8C", "#5D697B",
        "#CCD27F", "#94A5A1", "#790229", "#E383E6", "#7EA4C1", "#4E4452", "#4B2C00", "#620B70",
        "#314C1E", "#874AA6", "#E30091", "#66460A", "#EB9A8B", "#EAC3A3", "#98EAB3", "#AB9180",
        "#B8552F", "#1A2B2F", "#94DDC5", "#9D8C76", "#9C8333", "#94A9C9", "#392935", "#8C675E",
        "#CCE93A", "#917100", "#01400B", "#449896", "#1CA370", "#E08DA7", "#8B4A4E", "#667776",
        "#4692AD", "#67BDA8", "#69255C", "#D3BFFF", "#4A5132", "#7E9285", "#77733C", "#E7A0CC",
        "#51A288", "#2C656A", "#4D5C5E", "#C9403A", "#DDD7F3", "#005844", "#B4A200", "#488F69",
        "#858182", "#D4E9B9", "#3D7397", "#CAE8CE", "#D60034", "#AA6746", "#9E5585", "#BA6200"]

    assert len(colors) > ret_n
    ret_color = np.zeros([ret_n, 3])
    for i in range(ret_n):
        hex_color = colors[i][1:]
        ret_color[i] = np.array([int(hex_color[j:j + 2], 16) for j in (0, 2, 4)])
    return ret_color