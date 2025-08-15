import os
import glob
import numpy as np
import torch
import trimesh
import sys

# sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plyfile import PlyData, PlyElement
from mesh import Mesh
from lib.gs_utils.gs_renderer_pose_smplx import (
    find_points_inside_triangle, RGB2SH, SH2RGB,
    calculate_sides_and_heights, batch_vector_to_quaternion,
    inverse_sigmoid, normal_vector_of_plane_batch
)

def construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape, objects_dc_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    l += [f'f_dc_{i}' for i in range(features_dc_shape[1] * features_dc_shape[2])]
    l += [f'f_rest_{i}' for i in range(features_rest_shape[1] * features_rest_shape[2])]
    l.append('opacity')
    l += [f'scale_{i}' for i in range(scaling_shape[1])]
    l += [f'rot_{i}' for i in range(rotation_shape[1])]
    l += [f'obj_dc_{i}' for i in range(objects_dc_shape[1] * objects_dc_shape[2])]
    return l

def save_ply(path, xyz, features_dc, features_rest, opacities, scaling, rotation, objects_dc):
    normals = np.zeros_like(xyz)
    dtype_full = [(attr, 'f4') for attr in construct_list_of_attributes(
        features_dc.shape, features_rest.shape, scaling.shape, rotation.shape, objects_dc.shape)]
    
    elements = np.zeros(xyz.shape[0], dtype=dtype_full)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['opacity'] = opacities.flatten()

    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        elements[f'f_dc_{i}'] = features_dc.reshape(-1)[i::features_dc.shape[1]*features_dc.shape[2]]

    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        elements[f'f_rest_{i}'] = features_rest.reshape(-1)[i::features_rest.shape[1]*features_rest.shape[2]]

    for i in range(scaling.shape[1]):
        elements[f'scale_{i}'] = scaling[:, i]

    for i in range(rotation.shape[1]):
        elements[f'rot_{i}'] = rotation[:, i]

    for i in range(objects_dc.shape[1] * objects_dc.shape[2]):
        elements[f'obj_dc_{i}'] = objects_dc.reshape(-1)[i::objects_dc.shape[1]*objects_dc.shape[2]]

    PlyData([PlyElement.describe(elements, 'vertex')], text=True).write(path)

def mesh_to_ply_w_labels(ply_path, trans_ply_path, face_labels):
    max_sh_degree = 0
    mesh = trimesh.load(ply_path, process=False, maintain_order=True)
    verts = mesh.vertices

    points = torch.tensor(verts)
    faces = torch.tensor(np.array(mesh.faces, dtype=np.int64)).cuda()
    faces_verts = torch.tensor(points[faces.cpu()], dtype=torch.float32)
    ellipsoid_centers = find_points_inside_triangle(faces_verts).float().cuda()
    fused_point_cloud = ellipsoid_centers.detach().cpu().numpy()

    C0 = 0.28209479177387814
    shs = ((mesh.visual.face_colors[:, :3] / 255.0) - 0.5) / C0
    fused_color = RGB2SH(SH2RGB(torch.tensor(shs))).float().cuda()

    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features = features.detach().cpu().numpy()

    side_lengths, heights = calculate_sides_and_heights(faces_verts)
    scales = torch.stack([heights * 0.5, side_lengths * 0.5, 0.0001 * torch.ones_like(side_lengths)], dim=1)
    scales = torch.log(scales).to('cuda').detach().cpu().numpy()

    normals = normal_vector_of_plane_batch(faces_verts)
    rots = batch_vector_to_quaternion(normals)[:, [3, 0, 1, 2]].cuda().detach().cpu().numpy()
    opacities = inverse_sigmoid(torch.ones((faces_verts.shape[0], 1), device="cuda")).detach().cpu().numpy()

    num_objects = 16
    objects_dc = np.zeros((fused_point_cloud.shape[0], num_objects, 1))
    objects_dc[np.arange(fused_point_cloud.shape[0]), face_labels.astype(int), 0] = 1

    save_ply(trans_ply_path, fused_point_cloud, features[:, :, 0:1], features[:, :, 1:], opacities, scales, rots, objects_dc)

def obj_mesh_to_ply_mesh(in_mesh_path, out_ply_path):
    mesh = trimesh.load(in_mesh_path)
    mesh.visual = mesh.visual.to_color()
    mesh.export(out_ply_path)

def compute_face_labels(faces, vclass):
    face_labels = np.array([np.bincount(vclass[face]).argmax() for face in faces])
    return face_labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="path to the main dir")
    args = parser.parse_args()

    DIR = os.path.abspath(f'logs/output/{args.dir}')
    obj_files = glob.glob(f'{DIR}/*_clothed_smplx.obj')
    print(DIR, obj_files)

    for obj_path in obj_files:
        ply_path = obj_path.replace('.obj', '.ply')
        obj_mesh_to_ply_mesh(obj_path, ply_path)
        
        vclass_path = obj_path.replace('.obj', '_vclass.npy')
        vclass= np.load(vclass_path)
        
        mesh = Mesh.load(obj_path)
        faces = mesh.f.detach().cpu().numpy() 

        vclass_label = (torch.tensor(vclass).argmax(dim=1)).int().detach().cpu().numpy()
        face_mask_vote = compute_face_labels(faces, vclass_label)
        assert face_mask_vote.shape == np.zeros((20908)).shape
        mesh_to_ply_w_labels(ply_path, ply_path, face_mask_vote)
