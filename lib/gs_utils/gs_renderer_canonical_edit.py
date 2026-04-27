import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from lib.mesh_utils.mesh import Mesh
from lib import seg_config
from kiui.mesh_utils import decimate_mesh, clean_mesh

import kiui

from utils.map import PerVertQuaternion
from utils.data_utils import sample_bary_on_triangles, retrieve_verts_barycentric, retrieve_local_coord_system
import torch.nn.functional as F
from utils.map import quaternion_multiply
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh

from utils.map import standardize_quaternion, quaternion_raw_multiply, quaternion_multiply

def quaternion_conjugate(quaternion):
    """ Returns the conjugate of a quaternion. """
    q_conj = quaternion.clone()
    q_conj[..., 1:] *= -1
    return q_conj
    
def find_relative_rotation(base_quat, abs_rotation):
    """ Finds the relative rotation quaternion given the base and absolute rotations. """
    # Make sure both quaternions are standardized before calculations
    base_quat = standardize_quaternion(base_quat)
    abs_rotation = standardize_quaternion(abs_rotation)

    # Calculate the conjugate of the base quaternion
    base_conj = quaternion_conjugate(base_quat)

    # Multiply abs_rotation by the conjugate of base_quat
    relative_rotation = quaternion_raw_multiply(base_conj, abs_rotation)
    
    # Standardize the resulting quaternion
    return standardize_quaternion(relative_rotation)

def find_nearest_kdtree(A, B):
    
    A = np.asarray(A)
    max_index = A.shape[0]
    B = np.asarray(B)
    
    # Create a KDTree object for A
    tree = cKDTree(A)
    
    # Query the tree for the nearest neighbor to each point in B
    distances, indices = tree.query(B, k=1)  # k=1 finds the nearest neighbor

    mask = []
    new_indices = []
    
    # Calculate normals
    normals = []
    xyz_distance = []
    for i, index in enumerate(indices):
        if index >= max_index:
            mask.append(False)
            continue

        nearest_point = A[index]
        vector = B[i] - nearest_point
        distance = distances[i]  # Distance already computed by KDTree
        normal = vector / distance if distance != 0 else np.zeros_like(vector)
        normals.append(normal)

        xyz_distance.append(vector)
        mask.append(True)
        new_indices.append(index)
    
    return new_indices, normals, xyz_distance, mask


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

from utils.human_models import smpl_x
import cv2
import copy
# from gs_renderer_canonical import normal_vector_of_plane_batch
SET_NEG = True
Z_OFFSET = torch.tensor([-0.04])
smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

def load_from_smplx_obj(mesh):
    mesh = trimesh.load(mesh)
    verts = torch.tensor(mesh.vertices).cuda()
    faces = torch.tensor(mesh.faces).cuda()
    face_normals = torch.tensor(mesh.face_normals).cuda()
    return verts, faces, face_normals

def reset_arm_rotations(body_pose):
    """
    Resets the rotations of the arms in the SMPL-X body_pose tensor to zero.
    Args:
        body_pose (torch.Tensor): A tensor of shape (1, 63).
    Returns:
        torch.Tensor: Modified body_pose tensor with arm rotations set to zero.
    """
    # Indices for arm joints in the body_pose
    left_arm_indices = [15, 16, 17]  # Left shoulder
    right_arm_indices = [18, 19, 20]  # Right shoulder
    left_forearm_indices = [33, 34, 35]  # Left elbow
    right_forearm_indices = [36, 37, 38]  # Right elbow

    # Ensure the input tensor is modified in place (or create a copy if needed)
    modified_body_pose = body_pose.clone()  # Clone to avoid in-place modification

    # Flatten the tensor to access indices easily
    modified_body_pose[:, left_arm_indices + right_arm_indices + left_forearm_indices + right_forearm_indices] = 0.0

    return modified_body_pose


def load_from_smplx_path(smplx_fp, pose_type=None):
    Z_OFFSET = torch.tensor([-0.04])

    smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

    batch = {}
    data = np.load(smplx_fp)
    for k in data.files:
        batch[k] = data[k]

    root_pose = torch.tensor(batch['global_orient']).cuda()
    body_pose = torch.tensor(batch['body_pose']).cuda().reshape(-1).unsqueeze(0)
    lhand_pose = torch.tensor(batch['left_hand_pose']).cuda().reshape(-1).unsqueeze(0)
    rhand_pose = torch.tensor(batch['right_hand_pose']).cuda().reshape(-1).unsqueeze(0)
    jaw_pose = torch.tensor(batch['jaw_pose']).cuda()
    shape = torch.tensor(batch['betas']).cuda() # .unsqueeze(0)
    expr = torch.tensor(batch['expression']).cuda() # .unsqueeze(0)

    if pose_type is not None:

        if pose_type == "custom-pose":
            # body_pose = torch.zeros_like(body_pose).view(body_pose.shape[0], -1, 3)
            # body_pose = body_pose.view(body_pose.shape[0], -1, 3)
            # body_pose[:, 0] = torch.tensor([0., 0., 30 * np.pi / 180.])
            # body_pose[:, 1] = torch.tensor([0., 0., -30 * np.pi / 180.])
            # body_pose = body_pose.view(body_pose.shape[0], -1)
            body_pose = body_pose.view(body_pose.shape[0], -1, 3)
            body_pose[:, 15] = torch.tensor([0., 0., 0.])
            body_pose[:, 16] = torch.tensor([0., 0., 0.])
            body_pose = body_pose.view(body_pose.shape[0], -1)

        else:
            root_pose = torch.zeros_like(root_pose) 
            lhand_pose = torch.zeros_like(lhand_pose)
            rhand_pose = torch.zeros_like(rhand_pose)
            jaw_pose = torch.zeros_like(jaw_pose)
            expr = torch.zeros_like(expr)

            if pose_type == "t-pose":
                body_pose = torch.zeros_like(body_pose) 
            elif pose_type == "a-pose":
                body_pose = torch.zeros_like(body_pose).view(body_pose.shape[0], -1, 3)
                body_pose[:, 15] = torch.tensor([0., 0., -45 * np.pi / 180.])
                body_pose[:, 16] = torch.tensor([0., 0., 45 * np.pi / 180.])
                body_pose = body_pose.view(body_pose.shape[0], -1)

            elif pose_type == "da-pose":
                body_pose = torch.zeros_like(body_pose).view(body_pose.shape[0], -1, 3)
                body_pose[:, 0] = torch.tensor([0., 0., 30 * np.pi / 180.])
                body_pose[:, 1] = torch.tensor([0., 0., -30 * np.pi / 180.])
                body_pose = body_pose.view(body_pose.shape[0], -1)
        
    batch_size = root_pose.shape[0]

    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
    output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                reye_pose=zero_pose, expression=expr)
    verts = output.vertices[0] # .float() # .detach().cpu().numpy()

    if SET_NEG:
        verts[:, 1]*= -1
        verts[:, 2]*= -1

        translation = torch.tensor(batch['translation']).cuda()
        scale = torch.tensor(batch['scale']).cuda()

        if translation.shape[0] == 3:
            new_center = torch.cat([translation])# resize[0], resize[1]
        else:
            new_center = torch.cat([translation, Z_OFFSET.to(verts.device)])# resize[0], resize[1]
        verts = (verts - new_center) * scale # new_scale

        if pose_type is not None and pose_type!= 'custom-pose':
            verts[:, 1]*= -1
            verts[:, 2]*= -1
        
    faces = torch.tensor(smpl_x.face.astype('int'))
    faces_verts = verts[faces]
    face_normals = normal_vector_of_plane_batch(faces_verts)

    return verts, faces, face_normals

def load_from_smplx_path_custom(smplx_fp, pose=None, betas=None):
    Z_OFFSET = torch.tensor([-0.04])
    SET_NEG = True

    smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

    batch = {}
    data = np.load(smplx_fp)
    for k in data.files:
        batch[k] = data[k]

    # root_pose = torch.tensor(batch['global_orient']).cuda()
    if pose is not None:
        root_pose = torch.tensor(pose[:3]).cuda().reshape(-1).unsqueeze(0).float()
        body_pose = torch.tensor(pose[3 :66]).cuda().reshape(-1).unsqueeze(0).float()
    else:
        root_pose = torch.tensor(batch['global_orient']).cuda() # .unsqueeze(0)
        body_pose = torch.tensor(batch['body_pose']).cuda().reshape(-1).unsqueeze(0)
    # print(root_pose.shape, body_pose.shape)
    lhand_pose = torch.tensor(batch['left_hand_pose']).cuda().reshape(-1).unsqueeze(0)
    rhand_pose = torch.tensor(batch['right_hand_pose']).cuda().reshape(-1).unsqueeze(0)
    jaw_pose = torch.tensor(batch['jaw_pose']).cuda()
    if betas is not None:
        shape = torch.tensor(betas).cuda().reshape(-1).unsqueeze(0)
    else:
        shape = torch.tensor(batch['betas']).cuda() # .unsqueeze(0)
    expr = torch.tensor(batch['expression']).cuda() # .unsqueeze(0)

    # print(body_pose.shape)

    batch_size = root_pose.shape[0]

    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
    output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                reye_pose=zero_pose, expression=expr)
    verts = output.vertices[0] # .float() # .detach().cpu().numpy()

    if SET_NEG:
        verts[:, 1]*= -1
        verts[:, 2]*= -1

        translation = torch.tensor(batch['translation']).cuda()
        scale = torch.tensor(batch['scale']).cuda()

        new_center = torch.cat([translation])# resize[0], resize[1]
        verts = (verts - new_center) * scale # new_scale

        # if pose_type is not None:
        #     verts[:, 1]*= -1
        #     verts[:, 2]*= -1
        
    faces = torch.tensor(smpl_x.face.astype('int'))
    faces_verts = verts[faces]
    face_normals = normal_vector_of_plane_batch(faces_verts)

    return verts, faces, face_normals


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights

def distance_batch(points1, points2):
    return torch.norm(points1 - points2, dim=2)

def calculate_sides_and_heights(triangles):
    # Calculate the lengths of the sides for a batch of triangles
    side_lengths = distance_batch(triangles, torch.roll(triangles, -1, dims=1))

    # Calculate the areas of the triangles using Heron's formula
    s = torch.sum(side_lengths, dim=1) / 2
    area = torch.sqrt(s * (s - side_lengths[:, 0]) * (s - side_lengths[:, 1]) * (s - side_lengths[:, 2]))

    # Calculate the heights from the midpoints of the first side to the opposite vertex
    # Ensuring correct broadcasting of the division
    heights = 2 * area / side_lengths[:, 0]

    return side_lengths[:, 0], heights

def normal_vector_of_plane_batch(triangles):
    v1 = triangles[:, 1, :] - triangles[:, 0, :]
    v2 = triangles[:, 2, :] - triangles[:, 0, :]
    return torch.cross(v1, v2, dim=1)

def batch_vector_to_quaternion(normals):
    device = normals.device

    # Ensure normals are normalized
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    
    # Initial vector (z-axis), replicated for each normal in the batch
    initial_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    initial_vec_batch = initial_vec.expand(normals.size(0), -1)
    
    # Rotation axis for each pair of vectors (cross product)
    rotation_axis = torch.cross(initial_vec_batch, normals, dim=1)
    rotation_axis_norm = torch.linalg.norm(rotation_axis, dim=1, keepdim=True)
    
    # Handle the case when initial_vec and normals are parallel
    close_to_zero = rotation_axis_norm.squeeze() < 1e-6
    opposite_direction = torch.allclose(initial_vec_batch, -normals, atol=1e-6)
    
    # Angle between initial_vec and normals
    angle = torch.acos(torch.clamp(torch.sum(initial_vec_batch * normals, dim=1), -1.0, 1.0))
    
    # Generate quaternions
    sin_half_angle = torch.sin(angle / 2).unsqueeze(-1)
    cos_half_angle = torch.cos(angle / 2).unsqueeze(-1)
    
    quaternion = torch.zeros(normals.size(0), 4, dtype=torch.float32, device=device)
    quaternion[:, :3] = rotation_axis / rotation_axis_norm * sin_half_angle
    quaternion[:, 3] = cos_half_angle.squeeze()
    
    # Fix quaternions for parallel and opposite vectors
    quaternion[close_to_zero, :] = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device)
    quaternion[opposite_direction, :] = torch.tensor([initial_vec[1], -initial_vec[0], 0, 0], dtype=torch.float32, device=device)
    
    return quaternion

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._objects_dc = torch.empty(0)
        self.num_objects = seg_config.get().num_classes
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device = 'cuda'

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    ##### smplx only
    @property
    def get_smplx_xyz(self):
        # return self._xyz_smplx
        return self.base_smplx_xyz # empty

    # @property
    # def get_smplx_keypoints(self):
    #     # return self._xyz_smplx
    #     return self.base_smplx_xyz # empty
    
    @property
    def base_smplx_xyz(self):
        return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs_smplx, self.sample_bary_smplx)
    @property
    def get_smplx_features(self):
        features_dc = self._features_dc_smplx
        features_rest = self._features_rest_smplx
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_smplx_opacity(self):

        # return self.opacity_activation(opacity_smplx)
        return self.opacity_activation(self._opacity_smplx)
    
    @property
    def get_smplx_objects(self):
        return self._objects_dc_smplx
        # zeros = torch.zeros(self._objects_dc.shape[0], 1, 2).to(self._objects_dc.device)
        # return torch.cat((zeros, self._objects_dc), dim=-1)
    
    @property
    def get_smplx_scaling(self):
        # return self.scaling_activation(self._scaling_smplx)
        scaling_alter = self._face_scaling[self.sample_fidxs_smplx].detach()
        new_scales = self._scaling_smplx * scaling_alter
        new_scales[:, -1] *= 100000000
        new_scales[:, 1] *= 1.05
        new_scales[:, 0] *= 1.05
        return self.scaling_activation(new_scales)
    
    @property
    def get_smplx_scaling_out(self):
        # return self.scaling_activation(self._scaling_smplx)
        scaling_alter = self._face_scaling[self.sample_fidxs_smplx].detach()
        new_scales = self._scaling_smplx * scaling_alter
        # new_scales[:, -1] *= 100000000
        # new_scales[:, 1] *= 1.05
        # new_scales[:, 0] *= 1.05
        # print(new_scales.mean(0))
        return self.scaling_activation(new_scales)
    
    @property
    def get_smplx_rotation(self): # empty
        # return self.rotation_activation(self._rotation_smplx)
        # return self._rotation_smplx

        normals = torch.nn.functional.normalize(self.mesh_norms)
        rots = batch_vector_to_quaternion(normals)[:, [3, 0, 1, 2]].cuda()
        return self.rotation_activation(rots)
    
    def get_smplx_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_smplx_scaling, scaling_modifier, self._rotation_smplx)

    #################### learnable #############
    
    @property
    def get_learnable_xyz(self):
        # return self._xyz
    
        # xyz = self.base_normal * self._xyz[..., -1:]
        # return self.base_xyz + xyz

        # NOTE: account for normals 
        x_axis, y_axis, normal = self.base_origin
        xyz = self._xyz[:, 0:1] * x_axis + self._xyz[:, 1:2] * y_axis + self._xyz[:, 2:3] * normal # 
        return self.base_xyz - xyz

        # return self.base_xyz + self._xyz
    
    @property
    def base_xyz(self):
        return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)
    
    @property
    def base_normal(self):
        return torch.nn.functional.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
    
    @property
    def base_origin(self): # local coordinate system 
        return retrieve_local_coord_system(self.mesh_verts, self.cano_faces, 
                                                        self.sample_fidxs)
    
    @property
    def get_learnable_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_learnable_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_learnable_objects(self):
        return self._objects_dc
    
    @property
    def get_learnable_scaling(self):
        scaling_alter = self._face_scaling[self.sample_fidxs].detach()
        return self.scaling_activation(self._scaling * scaling_alter)
    
    @property
    def get_learnable_rotation(self):
        # return self.rotation_activation(self._rotation)
        # return self._rotation
        
        return self.rotation_activation(quaternion_multiply(self.base_quat, self._rotation))
    
    @property
    def base_quat(self):
        return torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)
    
    def get_learnable_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_learnable_scaling, scaling_modifier, self._rotation)


    ### FIXED ###
    @property
    def get_fixed_features(self):
        features_dc = self._features_dc_fixed
        features_rest = self._features_rest_fixed
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_fixed_opacity(self):
        return self.opacity_activation(self._opacity_fixed)
    
    @property
    def get_fixed_objects(self):
        return self._objects_dc_fixed
    
    @property
    def get_fixed_scaling(self):
        scaling_alter = self._face_scaling[self.sample_fidxs_fixed].detach()
        return self.scaling_activation(self._scaling_fixed * scaling_alter)
    
    @property
    def get_fixed_rotation(self):
        return self.rotation_activation(quaternion_multiply(self.base_quat_fixed, self._rotation_fixed))
    
    @property
    def base_quat_fixed(self):
        return torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs_fixed], self.sample_bary_fixed)
    
    def get_fixed_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_fixed_scaling, scaling_modifier, self._rotation_fixed)
    

    ############ NEW ############

    @property
    def get_scaling(self):
        # return self.scaling_activation(self._scaling)
        scaling = self.get_learnable_scaling
        scaling_smplx = self.get_smplx_scaling
        return torch.cat((scaling_smplx, scaling))

    @property
    def get_rotation(self):
        # return self.rotation_activation(self._rotation)
        # normals = torch.nn.functional.normalize(self.mesh_norms)
        # rots = batch_vector_to_quaternion(normals)[:, [3, 0, 1, 2]].cuda()
        # return self.rotation_activation(rots)

        # return self.rotation_activation()
        rotation_smplx = self.get_smplx_rotation
        rotation = self.get_learnable_rotation
        return torch.cat((rotation_smplx, rotation))

    @property
    def get_xyz(self):
        # return self._xyz
        # return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, self.sample_fidxs, self.sample_bary)
        xyz_smplx = self.get_smplx_xyz
        xyz = self.get_learnable_xyz
        return torch.cat((xyz_smplx, xyz))

    @property
    def get_features(self):
        features_dc = torch.cat((self._features_dc_smplx, self._features_dc))
        features_rest = torch.cat((self._features_rest_smplx, self._features_rest))
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return torch.cat((self._objects_dc_smplx, self.get_learnable_objects))
                         
    @property
    def get_opacity(self):

        clothing_fidxs = self.sample_fidxs
        smplx_fidxs = self.sample_fidxs_smplx
        occ_fidxs = torch.unique(clothing_fidxs) # smplx which clothing gaussians are tagged to
        occ_list = occ_fidxs.tolist()
        fill_list = torch.ones_like(smplx_fidxs).float()
        fill_list[occ_list] = inverse_sigmoid(torch.zeros_like(fill_list[occ_list]))
        opacity_smplx = self._opacity_smplx * fill_list.unsqueeze(-1)

        return self.opacity_activation(torch.cat((opacity_smplx, self._opacity)))
        # return self.opacity_activation(torch.cat((self._opacity_smplx, self._opacity)))

    #################### refine #############
    
    @property
    def get_refine_xyz(self):
        return self._xyz_refine
    
    @property
    def get_refine_features(self):
        features_dc = self._features_dc_refine
        features_rest = self._features_rest_refine
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_refine_opacity(self):
        return self.opacity_activation(self._opacity_refine)
    
    @property
    def get_refine_objects(self):
        return self._objects_dc_refine
    
    @property
    def get_refine_scaling(self):
        return self.scaling_activation(self._scaling_refine)
    
    @property
    def get_refine_rotation(self):
        return self._rotation_refine

    def get_refine_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_refine_scaling, scaling_modifier, self._rotation_refine)

    #################### end #############

    @torch.no_grad()
    def extract_learnable_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_learnable_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_learnable_xyz[mask]
        stds = self.get_learnable_scaling[mask]
        print(stds.mean(0))
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        # covs = self.covariance_activation(stds, 1, self._rotation[mask])
        
        covs = self.covariance_activation(stds, 1, self.get_learnable_rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)

        return occ
    

    @torch.no_grad()
    def extract_smplx_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_smplx_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_smplx_xyz[mask]
        stds = self.get_smplx_scaling_out[mask]
        print(stds.mean(0))
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        # covs = self.covariance_activation(stds, 1, self._rotation[mask])
        
        # covs = self.covariance_activation(stds, 1, self._rotation_smplx[mask])
        covs = self.covariance_activation(stds, 1, self.get_smplx_rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)

        return occ
    
    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self.get_rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)

        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # occ = self.extract_fields(resolution).detach().cpu().numpy()
        # occ = self.extract_smplx_fields(resolution).detach().cpu().numpy()
        occ = self.extract_learnable_fields(resolution).detach().cpu().numpy()

        import mcubes
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        # vertices, triangles = clean_mesh(vertices, triangles, remesh=False)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()
        print(v.shape, f.shape)
        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    ################## NEW ##################

    def setup_canonical(self, cano_verts, cano_norms, cano_faces):
        self.cano_verts = cano_verts
        self.cano_norms = cano_norms
        self.cano_faces = cano_faces

        # quaternion from cano to pose
        self.quat_helper = PerVertQuaternion(cano_verts, cano_faces).to(self.device)

        # # phong surface for triangle walk
        # self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,
        #                                   outer_loop=2, inner_loop=50, method='uvd').to(self.device)
        
    def create_from_canonical(self, cano_mesh, num_pts=10000, sample_fidxs=None, sample_bary=None):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device)
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)
        # self.setup_canonical(cano_verts, cano_norms, cano_faces)
        # self.mesh_verts = self.cano_verts
        # self.mesh_norms = self.cano_norms
        
        # sample on mesh
        if sample_fidxs is None or sample_bary is None:
            # num_samples = 10000 # self.config.get('num_init_samples', 10000)
            sample_fidxs, sample_bary = sample_bary_on_triangles(cano_faces.shape[0], num_pts)
        self.sample_fidxs = sample_fidxs.to(self.device)
        self.sample_bary = sample_bary.to(self.device)

        sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = F.normalize(sample_norms, dim=-1)

        pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
                              normals=sample_norms.detach().cpu().numpy(),
                              colors=torch.full_like(sample_verts, 0.5).float().cpu())
        # shs = np.random.random((sample_verts.shape[0], 3)) / 255.0
        # pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
        #                       normals=sample_norms.detach().cpu().numpy(),
        #                       colors=torch.tensor(SH2RGB(shs)).float().cpu())
        self.create_from_pcd(pcd)



    def create_from_canonical_smplx_winput(self, cano_mesh, input_ply, sample_fidxs=None, sample_bary=None):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device) # verts normals
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)
        self.mesh_verts = self.cano_verts
        self.mesh_norms = self.cano_norms

        num_faces = cano_mesh['mesh_faces'].shape[0]
        sample_fidxs =  torch.arange(0, num_faces)
        sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(num_faces, 1)
        self.sample_fidxs_smplx = sample_fidxs.to(self.device)
        self.sample_bary_smplx = sample_bary.to(self.device)

        sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs_smplx, self.sample_bary_smplx)
        # sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs_smplx, self.sample_bary_smplx)
        # sample_norms = F.normalize(sample_norms, dim=-1)
        # self.sample_norms = sample_norms

        # pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
        #                       normals=sample_norms.detach().cpu().numpy(),
        #                       colors=torch.full_like(sample_verts, 0.5).float().cpu())

        ####  create_from_pcd ####
        self.spatial_lr_scale = 1
        fused_point_cloud = torch.tensor(np.asarray(sample_verts.detach().cpu().numpy())).float().cuda()

        # loaded colors
        plydata = PlyData.read(input_ply)
        num_points = sample_verts.shape[0]
        features_dc = np.zeros((num_points, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((num_points, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        opacities = inverse_sigmoid(torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")
        features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda")


        # random init obj_id now
        # fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        # fused_objects = fused_objects[:,:,None]
        objects_dc = np.zeros((features_dc.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = self.quat_helper.cano_scales


        # # scales = torch.log(scales) # .to('cuda')
        # scales[:, -1] /= 100000000
        # scales[:, 1] /= 1.1
        # scales[:, 0] /= 1.1

        # scales[:, -1] /= 100
        # scales[:, 1] /= 1.1
        # scales[:, 0] /= 1.1

        normals = torch.nn.functional.normalize(self.mesh_norms)
        rots = batch_vector_to_quaternion(normals)[:, [3, 0, 1, 2]].cuda()

        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz_smplx = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc_smplx = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_smplx = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_smplx = nn.Parameter(scales.requires_grad_(True))
        self._rotation_smplx = nn.Parameter(rots.requires_grad_(True))
        self._opacity_smplx = nn.Parameter(opacities.requires_grad_(True))
        # self._objects_dc_smplx = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))
        self._objects_dc_smplx = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._face_scaling = torch.ones_like(self._scaling_smplx).cuda()

        self.n_smplx_points = num_points # self._xyz_smplx.shape[0]

    def create_from_canonical_smplx(self, cano_mesh, sample_fidxs=None, sample_bary=None):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device) # verts normals
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)
        self.mesh_verts = self.cano_verts
        self.mesh_norms = self.cano_norms
        
        # sample on mesh
        # if sample_fidxs is None or sample_bary is None:
        #     num_samples = 10000 # self.config.get('num_init_samples', 10000)
        #     sample_fidxs, sample_bary = sample_bary_on_triangles(cano_faces.shape[0], num_samples)

        num_faces = cano_mesh['mesh_faces'].shape[0]
        sample_fidxs =  torch.arange(0, num_faces)
        sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(num_faces, 1)
        self.sample_fidxs_smplx = sample_fidxs.to(self.device)
        self.sample_bary_smplx = sample_bary.to(self.device)

        sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs_smplx, self.sample_bary_smplx)
        sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs_smplx, self.sample_bary_smplx)
        sample_norms = F.normalize(sample_norms, dim=-1)
        self.sample_norms = sample_norms

        pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
                              normals=sample_norms.detach().cpu().numpy(),
                              colors=torch.full_like(sample_verts, 0.5).float().cpu())
        # shs = np.random.random((sample_verts.shape[0], 3)) / 255.0
        # pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
        #                       normals=sample_norms.detach().cpu().numpy(),
        #                       colors=torch.tensor(SH2RGB(shs)).float().cpu())
        # self.create_from_pcd(pcd)

        ####  create_from_pcd ####
        self.spatial_lr_scale = 1
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = self.quat_helper.cano_scales

        normals = torch.nn.functional.normalize(self.mesh_norms)
        rots = batch_vector_to_quaternion(normals)[:, [3, 0, 1, 2]].cuda()

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz_smplx = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc_smplx = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_smplx = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_smplx = nn.Parameter(scales.requires_grad_(True))
        self._rotation_smplx = nn.Parameter(rots.requires_grad_(True))
        self._opacity_smplx = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc_smplx = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))
        self._face_scaling = torch.ones_like(self._scaling_smplx).cuda()

        self.n_smplx_points = self._xyz_smplx.shape[0]

    def update_to_posed_mesh(self, posed_mesh=None):
        if posed_mesh is not None:
            self.mesh_verts = posed_mesh['mesh_verts'].float().to(self.device)
            self.mesh_norms = posed_mesh['mesh_norms'].float().to(self.device)

            self.per_vert_quat = self.quat_helper(self.mesh_verts)
            self.tri_quats = self.per_vert_quat[self.cano_faces]

        # self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)
        self._face_scaling = self.quat_helper.calc_face_length_change(self.mesh_verts)

    def update_to_cano_mesh(self):
        cano = {
            'mesh_verts': self.cano_verts,
            'mesh_norms': self.cano_norms,
        }
        self.update_to_posed_mesh(cano)

    ################## NEW ##################

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.seg_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l

    # def save_ply(self, path):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()
    #     obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def save_ply_absolute(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_features.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        features_rest = torch.cat((self._features_rest, self._features_rest_smplx))
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()
        obj_dc = self.get_objects.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_absolute_learnable(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.get_learnable_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_learnable_features.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        features_rest = self._features_rest
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_learnable_opacity.detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_learnable_scaling).detach().cpu().numpy()
        rotation = self.get_learnable_rotation.detach().cpu().numpy()
        obj_dc = self.get_learnable_objects.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes_fidxs(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        l.append('sample_fidxs') # L.append('sample_fidxs')
        return l

    # def save_ply_fidxs(self, path):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()
    #     obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     sample_fidxs = self.sample_fidxs.unsqueeze(-1).detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_fidxs()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc, sample_fidxs), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def save_ply_fidxs_learnable(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities= self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sample_fidxs = self.sample_fidxs.unsqueeze(-1).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_fidxs()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc, sample_fidxs), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def save_ply_fidxs(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz_fixed = self._xyz_fixed.detach().cpu().numpy()
        normals_fixed = np.zeros_like(xyz_fixed)
        f_dc_fixed = self._features_dc_fixed.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_fixed = self._features_rest_fixed.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities_fixed = self._opacity_fixed.detach().cpu().numpy()
        scale_fixed = self._scaling_fixed.detach().cpu().numpy()
        rotation_fixed = self._rotation_fixed.detach().cpu().numpy()
        obj_dc_fixed = self._objects_dc_fixed.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sample_fidxs_fixed = self.sample_fidxs_fixed.unsqueeze(-1).detach().cpu().numpy()

        xyz_learned = self._xyz.detach().cpu().numpy()
        normals_learned = np.zeros_like(xyz_learned)
        f_dc_learned = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_learned = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities_learned = self._opacity.detach().cpu().numpy()
        scale_learned = self._scaling.detach().cpu().numpy()
        rotation_learned = self._rotation.detach().cpu().numpy()
        obj_dc_learned = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sample_fidxs_learned = self.sample_fidxs.unsqueeze(-1).detach().cpu().numpy()

        print(xyz_fixed.shape, xyz_learned.shape)
        xyz = np.vstack((xyz_fixed, xyz_learned))
        normals = np.vstack((normals_fixed, normals_learned))
        f_dc = np.vstack((f_dc_fixed, f_dc_learned))
        f_rest = np.vstack((f_rest_fixed, f_rest_learned))
        opacities = np.vstack((opacities_fixed, opacities_learned))
        scale = np.vstack((scale_fixed, scale_learned))
        rotation = np.vstack((rotation_fixed, rotation_learned))
        obj_dc = np.vstack((obj_dc_fixed, obj_dc_learned))
        sample_fidxs = np.vstack((sample_fidxs_fixed, sample_fidxs_learned))

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_fidxs()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc, sample_fidxs), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def load_ply_fidxs_category(self, path, category, recolor):

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        obj_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc")]
        if len(obj_dc_names) > 0:
            obj_dc_names = sorted(obj_dc_names, key = lambda x: int(x.split('_')[-1]))
            objects_dc = np.zeros((xyz.shape[0], len(obj_dc_names)))
            for idx, attr_name in enumerate(obj_dc_names):
                objects_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            objects_dc = RGB2SH(torch.rand((xyz.shape[0],self.num_objects), device="cuda"))
        objects_dc = objects_dc[:,:,None]


        if isinstance(category, int):
            mask3d = torch.argmax(torch.tensor(objects_dc).squeeze(-1),dim=-1) == category
        else:
            category = torch.tensor(category) # .cuda()
            # nonskin_idxs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()
            mask3d = torch.isin(torch.argmax(torch.tensor(objects_dc).squeeze(-1),dim=-1), category)

        if recolor:
            num_train_pts = xyz[mask3d].shape[0]
            shs = np.random.random((num_train_pts, 3)) / 255.0
            colors = SH2RGB(shs)
            fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(torch.tensor(features_dc[mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra[mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))


        self._xyz = nn.Parameter(torch.tensor(xyz[mask3d], dtype=torch.float, device="cuda").requires_grad_(True))



        # self._features_dc = nn.Parameter(torch.tensor(features_dc[mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra[mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc[mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._xyz_fixed = nn.Parameter(torch.tensor(xyz[~mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_fixed = nn.Parameter(torch.tensor(features_dc[~mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_fixed = nn.Parameter(torch.tensor(features_extra[~mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_fixed = nn.Parameter(torch.tensor(opacities[~mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_fixed = nn.Parameter(torch.tensor(scales[~mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_fixed = nn.Parameter(torch.tensor(rots[~mask3d], dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc_fixed = nn.Parameter(torch.tensor(objects_dc[~mask3d], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        sample_fidxs_names = [p.name for p in plydata.elements[0].properties if p.name == 'sample_fidxs']
        if len(sample_fidxs_names) > 0:
            sample_fidxs = torch.tensor(np.asarray(plydata.elements[0][sample_fidxs_names[0]]), dtype=torch.long).cuda()
            sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(xyz.shape[0], 1).cuda()

            self.sample_fidxs = sample_fidxs[mask3d]
            self.sample_fidxs_fixed = sample_fidxs[~mask3d]

            self.sample_bary = sample_bary[mask3d]
            self.sample_bary_fixed = sample_bary[~mask3d]



        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree


    def load_ply_fidxs(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        obj_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc")]
        if len(obj_dc_names) > 0:
            obj_dc_names = sorted(obj_dc_names, key = lambda x: int(x.split('_')[-1]))
            objects_dc = np.zeros((xyz.shape[0], len(obj_dc_names)))
            for idx, attr_name in enumerate(obj_dc_names):
                objects_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            objects_dc = RGB2SH(torch.rand((xyz.shape[0],self.num_objects), device="cuda"))
        objects_dc = objects_dc[:,:,None]
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        sample_fidxs_names = [p.name for p in plydata.elements[0].properties if p.name == 'sample_fidxs']
        if len(sample_fidxs_names) > 0:
            self.sample_fidxs = torch.tensor(np.asarray(plydata.elements[0][sample_fidxs_names[0]]), dtype=torch.long).cuda()
            self.sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(xyz.shape[0], 1).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        obj_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc")]
        if len(obj_dc_names) > 0:
            obj_dc_names = sorted(obj_dc_names, key = lambda x: int(x.split('_')[-1]))
            objects_dc = np.zeros((xyz.shape[0], len(obj_dc_names)))
            for idx, attr_name in enumerate(obj_dc_names):
                objects_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            objects_dc = RGB2SH(torch.rand((xyz.shape[0],self.num_objects), device="cuda"))
            # objects_dc = objects_dc[:,:,None]
            # objects_dc = RGB2SH(torch.rand((xyz.shape[0],self.num_objects), device="cuda"))
        objects_dc = objects_dc[:,:,None]
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    #     embed_fn = path.replace('point_cloud.ply', 'v')
    #     gs_model.load_from_embedding(embed_fn)

    # def load_from_embedding(self, embed_fn):
    #     import json
    #     from pathlib import Path

    #     with open(embed_fn, 'r') as fp:
    #         cc = json.load(fp)

    #     mesh_fn = cc['cano_mesh']
    #     if not os.path.isabs(mesh_fn):
    #         mesh_fn = str(Path(embed_fn).parent / mesh_fn)
        
    #     cano_mesh = libcore.MeshCpu(mesh_fn)
    #     cano_verts = torch.tensor(cano_mesh.V).float().to(self.device)
    #     cano_norms = torch.tensor(cano_mesh.N).float().to(self.device)
    #     cano_faces = torch.tensor(cano_mesh.F).long().to(self.device)
    #     self.setup_canonical(cano_verts, cano_norms, cano_faces)

    #     self._xyz = torch.tensor(cc['_xyz']).float().to(self.device)
    #     if '_rotation' in cc:
    #         self._rotation = torch.tensor(cc['_rotation']).float().to(self.device)
    #     self.sample_fidxs = torch.tensor(cc['sample_fidxs']).long().to(self.device)
    #     self.sample_bary = torch.tensor(cc['sample_bary']).float().to(self.device)


    def load_ply_smplx(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # xyz[:, -1] +=0.02
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # scales = np.log(scales) # .to('cuda')
        # scales[:, -1] *= 100000000
        # scales[:, 1] *= 1.1
        # scales[:, 0] *= 1.1

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        self._xyz_smplx = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_smplx = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_smplx = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_smplx = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_smplx = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_smplx = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc_smplx = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.n_smplx_points = self._xyz_smplx.shape[0]

    def load_refine_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        self._xyz_refine = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_refine = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_refine = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_refine = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_refine = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_refine = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc_refine = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def reinitialize_positions(self):

        sel_xyz = self.get_learnable_xyz.detach().cpu().numpy()
        abs_rots = self.get_learnable_rotation.detach().cuda() # .cpu().numpy()
        
        sample_verts_np = self._xyz_smplx.detach().cpu().numpy()
        indices, normals, offset, new_mask = find_nearest_kdtree(sample_verts_np, sel_xyz) # avoid max index

        num_pts = len(sel_xyz)
        self.sample_fidxs = torch.tensor(indices).cuda()
        self.sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(num_pts, 1).cuda()
        sel_rots = batch_vector_to_quaternion(torch.tensor(normals))[:, [3, 0, 1, 2]]

        x_axis, y_axis, normal = self.base_origin
        V = self.base_xyz - torch.tensor(sel_xyz).cuda()

        # Transform V to the local coordinate system for each triangle
        V_x = torch.sum(V * x_axis, dim=1, keepdim=True)
        V_y = torch.sum(V * y_axis, dim=1, keepdim=True)
        V_z = torch.sum(V * normal, dim=1, keepdim=True)

        # Vector in local coordinates
        vector_local = torch.cat((V_x, V_y, V_z), dim=1)

        self._xyz = nn.Parameter(torch.tensor(vector_local, dtype=torch.float, device="cuda").requires_grad_(True))

        # abs_rots = torch.tensor(rots).cuda()
        rel_rots = find_relative_rotation(self.base_quat, abs_rots)
        self._rotation = nn.Parameter(torch.tensor(rel_rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def load_ply_fidxs_categorical(self, path, categories):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        obj_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc")]
        if len(obj_dc_names) > 0:
            obj_dc_names = sorted(obj_dc_names, key = lambda x: int(x.split('_')[-1]))
            objects_dc = np.zeros((xyz.shape[0], len(obj_dc_names)))
            for idx, attr_name in enumerate(obj_dc_names):
                objects_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            objects_dc = RGB2SH(torch.rand((xyz.shape[0],self.num_objects), device="cuda"))
        objects_dc = objects_dc[:,:,None]

        objects_dc_idx = np.argmax(np.squeeze(objects_dc, axis=-1), axis=-1)
        mask_idxs = np.isin(objects_dc_idx, np.array(categories))

        self._xyz = nn.Parameter(torch.tensor(xyz[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        sample_fidxs_names = [p.name for p in plydata.elements[0].properties if p.name == 'sample_fidxs']
        if len(sample_fidxs_names) > 0:
            self.sample_fidxs = torch.tensor(np.asarray(plydata.elements[0][sample_fidxs_names[0]]), dtype=torch.long).cuda()[mask_idxs]
            self.sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(xyz.shape[0], 1).cuda()[mask_idxs]

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_ply_categorical(self, path, categories):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        # opacities = np.ones_like(opacities) * 0.5

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        objects_dc_idx = np.argmax(np.squeeze(objects_dc, axis=-1), axis=-1)
        init_mask_idxs = np.isin(objects_dc_idx, np.array(categories))



        # self.sample_fidxs = # find nearest points to triangles
        # self.sample_bary = # find nearest points 

        # xyz is not absolute, relative to the nearest triangle
        # norm is not absolution, relative to nearest triangle

        sel_xyz = xyz[init_mask_idxs]
        
        sample_verts_np = self._xyz_smplx.detach().cpu().numpy()
        indices, normals, offset, new_mask = find_nearest_kdtree(sample_verts_np, sel_xyz) # avoid max index

        mask_idxs = init_mask_idxs.copy()  # Start with the initial mask
        mask_idxs[init_mask_idxs] = np.array(new_mask)
        # mask_idxs = init_mask_idxs * np.array(new_mask)
        sel_xyz = xyz[mask_idxs]
        num_pts = len(sel_xyz)

        # self._xyz = nn.Parameter(torch.tensor(xyz[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(rots[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc[mask_idxs], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))


        self.sample_fidxs = torch.tensor(indices).cuda()
        self.sample_bary = torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float32).repeat(num_pts, 1).cuda()
        sel_rots = batch_vector_to_quaternion(torch.tensor(normals))[:, [3, 0, 1, 2]]

        # NOTE: v1 did not account for normals
        # self._xyz = nn.Parameter(torch.tensor(offset, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(sel_rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # NOTE: v2 account for normals
        x_axis, y_axis, normal = self.base_origin
        # Calculate vectors from points P to centroids in global coordinates
        V = self.base_xyz - torch.tensor(sel_xyz).cuda()

        # Transform V to the local coordinate system for each triangle
        V_x = torch.sum(V * x_axis, dim=1, keepdim=True)
        V_y = torch.sum(V * y_axis, dim=1, keepdim=True)
        V_z = torch.sum(V * normal, dim=1, keepdim=True)

        # Vector in local coordinates
        vector_local = torch.cat((V_x, V_y, V_z), dim=1)

        self._xyz = nn.Parameter(torch.tensor(vector_local, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree


        # abs_rots = torch.tensor(rots).cuda()
        # rel_rots = find_relative_rotation(self.base_quat, abs_rots)
        # self._rotation = nn.Parameter(torch.tensor(rel_rots[mask_idxs], dtype=torch.float, device="cuda").requires_grad_(True))

        abs_rots = torch.tensor(rots[mask_idxs]).cuda()
        rel_rots = find_relative_rotation(self.base_quat, abs_rots)
        self._rotation = nn.Parameter(torch.tensor(rel_rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask[self.n_smplx_points:]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        self.sample_fidxs = self.sample_fidxs[valid_points_mask]
        self.sample_bary = self.sample_bary[valid_points_mask]
        
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum[:self.n_smplx_points], self.xyz_gradient_accum[self.n_smplx_points:][valid_points_mask]))

        self.denom = torch.cat((self.denom[:self.n_smplx_points], self.denom[self.n_smplx_points:][valid_points_mask]))
        self.max_radii2D = torch.cat((self.max_radii2D[:self.n_smplx_points], self.max_radii2D[self.n_smplx_points:][valid_points_mask]))

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'smplx' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_sample_fidxs, new_sample_bary):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "obj_dc": new_objects_dc,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        # self.sample_fidxs = new_sample_fidxs
        # self.sample_bary = new_sample_bary
        self.sample_fidxs = torch.cat([self.sample_fidxs, new_sample_fidxs], dim=0)
        self.sample_bary = torch.cat([self.sample_bary, new_sample_bary], dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=4):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask_raw = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = selected_pts_mask_raw[self.n_smplx_points:]
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_learnable_scaling, dim=1).values > self.percent_dense*scene_extent)
        # # NOTE: prune points 
        dist = self.calculate_sdf_loss().squeeze(0).squeeze(-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, dist<0)

        # NOTE: select N so that the nunber of points is at least maintained or growing

        stds = self.get_learnable_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # get learnable rotation?
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_learnable_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_learnable_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)

        new_sample_fidxs = self.sample_fidxs[selected_pts_mask].repeat(N)
        new_sample_bary = self.sample_bary[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc, new_sample_fidxs, new_sample_bary)

        prune_filter = torch.cat((selected_pts_mask_raw, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)[self.n_smplx_points:]
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_learnable_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]
        new_sample_fidxs = self.sample_fidxs[selected_pts_mask]
        new_sample_bary = self.sample_bary[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_sample_fidxs, new_sample_bary)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_smplx_triangles_normals(self, smplx_mesh):

        # cano_mesh = {
        #     'mesh_verts': verts,
        #     # 'mesh_norms': torch.tensor(obj.vertex_normals),
        #     'mesh_norms': face_normals,
        #     'mesh_faces':  faces
        # }

        # def auto_normals(vertices, faces):
        #     v0 = vertices[faces[:, 0], :]
        #     v1 = vertices[faces[:, 1], :]
        #     v2 = vertices[faces[:, 2], :]
        #     nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0))
        #     return nrm

        # normals = self.geometry.smplx_mesh.nrm.unsqueeze(0)
        # verts = torch.tensor(np.array(smplx_mesh.vertices), dtype=torch.float32, device='cuda') # .unsqueeze(0)
        # faces =  torch.tensor(np.array(smplx_mesh.triangles), dtype=torch.long, device='cuda') # .unsqueeze(0)
        verts = smplx_mesh['mesh_verts']
        faces = smplx_mesh['mesh_faces']
        normals = torch.nn.functional.normalize(smplx_mesh['mesh_norms'])
        self.smplx_verts = verts.unsqueeze(0)
        self.smplx_faces = faces.unsqueeze(0).cuda()
        self.smplx_triangles = face_vertices(self.smplx_verts, self.smplx_faces)
        self.smplx_normals = face_vertices(normals.unsqueeze(0), self.smplx_faces)


    def get_sdf(self, points):
        residues, pts_ind, _ = point_to_mesh_distance(points, self.smplx_triangles)
        closest_triangles = torch.gather(self.smplx_triangles, 1,
                                            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        closest_normals = torch.gather(self.smplx_normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                                                    3)).view(-1, 3, 3)

        bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)


        pts_norm = (closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
            [-1.0, 1.0, -1.0]).type_as(self.smplx_normals)
        pts_norm = F.normalize(pts_norm, dim=2)
        pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

        pts_signs = 2.0 * (check_sign(self.smplx_verts, self.smplx_faces[0], points).float() - 0.5)
        pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

        return pts_sdf.unsqueeze(0)
    
    def calculate_sdf_loss(self):

        # get mesh without smplx
        # cloth_verts, cloth_faces = self.geometry.get_verts_faces(_training=True)
        cloth_verts = self.get_learnable_xyz
        pts_sdf = self.get_sdf(cloth_verts.unsqueeze(0)).squeeze(0)
        return pts_sdf
    
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)


    def initialize_by_category(self, input=None, input_mesh=None, lgm_ply=None, reinitialize=False, category=2, recolor=False):
        # load checkpoint
        

        verts, faces, face_normals = load_from_smplx_path(input_mesh)
        cano_mesh = {
            'mesh_verts': verts,
            # 'mesh_norms': torch.tensor(obj.vertex_normals),
            'mesh_norms': face_normals,
            'mesh_faces':  faces
        }

        if input is not None:
            self.gaussians.create_from_canonical_smplx_winput(cano_mesh, input)
        else:
            self.gaussians.create_from_canonical_smplx(cano_mesh)
        self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        # self.gaussians.load_ply_fidxs_category(lgm_ply, category)
        self.gaussians.load_ply_fidxs_category(lgm_ply, category, recolor)
        if reinitialize:
            self.gaussians.reinitialize_positions()
        self.gaussians.get_smplx_triangles_normals(cano_mesh)

    def initialize_from_smplx_gaussians_fine(self, input=None, input_mesh=None, num_pts=10000, lgm_ply=None):
        # load checkpoint
        # self.gaussians.load_ply_smplx(input)

        # mesh to sdf
        # input_mesh = o3d.io.read_triangle_mesh(input_mesh)
        # self.gaussians.get_smplx_triangles_normals(input_mesh)

        if input_mesh.endswith(('.obj')):
            verts, faces, face_normals = load_from_smplx_obj(input_mesh)
        elif input_mesh.endswith(('.npz')):
            verts, faces, face_normals = load_from_smplx_path(input_mesh)

        
        cano_mesh = {
            'mesh_verts': verts,
            # 'mesh_norms': torch.tensor(obj.vertex_normals),
            'mesh_norms': face_normals,
            'mesh_faces':  faces
        }

        if input is not None:
            self.gaussians.create_from_canonical_smplx_winput_fine(cano_mesh, input)
        else:
            self.gaussians.create_from_canonical_smplx(cano_mesh, num_pts)
        self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        if lgm_ply is not None:
            self.gaussians.load_ply_categorical(lgm_ply, seg_config.get().nonskin_categories)
        else:
            self.gaussians.create_from_canonical(cano_mesh, num_pts)


        self.gaussians.get_smplx_triangles_normals(cano_mesh)

    def initialize_from_smplx_gaussians_v2(self, input=None, input_mesh=None, num_pts=10000, lgm_ply=None):
        # load checkpoint
        # self.gaussians.load_ply_smplx(input)

        # mesh to sdf
        # input_mesh = o3d.io.read_triangle_mesh(input_mesh)
        # self.gaussians.get_smplx_triangles_normals(input_mesh)

        if input_mesh.endswith(('.obj')):
            verts, faces, face_normals = load_from_smplx_obj(input_mesh)
        elif input_mesh.endswith(('.npz')):
            verts, faces, face_normals = load_from_smplx_path(input_mesh, pose_type='custom-pose')

        
        cano_mesh = {
            'mesh_verts': verts,
            # 'mesh_norms': torch.tensor(obj.vertex_normals),
            'mesh_norms': face_normals,
            'mesh_faces':  faces
        }

        if input is not None:
            self.gaussians.create_from_canonical_smplx_winput(cano_mesh, input)
        else:
            self.gaussians.create_from_canonical_smplx(cano_mesh, num_pts)
        self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        if lgm_ply is not None:
            # _v2 deliberately isolates a single tag (hair) rather than the full nonskin set.
            self.gaussians.load_ply_categorical(lgm_ply, seg_config.get().tag_categories['hair'])
        else:
            self.gaussians.create_from_canonical(cano_mesh, num_pts)


        self.gaussians.get_smplx_triangles_normals(cano_mesh)

    def initialize_from_smplx_gaussians(self, input=None, input_mesh=None, num_pts=10000, lgm_ply=None):
        # load checkpoint
        # self.gaussians.load_ply_smplx(input)

        # mesh to sdf
        # input_mesh = o3d.io.read_triangle_mesh(input_mesh)
        # self.gaussians.get_smplx_triangles_normals(input_mesh)

        if input_mesh.endswith(('.obj')):
            verts, faces, face_normals = load_from_smplx_obj(input_mesh)
        elif input_mesh.endswith(('.npz')):
            verts, faces, face_normals = load_from_smplx_path(input_mesh)

        
        cano_mesh = {
            'mesh_verts': verts,
            # 'mesh_norms': torch.tensor(obj.vertex_normals),
            'mesh_norms': face_normals,
            'mesh_faces':  faces
        }

        if input is not None:
            self.gaussians.create_from_canonical_smplx_winput(cano_mesh, input)
        else:
            self.gaussians.create_from_canonical_smplx(cano_mesh, num_pts)
        self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        if lgm_ply is not None:
            self.gaussians.load_ply_categorical(lgm_ply, seg_config.get().nonskin_categories)
        else:
            self.gaussians.create_from_canonical(cano_mesh, num_pts)


        self.gaussians.get_smplx_triangles_normals(cano_mesh)


    # def initialize_from_smplx_obj(self, input=None, input_mesh=None, num_pts=10000, lgm_ply=None):
    #     # load checkpoint
    #     # self.gaussians.load_ply_smplx(input)

    #     # mesh to sdf
    #     # input_mesh = o3d.io.read_triangle_mesh(input_mesh)
    #     # self.gaussians.get_smplx_triangles_normals(input_mesh)

    #     # verts, faces, face_normals = load_from_smplx_path(input_mesh)
    #     input_mesh

    #     cano_mesh = {
    #         'mesh_verts': verts,
    #         # 'mesh_norms': torch.tensor(obj.vertex_normals),
    #         'mesh_norms': face_normals,
    #         'mesh_faces':  faces
    #     }

    #     if input is not None:
    #         self.gaussians.create_from_canonical_smplx_winput(cano_mesh, input)
    #     else:
    #         self.gaussians.create_from_canonical_smplx(cano_mesh, num_pts)
    #     self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
    #     if lgm_ply is not None:
    #         nonskin_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #         # nonskin_categories = [2, 9, 10]
    #         skin_categories = [11, 12, 13, 14, 15]
    #         self.gaussians.load_ply_categorical(lgm_ply, nonskin_categories)
            
    #         # self.gaussians.load_ply(lgm_ply)
    #     else:
    #         self.gaussians.create_from_canonical(cano_mesh, num_pts)


    #     self.gaussians.get_smplx_triangles_normals(cano_mesh)

    def load_from_pretrained(self, input=None, input_mesh=None, lgm_ply=None, reinitialize=False):
        # load checkpoint
        # self.gaussians.load_ply_smplx(input)

        # mesh to sdf
        # input_mesh = o3d.io.read_triangle_mesh(input_mesh)
        # self.gaussians.get_smplx_triangles_normals(input_mesh)

        if input_mesh.endswith(('.obj')):
            verts, faces, face_normals = load_from_smplx_obj(input_mesh)
        elif input_mesh.endswith(('.npz')):
            verts, faces, face_normals = load_from_smplx_path(input_mesh)
        cano_mesh = {
            'mesh_verts': verts,
            # 'mesh_norms': torch.tensor(obj.vertex_normals),
            'mesh_norms': face_normals,
            'mesh_faces':  faces
        }

        # nonskin_categories = [2, 6, 9, 10]

        self.gaussians.create_from_canonical_smplx_winput(cano_mesh, input)
        self.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        
        self.gaussians.load_ply_fidxs(lgm_ply)
        # for removal expts
        # self.gaussians.load_ply_fidxs_categorical(lgm_ply, nonskin_categories)

        if reinitialize:
            self.gaussians.reinitialize_positions()


        self.gaussians.get_smplx_triangles_normals(cano_mesh)

    def initialize_refine_ply(self, input=None):
        # load checkpoint
        self.gaussians.load_refine_ply(input)
    
    # def initialize_from_lgm_wlabels(self, lgm_ply):
    #     nonskin_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     skin_categories = [11, 12, 13, 14, 15]
    #     self.gaussians.load_ply_categorical(lgm_ply, nonskin_categories)

    def render_cam(
        self,
        cam_param,
        img_shape,
        # viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass


        def get_view_matrix(R, t):
            Rt = torch.cat((R, t.view(3,1)),1)
            view_matrix = torch.cat((Rt, torch.FloatTensor([0,0,0,1]).cuda().view(1,4)))
            return view_matrix

        def get_proj_matrix(focal, princpt, img_shape, z_near, z_far, z_sign):
            fov = get_fov(focal, princpt, img_shape)
            fovY = fov[1]
            fovX = fov[0]
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * z_near
            bottom = -top
            right = tanHalfFovX * z_near
            left = -right
            z_sign = 1.0

            proj_matrix = torch.zeros(4, 4).float().cuda()
            proj_matrix[0, 0] = 2.0 * z_near / (right - left)
            proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
            proj_matrix[0, 2] = (right + left) / (right - left)
            proj_matrix[1, 2] = (top + bottom) / (top - bottom)
            proj_matrix[3, 2] = z_sign
            proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
            proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
            return proj_matrix

        def get_fov(focal, princpt, img_shape):
            fov_x = 2 * torch.atan(img_shape[1] / (2 * focal[0]))
            fov_y = 2 * torch.atan(img_shape[0] / (2 * focal[1]))
            fov = torch.FloatTensor([fov_x, fov_y]).cuda()
            return fov

        # Set up rasterization configuration
        bg=torch.ones((3)).float().cuda()
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
                sh_objs = self.gaussians.get_objects
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    
    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
                sh_objs = self.gaussians.get_objects
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


    def render_learnable_categorical(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        category=1
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_learnable_xyz,
                dtype=self.gaussians.get_learnable_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_learnable_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_learnable_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_learnable_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_learnable_scaling
            rotations = self.gaussians.get_learnable_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_learnable_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_learnable_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_learnable_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_learnable_features
                sh_objs = self.gaussians.get_learnable_objects
        else:
            colors_precomp = override_color

        # removal_thresh = 0.6
        # selected_obj_ids = torch.tensor(category).cuda()

        # logits3d = self.gaussians._objects_dc.permute(2,0,1)
        # prob_obj3d = torch.softmax(logits3d,dim=0) # logits3d # t
        # mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        # mask3d = mask.squeeze() # mask.any(dim=0).squeeze()
        # # print(mask3d.shape)

        # mask3d_convex = points_inside_convex_hull(self.gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        # mask3d = torch.logical_or(mask3d,mask3d_convex)
        # mask3d = torch.argmax(sh_objs.squeeze(1),dim=-1) == category

        if isinstance(category, int):
            mask3d = (torch.argmax(sh_objs.squeeze(1),dim=-1) == category)
        else:
            category = torch.tensor(category).cuda()
            # nonskin_idxs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()
            mask3d = torch.isin(torch.argmax(sh_objs.squeeze(1),dim=-1), category)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D[mask3d],
            means2D=means2D[mask3d],
            shs=shs[mask3d],
            sh_objs=sh_objs[mask3d],
            colors_precomp=colors_precomp,
            opacities=opacity[mask3d],
            scales=scales[mask3d],
            rotations=rotations[mask3d],
            cov3D_precomp=cov3D_precomp,
        )

        # rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    

    def render_learnable(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        # category=1
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_learnable_xyz,
                dtype=self.gaussians.get_learnable_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_learnable_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_learnable_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_learnable_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_learnable_scaling
            rotations = self.gaussians.get_learnable_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_learnable_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_learnable_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_learnable_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_learnable_features
                sh_objs = self.gaussians.get_learnable_objects
        else:
            colors_precomp = override_color

        # removal_thresh = 0.6
        # selected_obj_ids = torch.tensor(category).cuda()

        # logits3d = self.gaussians._objects_dc.permute(2,0,1)
        # prob_obj3d = torch.softmax(logits3d,dim=0) # logits3d # t
        # mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        # mask3d = mask.squeeze() # mask.any(dim=0).squeeze()
        # # print(mask3d.shape)

        # mask3d_convex = points_inside_convex_hull(self.gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        # mask3d = torch.logical_or(mask3d,mask3d_convex)
        # mask3d = torch.argmax(sh_objs.squeeze(1),dim=-1) == category

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    
    def render_smplx(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_smplx_xyz,
                dtype=self.gaussians.get_smplx_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_smplx_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_smplx_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_smplx_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_smplx_scaling
            rotations = self.gaussians.get_smplx_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_smplx_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_smplx_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_smplx_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_smplx_features
                sh_objs = self.gaussians.get_smplx_objects
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def render_categorical(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        category=2
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
                sh_objs = self.gaussians.get_objects
        else:
            colors_precomp = override_color

        mask3d = torch.argmax(sh_objs.squeeze(1),dim=-1) == category
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D[mask3d],
            means2D=means2D[mask3d],
            shs=shs[mask3d],
            sh_objs=sh_objs[mask3d],
            colors_precomp=colors_precomp,
            opacities=opacity[mask3d],
            scales=scales[mask3d],
            rotations=rotations[mask3d],
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def render_refine_ply(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        # category=1
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_refine_xyz,
                dtype=self.gaussians.get_refine_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_refine_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_refine_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_refine_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_refine_scaling
            rotations = self.gaussians.get_refine_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_refine_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_refine_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_refine_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_refine_features
                sh_objs = self.gaussians.get_objects
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image.detach(),
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "segmentation": rendered_segmentation.detach(),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    
    def render_categorical_removal(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        category=2
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
                sh_objs = self.gaussians.get_objects
        else:
            colors_precomp = override_color


        # mask3d = ~(torch.argmax(sh_objs.squeeze(1),dim=-1) == category)

        if isinstance(category, int):
            mask3d = ~(torch.argmax(sh_objs.squeeze(1),dim=-1) == category)
        else:
            category = torch.tensor(category).cuda()
            # nonskin_idxs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()
            mask3d = ~(torch.isin(torch.argmax(sh_objs.squeeze(1),dim=-1), category))

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D[mask3d],
            means2D=means2D[mask3d],
            shs=shs[mask3d],
            sh_objs=sh_objs[mask3d],
            colors_precomp=colors_precomp,
            opacities=opacity[mask3d],
            scales=scales[mask3d],
            rotations=rotations[mask3d],
            cov3D_precomp=cov3D_precomp,
        )
        # rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"image": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "segmentation": rendered_segmentation}
    

    def render_fixed(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_fixed_xyz,
                dtype=self.gaussians.get_fixed_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_fixed_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_fixed_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_fixed_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_fixed_scaling
            rotations = self.gaussians.get_fixed_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_fixed_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_fixed_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_fixed_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_fixed_features
                sh_objs = self.gaussians.get_fixed_objects
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_segmentation, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            sh_objs=sh_objs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"image": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "segmentation": rendered_segmentation}
    