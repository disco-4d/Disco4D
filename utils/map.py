import torch
from torch import nn
import torch.nn.functional as F
import trimesh
import numpy as np
import igl

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def tbn(triangles):
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a), dim=-1)
    d = b - a

    X = F.normalize(torch.cross(d, n), dim=-1)
    Y = F.normalize(torch.cross(d, X), dim=-1)
    Z = F.normalize(d, dim=-1)

    return torch.stack([X, Y, Z], dim=3)


def triangle2projection(triangles):
    R = tbn(triangles)
    T = triangles.unbind(-2)[0]
    I = torch.repeat_interleave(torch.eye(4, device=triangles.device)[None, None, ...], R.shape[1], 1)

    I[:, :, 0:3, 0:3] = R
    I[:, :, 0:3, 3] = T

    return I


def calculate_centroid(tris, dim=2):
    c = tris.sum(dim) / 3
    return c


def interpolate(xyzs, tris, neighbours, edge_mask):
    # Currently disabled
    return triangle2projection(tris)[0]
    N = xyzs.shape[0]
    factor = 4
    c_closests = calculate_centroid(tris)
    c_neighbours = calculate_centroid(neighbours)
    dc = torch.exp(-1 * torch.norm(xyzs - c_closests[0], dim=1, keepdim=True))
    dn = torch.exp(-factor * torch.norm(xyzs.repeat(1, 3).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask
    distances = torch.cat([dc, dn.reshape(N, -1)], dim=1)
    triangles = torch.cat([triangle2projection(tris)[0][:, None, ...], triangle2projection(neighbours)[0][:, None, ...].reshape(N, -1, 4, 4)], dim=1)
    normalization = distances.sum(-1, keepdim=True)
    weights = distances / normalization

    return (triangles * weights[..., None, None]).sum(1)


def project_position(xyzs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(xyzs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)
    Rt_canon = interpolate(xyzs, canonical_triangles, canonical_neighbours, edge_mask)

    homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, 0:1]], dim=1)[:, :, None]

    def_local = torch.matmul(Rt_def, homo)
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()


def project_direction(dirs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(dirs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)[:, 0:3, 0:3]
    Rt_canon = interpolate(dirs, canonical_triangles, canonical_neighbours, edge_mask)[:, 0:3, 0:3]

    def_local = torch.matmul(Rt_def, dirs[:, :, None])
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()

def get_triangles(mesh: trimesh.Trimesh):
    if isinstance(mesh, trimesh.Trimesh):
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces.astype(np.int64), dtype=torch.long)
        return vertices[faces]
    else:
        if isinstance(mesh.V, np.ndarray):
            vertices = torch.tensor(mesh.V, dtype=torch.float32)
            faces = torch.tensor(mesh.F.astype(np.int64), dtype=torch.long)
        else:
            vertices = mesh.V.float()

        if isinstance(mesh.F, np.ndarray):
            faces = torch.tensor(mesh.F.astype(np.int64), dtype=torch.long)
        else:
            faces = mesh.F.long()
        return vertices[faces]  

def rotation_matrix_to_quaternion(R):
    # tr = torch.eye(3)[None,...].repeat(R.shape[0], 1, 1).to(R)
    # w = torch.pow(1 + (tr * R).sum(-1).sum(-1), 0.5)/2
    # x = (R[:, 2, 1] - R[:, 1, 2])/4/w
    # y = (R[:, 0, 2] - R[:, 2, 0])/4/w
    # z = (R[:, 1, 0] - R[:, 0, 1])/4/w
    # quat = torch.stack([w, x, y, z], dim=-1)
    # return quat

    # from pytorch3d import transforms as tfs
    # return tfs.matrix_to_quaternion(R)
    return matrix_to_quaternion(R)

def quaternion_to_rotation_matrix(r):
    # norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    # q = r / norm[:, None]

    # R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # r = q[:, 0]
    # x = q[:, 1]
    # y = q[:, 2]
    # z = q[:, 3]

    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # R[:, 0, 1] = 2 * (x*y - r*z)
    # R[:, 0, 2] = 2 * (x*z + r*y)
    # R[:, 1, 0] = 2 * (x*y + r*z)
    # R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    # R[:, 1, 2] = 2 * (y*z - r*x)
    # R[:, 2, 0] = 2 * (x*z - r*y)
    # R[:, 2, 1] = 2 * (y*z + r*x)
    # R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    # return R
    # from pytorch3d import transforms as tfs
    # return tfs.quaternion_to_matrix(r)

    return quaternion_to_matrix(r)

def calc_per_face_Rt(cano_triangles, deform_triangles):
    # c2w for triangles
    cano_Rt = triangle2projection(cano_triangles)[0]
    deform_Rt = triangle2projection(deform_triangles)[0]

    # for X_c in cano
    # X_d = deform_Rt @ cano_Rt.inv() @ X_c
    return torch.einsum('bij,bjk->bik', deform_Rt, torch.inverse(cano_Rt))

def calc_per_vert_quaternion(cano_verts, cano_faces, mesh_verts):
    cano_triangles = cano_verts[cano_faces].unsqueeze(dim=0)
    deform_triangles = mesh_verts[cano_faces].unsqueeze(dim=0)
    per_face_Rt = calc_per_face_Rt(cano_triangles, deform_triangles)
    per_face_quat = rotation_matrix_to_quaternion(per_face_Rt[:, :3, :3])

    # https://github.com/libigl/libigl/blob/3cf08b7f681ed0e170d16d7a2efea61c3084be78/include/igl/per_vertex_normals.cpp#L61C8-L61C8
    A = igl.doublearea(cano_verts.numpy(), cano_faces.numpy())
    W = torch.from_numpy(A)[:, None].float()

    per_vert_w = torch.zeros(cano_verts.shape[0])
    per_vert_w = per_vert_w.scatter_(0, cano_faces.view(-1), W.repeat([1, 3]).view(-1), reduce='add')

    per_vert_quat = torch.zeros(cano_verts.shape[0], 4)
    per_vert_quat = per_vert_quat.scatter_(0, 
                                           cano_faces[:, :, None].repeat([1, 1, 4]).view(-1, 4), 
                                           (W * per_face_quat)[:, None, :].repeat([1, 3, 1]).view(-1, 4), reduce='add')
    per_vert_quat = per_vert_quat / per_vert_w[:, None]

    # fix nan
    per_vert_quat[per_vert_quat.isnan().any(dim=-1), :] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float32)

    # ### debug face Rt
    # v0 = torch.concat([cano_triangles[0], torch.ones_like(cano_triangles[0, :, :, :1])], dim=-1).float()
    # v01 = torch.einsum('nij,nkj->nki', per_face_Rt, v0)[:, :, :3]
    # mesh01 = libcore.MeshCpu()
    # mesh01.V = v01.reshape(-1, 3).detach().cpu().numpy()
    # mesh01.F = torch.linspace(0, mesh01.V.shape[0], mesh01.V.shape[0]).reshape(-1, 3).long().detach().cpu().numpy()
    # mesh01.save_to_obj('/mnt/e/dummy/01.obj')

    # ### debug R
    # per_vert_R = quaternion_to_rotation_matrix(per_vert_quat)
    # from model import libcore
    # mesh0 = libcore.MeshCpu()
    # mesh0.V = cano_verts.detach().cpu().numpy()
    # mesh0.F = cano_faces.detach().cpu().numpy()
    # mesh0.update_per_vertex_normals()
    # mesh1 = libcore.MeshCpu()
    # mesh1.V = mesh_verts.detach().cpu().numpy()
    # mesh1.F = cano_faces.detach().cpu().numpy()
    # mesh1.update_per_vertex_normals()
    # N01 = torch.einsum('nij,nj->ni', per_vert_R, torch.tensor(mesh0.N, dtype=torch.float))
    # (torch.tensor(mesh1.N, dtype=torch.float) - N01).abs().mean()

    return per_vert_quat

class PerVertQuaternion(nn.Module):
    def __init__(self, cano_verts, cano_faces, use_numpy=False):
        super().__init__()
        self.use_numpy = use_numpy
        self.prepare_cano_per_vert(cano_verts, cano_faces)

    def prepare_cano_per_vert(self, cano_verts, cano_faces):
        self.register_buffer('cano_verts', cano_verts)
        self.register_buffer('cano_faces', cano_faces)
        self.register_buffer('cano_triangles', cano_verts[cano_faces].unsqueeze(dim=0))

        vertices_faces = cano_verts[self.cano_faces]
        side_lengths, heights = calculate_sides_and_heights(vertices_faces)
        scales = torch.stack([heights*0.5, side_lengths*0.5, 0.0001 * torch.ones_like(side_lengths)], dim=1) # .to('cuda')
        # scales[:, -1] *= 100000000
        # scales[:, 1] *= 1.1
        # scales[:, 0] *= 1.1
        scales = torch.log(scales).to('cuda')
        # scales[:, -1] *= 1 # 00
        # scales[:, 1] *= 1.1
        # scales[:, 0] *= 1.1
        
        # scales[:, -1] /= 10
        # scales[:, 1] /= 1.1
        # scales[:, 0] /= 1.1

        self.register_buffer('cano_scales', scales)

        if self.use_numpy:
            # https://github.com/libigl/libigl/blob/3cf08b7f681ed0e170d16d7a2efea61c3084be78/include/igl/per_vertex_normals.cpp#L61C8-L61C8
            A = igl.doublearea(cano_verts.detach().cpu().numpy(), cano_faces.detach().cpu().numpy())
            W = torch.from_numpy(A)[:, None].float()

            per_vert_w_sum = torch.zeros(cano_verts.shape[0])
            per_vert_w_sum = per_vert_w_sum.scatter_(0, cano_faces.view(-1), W.repeat([1, 3]).view(-1), reduce='add')

            self.register_buffer('W', W)
            self.register_buffer('per_vert_w_sum', per_vert_w_sum[:, None])
        else:
            face_areas = calc_face_areas(cano_verts, cano_faces)
            self.register_buffer('face_areas', face_areas.clone())

    def calc_per_vert_quaternion(self, mesh_verts):
        cano_verts = self.cano_verts
        cano_faces = self.cano_faces
        per_face_quat = self.calc_per_face_quaternion(mesh_verts)

        if self.use_numpy:
            # face quat weighted to vert
            per_vert_quat = torch.zeros(cano_verts.shape[0], 4).to(per_face_quat.device)
            per_vert_quat = per_vert_quat.scatter_(0, 
                                                cano_faces[:, :, None].repeat([1, 1, 4]).view(-1, 4), 
                                                (self.W * per_face_quat)[:, None, :].repeat([1, 3, 1]).view(-1, 4), reduce='add')
            per_vert_quat = per_vert_quat / self.per_vert_w_sum

            # normalize
            per_vert_quat = F.normalize(per_vert_quat, eps=1e-6, dim=-1)

        else:
            faces_packed = cano_faces

            verts_quats = torch.zeros(cano_verts.shape[0], 4).to(per_face_quat.device)

            # NOTE: this is already applying the area weighting as the magnitude
            # of the cross product is 2 x area of the triangle.
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 0], self.face_areas * per_face_quat
            )
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 1], self.face_areas * per_face_quat
            )
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 2], self.face_areas * per_face_quat
            )

            per_vert_quat = F.normalize(verts_quats, eps=1e-6, dim=1)

        return per_vert_quat
    
    def calc_per_face_Rt(self, mesh_verts):
        cano_verts = self.cano_verts
        cano_faces = self.cano_faces
        cano_triangles = cano_verts[cano_faces].unsqueeze(dim=0)
        deform_triangles = mesh_verts[cano_faces].unsqueeze(dim=0)
        per_face_Rt = calc_per_face_Rt(cano_triangles, deform_triangles)
        return per_face_Rt
    
    def calc_per_face_quaternion(self, mesh_verts):
        per_face_Rt = self.calc_per_face_Rt(mesh_verts)
        per_face_quat = rotation_matrix_to_quaternion(per_face_Rt[:, :3, :3])
        return per_face_quat
    
    def forward(self, mesh_verts):
        return self.calc_per_vert_quaternion(mesh_verts)
    
    def calc_face_area_change(self, mesh_verts, damping=1e-4):
        areas = calc_face_areas(mesh_verts, self.cano_faces)
        change_ratio = (areas + damping) / (self.face_areas + damping)
        return change_ratio
    
    def calc_face_length_change(self, mesh_verts):
        vertices_faces = mesh_verts[self.cano_faces]
        side_lengths, heights = calculate_sides_and_heights(vertices_faces)
        scales = torch.stack([heights*0.5, side_lengths*0.5, 0.0001 * torch.ones_like(side_lengths)], dim=1) # .to('cuda')
        scales = torch.log(scales).to('cuda')
        # scales[:, -1] *= 100000000
        # scales[:, 1] *= 1.1
        # scales[:, 0] *= 1.1
        # scales[:, -1] *= 100000000
        # scales[:, 1] *= 1.1
        # scales[:, 0] *= 1.1

        change_ratio = scales/ self.cano_scales
        return change_ratio
    
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

def calc_face_areas(mesh_verts, mesh_faces):
    vertices_faces = mesh_verts[mesh_faces]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    face_areas = faces_normals.norm(dim=-1, keepdim=True) / 2.0
    return face_areas

def calc_face_lengths(mesh_verts, mesh_faces):
    vertices_faces = mesh_verts[mesh_faces]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    face_areas = faces_normals.norm(dim=-1, keepdim=True) / 2.0
    return face_areas

def calc_per_vert_rotation(cano_verts, cano_faces, mesh_verts):
    per_vert_quat = calc_per_vert_quaternion(cano_verts, cano_faces, mesh_verts)
    per_vert_R = quaternion_to_rotation_matrix(per_vert_quat)
    return per_vert_R
