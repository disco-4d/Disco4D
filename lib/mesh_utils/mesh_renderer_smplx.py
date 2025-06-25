import os
import math
import cv2
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import copy
from lib.mesh_utils.mesh import Mesh, safe_normalize


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0]
                                                                 and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(
                y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)


class Renderer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = 'cuda'

        if opt.train_smplx_params:
            from utils.human_models import smpl_x
            self.smplx_layer = copy.deepcopy(
                smpl_x.layer['neutral']).to(self.device)
            params = np.load(opt.smplx_params_path)
            self._root_pose = nn.Parameter(torch.tensor(
                params['global_orient'], dtype=torch.float32, device=self.device))
            self._body_pose = nn.Parameter(torch.tensor(
                params['body_pose'], dtype=torch.float32, device=self.device).reshape(-1).unsqueeze(0))
            self._lhand_pose = nn.Parameter(torch.tensor(
                params['left_hand_pose'], dtype=torch.float32, device=self.device).reshape(-1).unsqueeze(0))
            self._rhand_pose = nn.Parameter(torch.tensor(
                params['right_hand_pose'], dtype=torch.float32, device=self.device).reshape(-1).unsqueeze(0))
            self._jaw_pose = nn.Parameter(torch.tensor(
                params['jaw_pose'], dtype=torch.float32, device=self.device))
            self._shape = nn.Parameter(torch.tensor(
                params['betas'], dtype=torch.float32, device=self.device))
            self._expr = nn.Parameter(torch.tensor(
                params['expression'], dtype=torch.float32, device=self.device))
            self._cam_trans = nn.Parameter(torch.zeros(
                (1, 3), dtype=torch.float32, device=self.device))

        if '.ply' in self.opt.mesh:
            self.mesh = Mesh.load_trimesh(self.opt.mesh)
        else:
            self.mesh = Mesh.load(
                self.opt.mesh, resize=False, front_dir=opt.front_dir)

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        if opt.train_geo:
            self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))

        if self.mesh.albedo is not None:
            self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))
        elif self.mesh.vc is not None:
            C0 = 0.28209479177387814
            self.mesh.vc = (self.mesh.vc - 0.5) / C0
            self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.vc))
        else:
            self.raw_albedo = None

        if opt.rescale_smplx:
            if opt.load_optimised:
                data = np.load(opt.smplx_params_path)
                self.scale = nn.Parameter(torch.tensor(
                    data['scale'], dtype=torch.float32, device=self.device))
                self.trans = nn.Parameter(torch.tensor(
                    data['translation'], dtype=torch.float32, device=self.device))
            else:
                self.scale = nn.Parameter(
                    torch.ones(1, device=self.device) * 0.8)
                self.trans = nn.Parameter(torch.tensor(
                    [-0.0400, 0.4400], dtype=torch.float32, device=self.device))
                self.trans_z = nn.Parameter(torch.tensor(
                    [0.], dtype=torch.float32, device=self.device))

        if opt.train_tex:
            self.trainable = False
            self.bg = torch.tensor(
                [1, 1, 1], dtype=torch.float32, device=self.device)

        if opt.train_seg:
            num_class = 16
            self.vclass = nn.Parameter(
                torch.rand(self.mesh.v.shape[0], num_class))
            self.features = torch.arange(0, num_class).to('cuda')

    def get_params(self):
        params = []
        if self.opt.train_tex:
            params.append({'params': self.raw_albedo,
                          'lr': self.opt.texture_lr, 'name': 'tex'})
        if self.opt.train_seg:
            params.append({'params': self.vclass, 'lr': self.opt.seg_lr})
        if self.opt.train_geo:
            params.append({'params': self.v_offsets,
                          'lr': self.opt.geom_lr, 'name': 'geo'})
        if self.opt.rescale_smplx:
            params.extend([
                {'params': self.scale, 'lr': 1e-3, 'name': 'scale'},
                {'params': self.trans, 'lr': 1e-3, 'name': 'trans'},
                {'params': self.trans_z, 'lr': 1e-3, 'name': 'trans_z'},
            ])
        if self.opt.train_smplx_params:
            pose_lr = self.opt.smplx_lr
            params.extend([
                {'params': self._root_pose, 'lr': pose_lr, 'name': 'root_pose'},
                {'params': self._body_pose, 'lr': pose_lr, 'name': 'body_pose'},
                {'params': self._lhand_pose, 'lr': pose_lr, 'name': 'lhand_pose'},
                {'params': self._rhand_pose, 'lr': pose_lr, 'name': 'rhand_pose'},
                {'params': self._jaw_pose, 'lr': pose_lr, 'name': 'jaw_pose'},
                {'params': self._shape, 'lr': pose_lr, 'name': 'shape'},
                {'params': self._expr, 'lr': 0, 'name': 'expr'},
                {'params': self._cam_trans, 'lr': 0, 'name': 'cam_trans'},
            ])
        return params

    def init_texture_params(self, mask):
        # boundary_mask = (self.mesh.albedo.sum(-1) > 2.7).to(torch.uint8)
        # boundary_mask = torch.all(self.mesh.albedo > 0.85, dim=-1).to(torch.uint8)
        boundary_mask = torch.all(
            self.mesh.albedo > 0.95, dim=-1).to(torch.uint8)
        mask = torch.all(self.mesh.albedo == 0, dim=-1).to(torch.uint8)
        combined_mask = torch.logical_or(boundary_mask, mask).to(torch.uint8)

        self.texc_mask = combined_mask.unsqueeze(-1).repeat(1, 1, 3)
        self.inverse_texc_mask = (self.texc_mask == 0).to(torch.uint8)

        self.old_albedo = self.mesh.albedo.clone()
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(
            torch.ones_like(self.old_albedo) * 0.5))

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros(
            (1, 3), device=self.device).repeat(batch_size, 1)
        output = self.smplx_layer(
            betas=shape,
            body_pose=body_pose,
            global_orient=root_pose,
            right_hand_pose=rhand_pose,
            left_hand_pose=lhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=expr
        )
        mesh_cam = output.vertices
        joint_cam = output.joints
        return mesh_cam, joint_cam

    def forward_model(self):
        mesh_cam, joint_cam = self.get_coord(
            self._root_pose,
            self._body_pose,
            self._lhand_pose,
            self._rhand_pose,
            self._jaw_pose,
            self._shape,
            self._expr,
            self._cam_trans
        )
        return {'mesh_cam': mesh_cam, 'joint_cam': joint_cam}

    def export_mesh(self, save_path):
        if self.opt.train_smplx_params:
            out = self.forward_model()
            v = out['mesh_cam'].squeeze(0)
        elif self.opt.train_geo:
            v = (self.mesh.v + self.v_offsets).detach()
        else:
            v = self.mesh.v.detach()

        if self.opt.rescale_smplx:
            v[:, 1] *= -1
            v[:, 2] *= -1
            new_center = self.trans if self.trans.shape[0] == 3 else torch.cat(
                [self.trans, self.trans_z])
            v = (v - new_center) * self.scale

        if self.opt.train_tex:
            self.mesh.albedo = self.old_albedo.detach() * self.inverse_texc_mask + \
                torch.sigmoid(self.raw_albedo.detach()) * self.texc_mask
        else:
            self.mesh.v = v
            self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.mesh.write(save_path)

        if self.opt.train_smplx_params:
            opt_param = {
                'global_orient': self._root_pose.detach().cpu().numpy(),
                'body_pose': self._body_pose.detach().cpu().numpy(),
                'left_hand_pose': self._lhand_pose.detach().cpu().numpy(),
                'right_hand_pose': self._rhand_pose.detach().cpu().numpy(),
                'jaw_pose': self._jaw_pose.detach().cpu().numpy(),
                'leye_pose': np.zeros((1, 3)),
                'reye_pose': np.zeros((1, 3)),
                'betas': self._shape.detach().cpu().numpy(),
                'expression': self._expr.detach().cpu().numpy(),
                'transl': self._cam_trans.detach().cpu().numpy(),
                'scale': self.scale.detach().cpu().numpy(),
                'translation': torch.cat([self.trans, self.trans_z]).detach().cpu().numpy()
            }
            npz_path = save_path.replace('.obj', '_smplx.npz')
            np.savez(npz_path, **opt_param)

    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1, texture_filter='linear-mipmap-linear'):

        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0

        results = {}

        # get v
        if self.opt.train_geo:
            v = self.mesh.v + self.v_offsets  # [N, 3]
        elif self.opt.train_smplx_params:
            out = self.forward_model()  # ['mesh_cam'].squeeze(0)
            v = out['mesh_cam'].squeeze(0)
        else:
            v = self.mesh.v

        if self.opt.rescale_smplx:
            v[:, 1] *= -1
            v[:, 2] *= -1
            if self.trans.shape[0] == 3:
                new_center = self.trans
            else:
                new_center = torch.cat([self.trans, self.trans_z])
            v = (v - new_center) * self.scale  # new_scale

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(
            0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        # alpha = (rast[0, ..., 3:] > 0).float()
        alpha = dr.antialias(
            (rast[..., -1:] > 0).float(), rast, v_clip, self.mesh.f).squeeze(0)
        # [1, H, W, 1]
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f)
        depth = depth.squeeze(0)  # [H, W, 1]

        if self.opt.train_tex:
            # actually disparity (1 / depth), to align with controlnet
            disp = -1 / (v_cam[..., [2]] + 1e-20)
            disp = (disp - disp.min()) / (disp.max() -
                                          disp.min() + 1e-20)  # pre-normalize
            depth, _ = dr.interpolate(disp, rast, self.mesh.f)  # [1, H, W, 1]
            depth = depth.clamp(0, 1).squeeze(0)  # [H, W, 1]

            alpha = (rast[..., 3:] > 0).float()

            # rgb texture
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(
                0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')

            if self.trainable:
                trainable_albedo = self.old_albedo * self.inverse_texc_mask + \
                    torch.sigmoid(self.raw_albedo) * self.texc_mask
                albedo = dr.texture(trainable_albedo.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [1, H, W, 3]
            else:
                albedo = dr.texture(self.mesh.albedo.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [1, H, W, 3]

        else:
            if self.mesh.vt is not None:
                texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(
                    0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
                albedo = dr.texture(self.raw_albedo.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode=texture_filter)  # [1, H, W, 3]
                albedo = torch.sigmoid(albedo)
            else:
                albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(
                    0).contiguous(), rast, self.mesh.f, rast_db=rast_db, diff_attrs='all')
                albedo = torch.sigmoid(albedo)

        # get vn and render normal
        if self.opt.train_geo or self.opt.train_smplx_params:
            i0, i1, i2 = self.mesh.f[:, 0].long(
            ), self.mesh.f[:, 1].long(), self.mesh.f[:, 2].long()
            v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = safe_normalize(face_normals)

            vn = torch.zeros_like(v)
            vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
            vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
            vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

            vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn,
                             torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
        else:
            vn = self.mesh.vn

        if self.opt.train_smplx_params:  # and self.opt.rescale_smplx:
            # Convert NDC to screen space
            ndc = v_clip / v_clip[..., 3:]
            screen_x = (ndc[..., 0] + 1) * 1024 / 2
            screen_y = (ndc[..., 1] + 1) * 1024 / 2  # Flip y-axis if necessary

            # Combine x and y coordinates
            screen_vertices = torch.stack(
                [screen_x, screen_y], dim=-1).squeeze(0)

            results['proj_verts'] = screen_vertices
            j = out['joint_cam'].squeeze(0)

            if self.opt.rescale_smplx:
                j[:, 1] *= -1
                j[:, 2] *= -1
                j = (j - new_center) * self.scale  # new_scale
            j_cam = torch.matmul(F.pad(j, pad=(
                0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
            j_clip = j_cam @ proj.T
            j_ndc = j_clip / j_clip[..., 3:]
            screen_j_x = (j_ndc[..., 0] + 1) * 1024 / 2
            screen_j_y = (j_ndc[..., 1] + 1) * 1024 / \
                2  # Flip y-axis if necessary
            screen_joints = torch.stack(
                [screen_j_x, screen_j_y], dim=-1).squeeze(0)

            results['proj_lmk'] = screen_joints

            results['v'] = v

        if self.opt.train_tex:
            alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(
                0).clamp(0, 1)  # [H, W, 3]
            # Compute face indices buffer
            # Face indices are stored in the alpha channel (subtract 1 to get 0-based index)
            face_indices = rast[..., 3] - 1
            # Ensure indices are non-negative
            face_indices = face_indices.clamp(min=0).long()

            # extra texture (hard coded)
            if hasattr(self.mesh, 'cnt'):
                cnt = dr.texture(self.mesh.cnt.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                cnt = dr.antialias(cnt, rast, v_clip,
                                   self.mesh.f).squeeze(0)  # [H, W, 3]
                # 1 means no-inpaint in background
                cnt = alpha * cnt + (1 - alpha) * 1
                results['cnt'] = cnt

            if hasattr(self.mesh, 'viewcos_cache'):
                viewcos_cache = dr.texture(self.mesh.viewcos_cache.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                viewcos_cache = dr.antialias(
                    viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0)  # [H, W, 3]
                results['viewcos_cache'] = viewcos_cache

            if hasattr(self.mesh, 'ori_albedo'):
                ori_albedo = dr.texture(self.mesh.ori_albedo.unsqueeze(
                    0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                ori_albedo = dr.antialias(
                    ori_albedo, rast, v_clip, self.mesh.f).squeeze(0)  # [H, W, 3]
                ori_albedo = alpha * ori_albedo + (1 - alpha) * self.bg
                results['ori_image'] = ori_albedo

        normal, _ = dr.interpolate(vn.unsqueeze(
            0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])
        normal = alpha * normal + (1 - alpha) * bg_color

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(
            0).clamp(0, 1)  # [H, W, 3]
        albedo = alpha * albedo + (1 - alpha) * bg_color
        # ssaa
        if ssaa != 1:
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = albedo
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos
        if self.opt.train_tex:
            results['uvs'] = texc.squeeze(0)
            results['face_indices'] = face_indices
        if self.opt.train_seg:
            seg, _ = dr.interpolate(self.vclass.unsqueeze(
                0).contiguous(), rast, self.mesh.f)
            seg = seg.squeeze(0)
            results['segmentation'] = seg

        return results
