import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh import Mesh, safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

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

        self.mesh = Mesh.load(self.opt.mesh, resize=False, front_dir=opt.front_dir)

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
        
        # # extract trainable parameters
        # self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
        # self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))
        

        self.device = 'cuda'
        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        self.trainable = False

        if self.opt.train_seg:
            self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))
            num_class = 16
            self.vclass = nn.Parameter(torch.rand(self.mesh.v.shape[0], num_class))
            self.features =  torch.arange(0, num_class).to('cuda')
            
    def init_params(self, mask):
        boundary_mask = torch.all(self.mesh.albedo > 0.95, dim=-1).to(torch.uint8)
        mask = torch.all(self.mesh.albedo == 0, dim=-1).to(torch.uint8)
        combined_mask = torch.logical_or(boundary_mask, mask).to(torch.uint8)
        
        self.texc_mask = combined_mask.unsqueeze(-1).repeat(1, 1, 3)
        self.inverse_texc_mask = (self.texc_mask == 0).to(torch.uint8)

        self.old_albedo = self.mesh.albedo.clone()
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(torch.ones_like(self.old_albedo)* 0.5))


    def init_params_contours(self, mask, contour_mask):
        boundary_mask = torch.all(contour_mask > 0., dim=-1).to(torch.uint8)
        mask = torch.all(self.mesh.albedo == 0, dim=-1).to(torch.uint8)
        combined_mask = torch.logical_or(boundary_mask, mask).to(torch.uint8)
        
        self.texc_mask = combined_mask.unsqueeze(-1).repeat(1, 1, 3)
        self.inverse_texc_mask = (self.texc_mask == 0).to(torch.uint8)

        self.old_albedo = self.mesh.albedo.clone()
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(torch.ones_like(self.old_albedo)* 0.5))


    def get_params(self):
        params = [
            {'params': self.raw_albedo, 'lr': self.opt.texture_lr},
        ]

        if self.opt.train_geo:
            params.append({'params': self.v_offsets, 'lr': self.opt.geom_lr})

        if self.opt.train_seg:
            params.append({'params': self.vclass, 'lr': self.opt.seg_lr})

        return params

    @torch.no_grad()
    def export_mesh(self, save_path):
        if self.opt.train_seg:

            self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
            self.mesh.vclass = self.vclass.detach()
            self.mesh.write(save_path)
        else:
            # self.mesh.v = (self.mesh.v + self.v_offsets).detach()
            # self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
            self.mesh.albedo = self.old_albedo.detach() * self.inverse_texc_mask + torch.sigmoid(self.raw_albedo.detach()) * self.texc_mask
            self.mesh.write(save_path)

    @staticmethod
    def proj_from_fovx(camera_angle_x, W, H, near=0.01, far=100.0, y_down=True):
        """
        Build an OpenGL-style perspective matrix using horizontal FOV.
        If y_down=True, the matrix encodes image coords (v downward).
        """
        aspect = W / H
        fx_ndc = 1.0 / np.tan(camera_angle_x / 2.0)         # = proj[0,0]
        fy_ndc = aspect / np.tan(camera_angle_x / 2.0)      # = proj[1,1] (if y_up)
        sgn = -1.0 if y_down else 1.0

        nf = 1.0 / (near - far)
        proj = np.array([
            [fx_ndc, 0.0,    0.0,                          0.0 ],
            [0.0,    sgn*fy_ndc, 0.0,                      0.0 ],
            [0.0,    0.0,   (far + near) * nf,  (2.0 * far * near) * nf],
            [0.0,    0.0,   -1.0,                          0.0 ],
        ], dtype=np.float32)
        return proj


    @torch.no_grad()
    def export_mesh_raw(self, save_path):
        # self.mesh.v = (self.mesh.v + self.v_offsets).detach()
        # self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
        # self.mesh.albedo = self.old_albedo.detach() * self.inverse_texc_mask + torch.sigmoid(self.raw_albedo.detach()) * self.texc_mask
        self.mesh.write(save_path)
    
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
            v = self.mesh.v + self.v_offsets # [N, 3]
        else:
            v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        if self.opt.train_seg:

            alpha = (rast[0, ..., 3:] > 0).float()
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
            depth = depth.squeeze(0) # [H, W, 1]

            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            albedo = dr.texture(self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]
            albedo = torch.sigmoid(albedo)
            # get vn and render normal
            vn = self.mesh.vn
            
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal[0])
            seg, _ = dr.interpolate(self.vclass.unsqueeze(0).contiguous(), rast, self.mesh.f)
            seg = seg.squeeze(0)

            # rotated normal (where [0, 0, 1] always faces camera)
            rot_normal = normal @ pose[:3, :3]
            viewcos = rot_normal[..., [2]]

            # antialias
            albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            albedo = alpha * albedo + (1 - alpha) * bg_color
            

        else:
            disp = -1 / (v_cam[..., [2]] + 1e-20)
            disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
            depth, _ = dr.interpolate(disp, rast, self.mesh.f) # [1, H, W, 1]
            depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]

            alpha = (rast[..., 3:] > 0).float()

            # rgb texture
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            
            if self.trainable:
                trainable_albedo = self.old_albedo * self.inverse_texc_mask + torch.sigmoid(self.raw_albedo) * self.texc_mask
                albedo = dr.texture(trainable_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]
            else:
                albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]


            # get vn and render normal
            vn = self.mesh.vn
            
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal[0])

            # rotated normal (where [0, 0, 1] always faces camera)
            rot_normal = normal @ pose[:3, :3]

            # rot normal z axis is exactly viewdir-normal cosine
            viewcos = rot_normal[..., [2]].abs() # double-sided

            # antialias
            albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
            alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

            # replace background
            albedo = alpha * albedo + (1 - alpha) * self.bg
            normal = alpha * normal + (1 - alpha) * self.bg_normal
            rot_normal = alpha * rot_normal + (1 - alpha) * self.bg_normal

            # Compute face indices buffer
            face_indices = rast[..., 3] - 1  # Face indices are stored in the alpha channel (subtract 1 to get 0-based index)
            face_indices = face_indices.clamp(min=0).long()  # Ensure indices are non-negative


            # extra texture (hard coded)
            if hasattr(self.mesh, 'cnt'):
                cnt = dr.texture(self.mesh.cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                cnt = dr.antialias(cnt, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
                cnt = alpha * cnt + (1 - alpha) * 1 # 1 means no-inpaint in background
                results['cnt'] = cnt
            
            if hasattr(self.mesh, 'viewcos_cache'):
                viewcos_cache = dr.texture(self.mesh.viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
                results['viewcos_cache'] = viewcos_cache

            if hasattr(self.mesh, 'ori_albedo'):
                ori_albedo = dr.texture(self.mesh.ori_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                ori_albedo = dr.antialias(ori_albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
                ori_albedo = alpha * ori_albedo + (1 - alpha) * self.bg
                results['ori_image'] = ori_albedo

        # ssaa
        if ssaa != 1:
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = albedo.clamp(0, 1)
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos

        if self.opt.train_seg:
            seg = scale_img_hwc(seg, (h0, w0))
            results['segmentation'] = seg
        else:

            results['uvs'] = texc.squeeze(0)
            results['face_indices'] = face_indices  

        return results