import os
import sys

from lib.ops.mesh_to_gs import mesh_to_ply_w_labels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import trimesh
import rembg

from utils.cam_utils import orbit_camera, OrbitCamera, undo_orbit_camera
import json

from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign
import open3d as o3d

from utils.grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh
from lib import seg_config
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import kiui
import random
import glob as _glob  # phase 3 SVD seg loader uses glob; alias to avoid clobbering local 'glob' in __main__

SD_INPAINT = True
pil_to_tensor = transforms.ToTensor()


def remap_obj(obj1_path, save_path, opt):
    """Copy the SMPLX template's UV layout onto obj1_path and write it to save_path.
    Operates on a local opt copy so caller's train_tex / rescale_smplx flags aren't clobbered
    (the renderer needs train_tex=True downstream to emit 'uvs').
    """
    import copy
    from mesh_renderer_smplx import Renderer
    local_opt = copy.deepcopy(opt)
    local_opt.mesh = obj1_path
    local_opt.train_tex = False
    local_opt.rescale_smplx = False
    device = 'cuda'
    renderer = Renderer(local_opt).to(device)
    smplx_tex_mesh = 'data/smplx_uv/smplx_tex.obj'
    mesh = Mesh.load(smplx_tex_mesh, resize=False, front_dir=local_opt.front_dir)
    renderer.mesh.ft = mesh.ft
    renderer.mesh.vt = mesh.vt
    renderer.export_mesh(save_path)


def dilate_image(image, mask, iterations):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image


def _replace_pixels_in_mask(texture_path, mask_path, output_path, threshold, mode):
    """In-place texture cleanup over a SMPLX part region. Pixels in mask>0 that are
    'almost {white|black}' (channel intensity within `threshold` of 255 or 0) are
    replaced with the mean of valid (non-extremal) pixels in that mask region.
    Ported from lib/dep/replace_hand_text*.py and consolidated.
    """
    assert mode in ('white', 'black')
    texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if texture is None or mask is None:
        print(f"[phase4] skip: could not load texture {texture_path} or mask {mask_path}")
        return False

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if mode == 'white':
        extremal = np.all(texture >= (255 - threshold), axis=2)
    else:
        extremal = np.all(texture <= threshold, axis=2)

    valid_mask = (mask_binary > 0) & (~extremal)
    if np.any(valid_mask):
        mean_color = np.mean(texture[valid_mask], axis=0).astype(np.uint8)
    else:
        print(f"[phase4] no non-{mode} pixels in {os.path.basename(mask_path)}; using gray fallback")
        mean_color = np.array([128, 128, 128], dtype=np.uint8)

    replace_mask = (mask_binary > 0) & extremal
    n_replaced = int(replace_mask.sum())
    texture[replace_mask] = mean_color
    cv2.imwrite(output_path, texture)
    return n_replaced


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

def differentiable_mask_loss(points, mask):
    """
    Calculate a differentiable loss based on whether points are within a mask using bilinear interpolation.

    :param points: Tensor of points with shape (batch_size, n_points, 2).
                   Each point is represented as [x, y].
    :param mask: 2D Tensor representing the mask.
    :return: Loss value.
    """
    # Normalize points to be in the range [-1, 1] as expected by grid_sample
    h, w = mask.shape[-2:]
    points = points.clone()
    points[..., 0] = (points[..., 0] / (w - 1)) * 2 - 1
    points[..., 1] = (points[..., 1] / (h - 1)) * 2 - 1

    # Reshape for grid_sample: [batch_size, n_points, 1, 2]
    points = points.unsqueeze(2)

    # Reshape mask for grid_sample: [batch_size, 1, H, W]
    mask = mask.unsqueeze(1)

    # Sample the mask using bilinear interpolation
    sampled_mask = F.grid_sample(mask.float(), points, mode='bilinear', align_corners=False)

    # Compute loss: penalize points where the interpolated mask value is low
    loss = (1 - sampled_mask).sum()
    return loss


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        # self.mode = "image"\
        if self.opt.rescale_smplx:
            self.mode = "alpha"
        else:
            self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        # if self.opt.rescale_smplx:
        # from mesh_smplx_renderer2 import Renderer
        from mesh_renderer_smplx import Renderer
        self.renderer = Renderer(opt).to(self.device)
        # elif self.opt.rescale_obj:
        #     from mesh_renderer import Renderer
        #     self.renderer = Renderer(opt).to(self.device)

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = True # True
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        # self.lpips_loss = LPIPS(net='vgg').to(self.device)
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
            self.initialize(keep_ori_albedo=False)

        if self.opt.input_back is not None:
            self.load_input_back(self.opt.input_back)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # face-level inpaint tracking (SMPLX has 20908 faces)
        self.face_inpainting_mask = torch.zeros(20908).float().cuda()
        self.opt.texture_size = getattr(self.opt, 'texture_size', 1024)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

        self.smplx_end_step = 10000

        if self.opt.opt_lgm:
            self.get_lgm_triangles_normals(self.opt.lgm_mesh)




    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # Phase 2 setup: when training textures without geometry fit (inpaint-batch mode),
        # reset raw_albedo as a fresh trainable parameter via the renderer.
        if self.opt.train_tex and not self.opt.rescale_smplx:
            self.renderer.trainable = True
            self.renderer.init_texture_params(self.face_inpainting_mask)

        # Phase 3 setup: load Segformer + SVD seg masks. num_frames must be set on opt.
        if self.opt.train_seg:
            if not hasattr(self, 'cls_criterion'):
                self.prepare_seg_guidance()
            if self.opt.input_svd is not None and not hasattr(self, 'mv_seg'):
                self.num_frames = self.opt.num_frames
                self.load_svd_input(self.opt.input_svd)

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)

        self.fixed_cam = (pose, self.cam.perspective)
        

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

        # input image
        if self.opt.input_back is not None:
            self.input_back_img_torch = torch.from_numpy(self.input_back_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_back_img_torch = F.interpolate(self.input_back_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_back_mask_torch = torch.from_numpy(self.input_back_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_back_mask_torch = F.interpolate(self.input_back_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            pose = orbit_camera(self.opt.elevation, 180, self.opt.radius)

            self.fixed_back_cam = (pose, self.cam.perspective)
        

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        # Phase 2 (texture-only refinement) and Phase 3 (segmentation) use the static LRs from
        # renderer.get_params() (opt.texture_lr / opt.seg_lr). The schedule below is phase-1
        # specific (scale→pose→shape ramp).
        if not self.opt.rescale_smplx or self.opt.train_seg:
            return

        if iteration< 500:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "pose":
                    # lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = 0
                if param_group["name"] == "size":
                    param_group['lr'] = 1e-3
                # if param_group["name"] == "size_z":
                #     param_group['lr'] = 1e-4
        elif self.smplx_end_step > iteration > 500:
            
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "pose":
                    # lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = 1e-4
                if param_group["name"] == "shape":
                    param_group['lr'] = 1e-3
                if param_group["name"] == "size":
                    param_group['lr'] = 1e-4
                if param_group["name"] == "size_z":
                    param_group['lr'] = 1e-4
        elif iteration >= self.smplx_end_step:
            self.opt.train_tex = True
            # self.opt.train_smplx_params = False
            self.opt.rescale_smplx = True
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "pose":
                    # lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = 0.
                if param_group["name"] == "tex":
                    param_group['lr'] = 0.2

    def get_lgm_triangles_normals(self, smplx_mesh):

        smplx_mesh = o3d.io.read_triangle_mesh(smplx_mesh)

        def auto_normals(vertices, faces):
            v0 = vertices[faces[:, 0], :]
            v1 = vertices[faces[:, 1], :]
            v2 = vertices[faces[:, 2], :]
            nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0))
            return nrm

        # normals = self.geometry.smplx_mesh.nrm.unsqueeze(0)
        verts = torch.tensor(np.array(smplx_mesh.vertices), dtype=torch.float32, device='cuda') # .unsqueeze(0)
        faces =  torch.tensor(np.array(smplx_mesh.triangles), dtype=torch.long, device='cuda') # .unsqueeze(0)
        normals = auto_normals(verts, faces)
        self.smplx_verts = verts.unsqueeze(0)
        self.smplx_faces = faces.unsqueeze(0)

        self.smplx_triangles = face_vertices(self.smplx_verts, self.smplx_faces)
        self.smplx_normals = face_vertices(normals.unsqueeze(0), self.smplx_faces)

    # from kiui.lpips import LPIPS
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
        if torch.isnan(pts_norm).any():
            # Handle the case where there are one or more NaN values in pts_norm
            print("NaN values detected in pts_norm.")

            # break
        # norm_magnitude = torch.norm(pts_norm, dim=2, keepdim=True)
        # pts_norm = torch.where(norm_magnitude > epsilon, pts_norm / norm_magnitude, default_direction)

        epsilon = 1e-8
        pts_dist = torch.sqrt(residues + epsilon) / torch.sqrt(torch.tensor(3) + epsilon)
        if torch.isnan(pts_dist).any():
            # Handle the case where there are one or more NaN values in pts_norm
            print("NaN values detected in pts_dist.")

        pts_signs = 2.0 * (check_sign(self.smplx_verts, self.smplx_faces[0], points).float() - 0.5)
        pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

        return pts_sdf.unsqueeze(0)

    # ---------------------------------------------------------------------
    # Phase 3: mesh segmentation pipeline (ported from main2_mesh_seg.py)
    # ---------------------------------------------------------------------

    def prepare_seg_guidance(self):
        """Load the segmentation model for novel-view pseudo-labels and set up the seg loss criterion.
        Model + label space (num_classes, cat_map) come from lib.seg_config."""
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        seg_cfg = seg_config.get()
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = seg_cfg.num_classes
        self.transform = transforms.ToPILImage()
        self.seg_processor = SegformerImageProcessor.from_pretrained(seg_cfg.model_name)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_cfg.model_name)
        self.cat_map = dict(seg_cfg.cat_map)

    @torch.no_grad()
    def get_seg_inference(self, pred_image):
        image = self.transform(pred_image)
        inputs = self.seg_processor(images=image, return_tensors="pt")
        outputs = self.seg_model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pseudo_gt_seg = pred_seg.clone()
        for k, v in self.cat_map.items():
            pseudo_gt_seg = torch.where(pred_seg == k, v, pseudo_gt_seg)
        return pseudo_gt_seg

    @torch.no_grad()
    def get_cameras(self, elevation, azimuths_deg, radius):
        return [(orbit_camera(elevation, az, radius), self.cam.perspective) for az in azimuths_deg]

    def load_svd_input(self, input_dir):
        """Load pre-computed Segformer-supervised seg masks from SVD-rendered views.
        Sets self.mv_seg (list of [1,1,H,W] long tensors) and self.mv_cameras
        (list of (pose, perspective) pairs aligned with mv_seg)."""
        file_list = sorted(_glob.glob(f'{input_dir}/*_seg.pt'))
        if len(file_list) == 0:
            raise FileNotFoundError(f"[phase3] no *_seg.pt found under {input_dir}")
        self.mv_seg = []
        for seg_file in file_list:
            print(f'[INFO] load seg from {seg_file}...')
            input_seg = torch.load(seg_file)
            input_seg_torch = input_seg.unsqueeze(0).unsqueeze(0).to(self.device).float()
            input_seg_torch = F.interpolate(input_seg_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False).long()
            self.mv_seg.append(input_seg_torch)

        # Build matching camera poses. Special-case 80/40/37 to preserve original behavior;
        # otherwise fall back to a uniform circle (fits our 20-frame case and any other count).
        if self.num_frames == 80:
            indices = [0, 1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24, 26,
                       27, 29, 31, 32, 34, 36, 37, 39, 41, 42, 44, 46, 47, 49, 51, 52, 54,
                       55, 57, 59, 60, 62, 64, 65, 67, 69, 70, 72, 74, 75, 77, 79]
            step = 360 / 80
            ang_list = [-i * step for i in range(80)]
            azimuths_deg = list(np.array(ang_list)[indices])
        elif self.num_frames == 40:
            step = 360 / 40
            azimuths_deg = [i * step for i in range(40)]
        elif self.num_frames == 37:
            step = 360 / 40
            azimuths_deg = [i * step for i in range(37)]
        else:
            azimuths_deg = list(np.linspace(0, 360, self.num_frames, endpoint=False))

        self.mv_cameras = self.get_cameras(self.opt.elevation, azimuths_deg, self.opt.radius)
        if len(self.mv_cameras) != len(self.mv_seg):
            print(f'[WARN] num_frames={self.num_frames} produced {len(self.mv_cameras)} cameras but {len(self.mv_seg)} seg files; truncating to min.')
            n = min(len(self.mv_cameras), len(self.mv_seg))
            self.mv_cameras = self.mv_cameras[:n]
            self.mv_seg = self.mv_seg[:n]

    def get_known_view_loss(self, idx, step_ratio):
        cam, gt_seg = self.mv_cameras[idx], self.mv_seg[idx]
        ssaa = min(2.0, max(0.125, 2 * np.random.random()))
        out = self.renderer.render(*cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)
        pred_seg = out["segmentation"].permute(2, 0, 1).contiguous().unsqueeze(0)
        loss_seg = self.cls_criterion(pred_seg, gt_seg.squeeze(0).cuda()).squeeze(0).mean()
        return 1000 * step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))

    def _train_step_seg(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)
            loss = 0

            if self.opt.use_known_view:
                # original 37-frame variant biased the front view; keep that, otherwise
                # rely on the random sample below to cover the front uniformly.
                if self.num_frames == 37:
                    front_idx = random.choice([0, 20])
                    loss = loss + self.get_known_view_loss(front_idx, step_ratio)
                idx = random.randint(0, len(self.mv_seg) - 1)
                loss = loss + self.get_known_view_loss(idx, step_ratio)
            else:
                # novel view supervised by Segformer pseudo-GT
                render_resolution = 512
                min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
                max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius)
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)
                pred_seg = out["segmentation"].permute(2, 0, 1).contiguous().unsqueeze(0)
                pseudo_gt_seg = self.get_seg_inference(out["image"].permute(2, 0, 1).contiguous())
                loss_seg = self.cls_criterion(pred_seg, pseudo_gt_seg.unsqueeze(0).cuda()).squeeze(0).mean()
                loss = loss + 1000 * step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        self.need_update = True

    def save_vclass(self):
        """Phase 3 only saves vclass.npy alongside the (untouched) clothed mesh."""
        os.makedirs(self.opt.outdir, exist_ok=True)
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        vclass_path = path.replace('.obj', '_vclass.npy')
        np.save(vclass_path, self.renderer.vclass.detach().cpu().numpy())
        print(f"[INFO] saved vclass to {vclass_path}")

    # ---------------------------------------------------------------------
    # Phase 2: inpaint-batch pipeline (ported from main2_inpaint_batch.py)
    # ---------------------------------------------------------------------

    def prepare_sd_guidance(self):
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            device='cuda',
        )
        self.pipe = pipe.to("cuda")

    def prepare_guidance(self):
        self.opt.control_mode = ['depth_inpaint']
        self.opt.model_key = 'philz1337/revanimated'
        self.opt.lora_keys = []
        self.guidance = None

        self.opt.posi_prompt = "masterpiece, high quality"
        self.negative_prompt = "bad quality, worst quality, shadows"
        self.prompt = self.opt.posi_prompt + ', ' + self.prompt

        if self.guidance is None:
            print(f'[INFO] loading guidance model...')
            from guidance.sd_inpaint_utils import StableDiffusion
            self.guidance = StableDiffusion(self.device, control_mode=self.opt.control_mode, model_key=self.opt.model_key, lora_keys=self.opt.lora_keys)
            print(f'[INFO] loaded guidance model!')

        print(f'[INFO] encoding prompt...')
        nega = self.guidance.get_text_embeds([self.negative_prompt])
        self.guidance_embeds = {}
        posi = self.guidance.get_text_embeds([self.prompt])
        self.guidance_embeds['default'] = torch.cat([nega, posi], dim=0)
        for d in ['front', 'side', 'back', 'top', 'bottom']:
            posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
            self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)

    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):
        h = w = 1024  # int(self.opt.texture_size)

        self.albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        self.cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        self.viewcos_cache = -torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        if keep_ori_albedo:
            self.albedo = self.renderer.mesh.albedo.clone()
            self.cnt += 1
            self.viewcos_cache *= -1

        self.renderer.mesh.albedo = self.albedo
        self.renderer.mesh.cnt = self.cnt
        self.renderer.mesh.viewcos_cache = self.viewcos_cache

    @torch.no_grad()
    def backup(self):
        self.backup_albedo = self.albedo.clone()
        self.backup_cnt = self.cnt.clone()
        self.backup_viewcos_cache = self.renderer.mesh.viewcos_cache.clone()

    @torch.no_grad()
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(self.opt.texture_size)
        self.backup()

        mask = self.cnt.squeeze(-1) > 0
        mask = mask.view(h, w).detach().cpu().numpy()

        self.albedo = dilate_image(self.albedo, mask, iterations=int(h * 0.2))
        self.cnt = dilate_image(self.cnt, mask, iterations=int(h * 0.2))

        self.update_mesh_albedo()

    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.opt.texture_size)
        self.backup()

        cur_albedo = self.renderer.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def image_inpaint(self, pose):
        h = w = 1024
        H = W = 1024
        cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > cos_thresh)
        proj_mask = proj_mask.view(1, 1, H, W).float().view(-1).bool()
        uvs = out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous()

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = self.input_img_torch.clone()
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]

        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]

        self.update_mesh_albedo()

        viewcos = out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    @torch.no_grad()
    def image_inpaint_input(self, pose, torch_img, torch_mask):
        h = w = 1024
        H = W = 1024
        cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > cos_thresh)
        proj_mask = proj_mask.view(1, 1, H, W).float()
        proj_mask = (proj_mask * torch_mask).view(-1).bool()

        uvs = out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous()
        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = torch_img.clone()
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]

        # exclude already-inpainted faces
        face_indices = out['face_indices'].view(-1)[proj_mask]
        face_mask = self.face_inpainting_mask[face_indices].bool()
        uvs = uvs[~face_mask]
        rgbs = rgbs[~face_mask]
        face_indices = face_indices[~face_mask]

        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]
        self.face_inpainting_mask[face_indices] = 1

        self.update_mesh_albedo()

        viewcos = out['viewcos'].squeeze(0).permute(2, 0, 1).unsqueeze(0).contiguous()
        viewcos = viewcos.view(-1, 1)[proj_mask]
        viewcos = viewcos[~face_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    @torch.no_grad()
    def inpaint_view(self, pose):
        h = w = 1024
        H = W = 1024
        self.opt.refine_strength = 0.6
        self.opt.vis = False
        self.opt.cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        valid_pixels = out['alpha'].squeeze(-1).nonzero()
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h, min_w, max_h, max_w = 0, 0, H, W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h + 1, min_w:max_w + 1], size, mode=mode)

        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous())

        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1
        viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous())
        viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous())
        mask_refine = ((viewcos_old < viewcos) & ~mask_generate)
        mask_keep = (~mask_generate & ~mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()
        mask_generate_blur = mask_generate

        if not (mask_generate > 0.5).any():
            return

        control_images = {}

        if 'normal' in self.opt.control_mode:
            rot_normal = out['rot_normal']
            rot_normal[..., 0] *= -1
            control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512))

        if 'depth' in self.opt.control_mode:
            depth = out['depth']
            control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1)

        if 'ip2p' in self.opt.control_mode:
            ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512))
            control_images['ip2p'] = ori_image

        if 'inpaint' in self.opt.control_mode:
            image_generate = image.clone()
            image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1
            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            control_images['inpaint'] = image_generate

            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear')
            latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear')
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear')
            control_images['latents_mask'] = latents_mask
            control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep
            control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype))

        if 'depth_inpaint' in self.opt.control_mode:
            image_generate = image.clone()
            image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1
            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1)
            control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1)

            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear')
            latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear')
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear')
            control_images['latents_mask'] = latents_mask
            control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep
            control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype))

        # view-direction token
        ver, hor, _ = undo_orbit_camera(pose)
        if ver <= -60:
            d = 'top'
        elif ver >= 60:
            d = 'bottom'
        else:
            if abs(hor) < 30: d = 'front'
            elif abs(hor) < 90: d = 'side'
            else: d = 'back'
        text_embeds = self.guidance_embeds[d]

        rgbs = self.guidance(text_embeds, height=512, width=512, control_images=control_images, refine_strength=self.opt.refine_strength).float()

        if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
            scale = W // rgbs.shape[-1]
            rgbs = rgbs.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
            rgbs = (rgbs * 255).astype(np.uint8)
            rgbs = kiui.sr.sr(rgbs, scale=scale)
            rgbs = rgbs.astype(np.float32) / 255
            rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)

        rgbs = rgbs * (1 - mask_keep) + image * mask_keep

        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.opt.cos_thresh)
        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]

        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]

        self.update_mesh_albedo()

        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    @torch.no_grad()
    def generate(self):
        self.opt.camera_path = getattr(self.opt, 'camera_path', 'front')

        front_pose = orbit_camera(0, 0, self.opt.radius)
        self.image_inpaint_input(front_pose, self.input_img_torch, self.input_mask_torch)
        back_pose = orbit_camera(180, 0, self.opt.radius)
        self.image_inpaint_input(back_pose, self.input_back_img_torch, self.input_back_mask_torch)

        if self.opt.camera_path == 'default':
            vers = [-15] * 8 + [-89.9, 89.9] + [45]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.opt.camera_path == 'front':
            vers = [0] * 8 + [-89.9, 89.9] + [45]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.opt.camera_path == 'top':
            vers = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            hors = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif self.opt.camera_path == 'side':
            vers = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            hors = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        else:
            raise NotImplementedError(f'camera path {self.opt.camera_path} not implemented!')

        start_t = time.time()

        pose = orbit_camera(0, 0, self.cam.radius)
        self.image_inpaint(pose)

        print(f'[INFO] start generation...')
        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            pose = orbit_camera(ver, hor, self.cam.radius)
            self.inpaint_view(pose)
            self.need_update = True
            self.test_step()

        self.dilate_texture()
        self.deblur()

        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    def train_step(self):
        # Phase 3 (segmentation) takes a separate path — vclass-only training, no SMPLX/zero123/SD.
        if self.opt.train_seg:
            return self._train_step_seg()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # self.prepare_train()


        for _ in range(self.train_steps):

            
            step_ratio = min(1, self.step / self.opt.iters_refine)

            self.update_learning_rate(self.step)

            loss = 0
            losses = {}

            ### known view
            if self.input_img_torch is not None:

                # ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                ssaa = 2.0
                out = self.renderer.render(*self.fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                # # rgb loss # HARDCODED for now
                # if self.opt.train_tex and self.step >= self.smplx_end_step: 
                if self.opt.train_tex:
                    image = out["image"] # [H, W, 3] in [0, 1]
                    valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()

                    if self.opt.exclude_cloth:
                        losses['rgb_loss'] = F.mse_loss(image * valid_mask * exclude_mask, self.input_img_torch_channel_last * valid_mask * exclude_mask)
                    else:
                        losses['rgb_loss'] = F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

                if self.opt.rescale_smplx and self.step < self.smplx_end_step:
                    losses['mask_loss'] = F.mse_loss(out["alpha"][:, :, 0], self.input_mask_torch[0, 0]) * 10000


                    back_out = self.renderer.render(*self.fixed_back_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                    losses['back_mask_loss'] = F.mse_loss(back_out["alpha"][:, :, 0], self.input_back_mask_torch[0, 0]) * 10000

            # if self.opt.train_tex and self.step >= self.smplx_end_step:
            if self.opt.train_tex:
                ### novel view (manual batch)
                render_resolution = 512
                images = []
                vers, hors, radii = [], [], []
                # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
                min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
                max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
                for _ in range(self.opt.batch_size):

                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)

                    # random render resolution
                    ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                    out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                    image = out["image"] # [H, W, 3] in [0, 1]
                    image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                    images.append(image)
                
                images = torch.cat(images, dim=0)

                # import kiui
                # kiui.lo(hor, ver)
                # kiui.vis.plot_image(image)

                # guidance loss
                if self.enable_sd:

                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.guidance_sd.refine(images, strength=0.6).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    losses['sd_guidance_loss'] = self.opt.lambda_sd * F.mse_loss(images, refined_images)

                if self.enable_zero123:
                    # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                    refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=0.6).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    losses['zero123_guidance_loss'] = self.opt.lambda_zero123 * F.mse_loss(images, refined_images) * 100
                    # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

                # if self.step >= 100:
                #     self.opt.rescale_smplx = False
                #     self.opt.train_smplx_params = True

            # HARDCODED for now
            if self.opt.train_smplx_params and self.smplx_end_step> self.step > 500:


                pred_lmk = out['proj_lmk'][self.src_idxs]
                losses['lmk_loss'] = torch.abs(self.gt_lmk- pred_lmk).sum() # * 10

                # silhouette loss: penalise points outside the mask
                losses['within_mask_loss'] =   differentiable_mask_loss(out['proj_verts'].unsqueeze(0), self.input_mask_torch[0]) * 1

            
                # ensure that the smplx mesh lies as close to the smplx points as possible; and below the 
                if self.opt.opt_lgm:
                    verts = out['v']
                    pts_sdf = self.get_sdf(verts.unsqueeze(0)).squeeze(0)
                    losses['sdf_loss'] = torch.relu(-pts_sdf).sum() * 50  # * 5000 # * 0.1 # * 50
                

                # silhouette loss: penalise points outside the mask
                # losses['within_back_mask_loss'] =   differentiable_mask_loss(back_out['proj_verts'].unsqueeze(0), self.input_back_mask_torch[0]) * 1

                

            for k in losses:
                loss += losses[k]
            # optimize step

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step += 1

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            loss_string = '\n'.join([f'{loss_name} = {loss_value.item():.4f}' for loss_name, loss_value in losses.items()])
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f} {loss_string}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            if self.mode in ['canonical_image']:
                out_cat = self.renderer.render_canonical(self.cam.pose, self.cam.perspective, self.H, self.W)
                buffer_image = out_cat["image"]
            else:
                out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)
                buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()
            
            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # mask = img[..., 3:]/255
        # mask = cv2.resize(
        #     mask, (self.W, self.H), interpolation=cv2.INTER_AREA
        # )
        # self.input_mask = mask > 0
        # if img.shape[-1] == 3:
        #     if self.bg_remover is None:
        #         self.bg_remover = rembg.new_session()
        #     img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(
            img, (self.W, self.H), interpolation=cv2.INTER_AREA
        )

        img = img.astype(np.float32) / 255.0

        self.input_mask = (img[..., 3:] > 0).astype(np.float32)
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # materialize torch tensors here so phase-2 (generate) can run right after __init__
        ref_size = getattr(self.opt, 'ref_size', 1024)
        self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_img_torch = F.interpolate(self.input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
        self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_mask_torch = F.interpolate(self.input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
        self.input_img_torch_channel_last = self.input_img_torch[0].permute(1, 2, 0).contiguous()

        # # load prompt
        # file_prompt = file.replace("_rgba.png", "_caption.txt")
        # if os.path.exists(file_prompt):
        #     print(f'[INFO] load prompt from {file_prompt}...')
        #     with open(file_prompt, "r") as f:
        #         self.prompt = f.read().strip()

        if self.opt.exclude_cloth:
            self.exclude_mask = None  # TODO: wire up cloth-exclusion mask when exclude_cloth is enabled

        if self.opt.train_smplx_params:
            # load keypoints
            json_path = file.replace('/img/', '/mmpose/predictions/').replace('.png', '.json')
            with open(json_path, 'r') as file:
                kp_data = json.load(file)
            self.gt_lmk = torch.tensor(np.array(kp_data[0]['keypoints'])).cuda()
            self.src_idxs = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 60, \
                             61, 62, 63, 64, 65, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, \
                                137, 138, 139, 140, 141, 142, 143, 76, 77, 78, 79, 80, 81, 82, 83, 84, \
                                    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, \
                                        102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, \
                                            116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 20, 37, 38, 39, \
                                                66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 21, \
                                                    52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]


    def load_input_back(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_back_mask = img[..., 3:]
        # white bg
        self.input_back_img = img[..., :3] * self.input_back_mask + (1 - self.input_back_mask)
        # bgr to rgb
        self.input_back_img = self.input_back_img[..., ::-1].copy()

        ref_size = getattr(self.opt, 'ref_size', 1024)
        self.input_back_img_torch = torch.from_numpy(self.input_back_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_back_img_torch = F.interpolate(self.input_back_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
        self.input_back_mask_torch = torch.from_numpy(self.input_back_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_back_mask_torch = F.interpolate(self.input_back_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal", "canonical_image"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        # save
        self.save_model()
        

# =============================================================================
# Pipeline orchestration (phase 1 scale_back, phase 2 inpaint, phase 3 mesh_to_gs)
# =============================================================================

def _phase1_configure(opt):
    """Phase 1: fit SMPLX params to image silhouette. Mutates opt in place."""
    opt.front_dir = '+z'
    opt.train_seg = False
    opt.iters_refine = 3000
    opt.rescale_smplx = True
    opt.train_tex = False


def _phase2_configure(opt, DIR, FID):
    """Phase 2: inpaint texture via ControlNet. Returns (orig_mesh, remapped_mesh)."""
    orig_mesh = f'logs/output/{DIR}/{FID}.obj'
    remapped_mesh = f'logs/output/{DIR}/{FID}_remapped.obj'
    opt.mesh = remapped_mesh
    opt.input = f"data/{DIR}/img/{FID}.png"
    opt.input_back = f"data/{DIR}/svd/{FID}/rgba/0010.png"
    opt.save_path = f"output/{DIR}/{FID}_clothed_smplx"
    opt.iters_refine = 750
    opt.front_dir = '+z'
    opt.train_seg = False
    opt.train_tex = True
    opt.rescale_smplx = False
    opt.train_smplx_params = False
    opt.lambda_zero123 = 1
    opt.lambda_sd = 0
    return orig_mesh, remapped_mesh


def run_phase1(opt):
    """Phase 1 — scale_back. Writes rescaled mesh via gui.save_model()."""
    gui = GUI(opt)
    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
    del gui
    torch.cuda.empty_cache()


def run_phase2(opt, orig_mesh, remapped_mesh):
    """Phase 2 — inpaint_batch production flow: remap UVs, seed texture from front+back
    views via UV projection, then refine with zero123-guided training.
    Mirrors main2_inpaint_batch.py:1632-1636.
    """
    if not os.path.exists(orig_mesh):
        raise FileNotFoundError(f"[phase2] expected phase-1 output at {orig_mesh}")
    remap_obj(orig_mesh, remapped_mesh, opt)

    gui = GUI(opt)

    # Seed texture by projecting input front + back images onto the mesh UV.
    front_pose = orbit_camera(0, 0, gui.opt.radius)
    gui.image_inpaint_input(front_pose, gui.input_img_torch, gui.input_mask_torch)
    back_pose = orbit_camera(0, 180, gui.opt.radius)
    gui.image_inpaint_input(back_pose, gui.input_back_img_torch, gui.input_back_mask_torch)

    # Refine texture with zero123 guidance (lambda_zero123=1, lambda_sd=0 set by _phase2_configure).
    gui.train(opt.iters_refine)

    clothed_obj = os.path.join(opt.outdir, opt.save_path + '.' + opt.mesh_format)
    del gui
    torch.cuda.empty_cache()
    return clothed_obj


def _phase3_configure(opt, DIR, FID):
    """Phase 3: mesh segmentation. Returns the SVD seg dir (caller checks existence)."""
    clothed_obj = f'logs/output/{DIR}/{FID}_clothed_smplx.obj'
    input_svd = f'data/{DIR}/svd/{FID}/seg'
    opt.mesh = clothed_obj
    # Keep the canonical save_path so vclass lands at {FID}_clothed_smplx_vclass.npy
    opt.save_path = f"output/{DIR}/{FID}_clothed_smplx"
    opt.train_seg = True
    opt.train_tex = False
    opt.rescale_smplx = False
    opt.train_smplx_params = False
    opt.iters_refine = 3000
    opt.radius = 2.0
    opt.front_dir = '+z'
    opt.use_known_view = True
    opt.input_svd = input_svd
    opt.lambda_zero123 = 0
    opt.lambda_sd = 0
    if not hasattr(opt, 'seg_lr') or opt.seg_lr is None:
        opt.seg_lr = 1e-2
    # default num_frames from disk if YAML didn't set it
    if not hasattr(opt, 'num_frames') or opt.num_frames is None:
        if os.path.isdir(input_svd):
            opt.num_frames = len([f for f in os.listdir(input_svd) if f.endswith('_seg.pt')])
        else:
            opt.num_frames = 0  # will fail in load_svd_input with a clear message
    return input_svd


def run_phase3(opt, input_svd):
    """Phase 3 — mesh segmentation. Trains per-vertex 16-class labels supervised by
    SVD-rendered Segformer masks (known-view) + Segformer pseudo-GT on novel views.
    Writes vclass.npy next to the clothed mesh; does NOT re-export the obj.
    """
    if not os.path.isdir(input_svd):
        print(f"[phase3] skipping segmentation: {input_svd} not found (run SVD seg preprocessing first)")
        return None
    if not os.path.exists(opt.mesh):
        raise FileNotFoundError(f"[phase3] expected phase-2 output at {opt.mesh}")

    gui = GUI(opt)
    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
    gui.save_vclass()

    vclass_path = os.path.join(opt.outdir, opt.save_path + '_vclass.npy')
    del gui
    torch.cuda.empty_cache()
    return vclass_path


def _phase4_configure(opt, DIR, FID):
    """Phase 4: SMPLX texture cleanup (hand/skin/clothing fill via per-part masks).
    All settings are tunable via YAML or CLI override."""
    opt.albedo_path = f'logs/output/{DIR}/{FID}_clothed_smplx_albedo.png'
    opt.albedo_backup_path = f'logs/output/{DIR}/{FID}_clothed_smplx_albedo_backup.png'
    opt.part_seg_dir = 'data/smplx_uv/part_seg'
    # Default fix sets: union of replace_hand_text2.py and replace_hand_text3.py.
    # Part indices: 1,2 = head/face   4,5 = neck/shoulders   15 = body   16-19 = hands.
    if not hasattr(opt, 'tex_white_parts') or opt.tex_white_parts is None:
        opt.tex_white_parts = [1, 2, 4, 5, 15, 16, 17, 18, 19]
    if not hasattr(opt, 'tex_black_parts') or opt.tex_black_parts is None:
        opt.tex_black_parts = [15, 16, 17, 18, 19]
    if not hasattr(opt, 'tex_white_threshold') or opt.tex_white_threshold is None:
        opt.tex_white_threshold = 100
    if not hasattr(opt, 'tex_black_threshold') or opt.tex_black_threshold is None:
        opt.tex_black_threshold = 60
    return opt.albedo_path


def run_phase4(opt):
    """Phase 4 — texture cleanup. Backs up the post-inpaint albedo on first run, then
    fills washed-out (white) and unset (black) pixels per SMPLX body part with the
    region's mean color. Always re-runs (no skip-if-exists). Modifies the albedo
    PNG in place; phase 5 (mesh_to_gs) reads the cleaned-up version.
    """
    import shutil

    if not os.path.exists(opt.albedo_path):
        print(f"[phase4] skipping: {opt.albedo_path} not found (run phase 2 first)")
        return None
    if not os.path.isdir(opt.part_seg_dir):
        print(f"[phase4] skipping: part-seg dir {opt.part_seg_dir} missing")
        return None

    # First-ever run preserves the pristine post-inpaint albedo. Subsequent runs leave
    # the backup alone — re-runs operate on the already-modified albedo (idempotent
    # for white/black-fill since extremal pixels are already filled in). To force a
    # clean restart, delete the _backup.png manually.
    if not os.path.exists(opt.albedo_backup_path):
        shutil.copyfile(opt.albedo_path, opt.albedo_backup_path)
        print(f"[phase4] backed up albedo to {opt.albedo_backup_path}")

    print(f"[phase4] white-fill (threshold={opt.tex_white_threshold}) on parts {list(opt.tex_white_parts)}")
    for part_idx in opt.tex_white_parts:
        mask_path = os.path.join(opt.part_seg_dir, f'{part_idx:02d}.png')
        n = _replace_pixels_in_mask(opt.albedo_path, mask_path, opt.albedo_path,
                                    threshold=opt.tex_white_threshold, mode='white')
        if n is not False:
            print(f"  part {part_idx:02d}: replaced {n} white pixels")

    print(f"[phase4] black-fill (threshold={opt.tex_black_threshold}) on parts {list(opt.tex_black_parts)}")
    for part_idx in opt.tex_black_parts:
        mask_path = os.path.join(opt.part_seg_dir, f'{part_idx:02d}.png')
        n = _replace_pixels_in_mask(opt.albedo_path, mask_path, opt.albedo_path,
                                    threshold=opt.tex_black_threshold, mode='black')
        if n is not False:
            print(f"  part {part_idx:02d}: replaced {n} black pixels")

    return opt.albedo_path


def run_phase5(clothed_obj):
    """Phase 5 — mesh → Gaussian splats. Requires a vclass.npy sibling from phase 3."""
    from lib.ops.mesh_to_gs import obj_mesh_to_ply_mesh, compute_face_labels

    ply_path = clothed_obj.replace('.obj', '.ply')
    obj_mesh_to_ply_mesh(clothed_obj, ply_path)

    vclass_path = clothed_obj.replace('.obj', '_vclass.npy')
    if not os.path.exists(vclass_path):
        print(f"[phase5] skipping GS conversion: {vclass_path} not found (run phase 3 first)")
        return None

    vclass = np.load(vclass_path)
    mesh = Mesh.load(clothed_obj)
    faces = mesh.f.detach().cpu().numpy()
    vclass_label = torch.tensor(vclass).argmax(dim=1).int().detach().cpu().numpy()
    face_mask_vote = compute_face_labels(faces, vclass_label)
    mesh_to_ply_w_labels(ply_path, ply_path, face_mask_vote)
    return ply_path


if __name__ == "__main__":
    import argparse
    import copy
    from omegaconf import OmegaConf
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="path to the main dir (under configs/ and data/)")
    parser.add_argument("--phases", default="1,2,3,4,5",
                        help="comma-separated phases: 1=scale_back, 2=inpaint, 3=mesh_seg, 4=tex_cleanup, 5=mesh_to_gs")
    parser.add_argument("--seg_contract", default='configs/segmentation.yaml',
                        help="yaml defining the segmentation model + label space (see lib/seg_config.py)")
    args, extras = parser.parse_known_args()

    if args.seg_contract and os.path.exists(args.seg_contract):
        seg_config.set_active(args.seg_contract)

    DIR = args.dir
    phases = {int(p) for p in args.phases.split(',') if p.strip()}
    cli_overrides = OmegaConf.from_cli(extras)

    for yaml_file in tqdm.tqdm(sorted(glob.glob(f'configs/{DIR}/*.yaml'))):
        print(f'\n=== {yaml_file} ===')
        FID = os.path.splitext(os.path.basename(yaml_file))[0]
        base_opt = OmegaConf.merge(OmegaConf.load(yaml_file), cli_overrides)

        if 1 in phases:
            opt1 = copy.deepcopy(base_opt)
            _phase1_configure(opt1)
            run_phase1(opt1)

        clothed_obj = None
        if 2 in phases:
            opt2 = copy.deepcopy(base_opt)
            orig_mesh, remapped_mesh = _phase2_configure(opt2, DIR, FID)
            clothed_obj = run_phase2(opt2, orig_mesh, remapped_mesh)

        if 3 in phases:
            opt3 = copy.deepcopy(base_opt)
            input_svd = _phase3_configure(opt3, DIR, FID)
            run_phase3(opt3, input_svd)

        if 4 in phases:
            opt4 = copy.deepcopy(base_opt)
            _phase4_configure(opt4, DIR, FID)
            run_phase4(opt4)

        if 5 in phases:
            if clothed_obj is None:
                # derive from convention when phase 2 was skipped
                opt5 = copy.deepcopy(base_opt)
                opt5.save_path = f"output/{DIR}/{FID}_clothed_smplx"
                clothed_obj = os.path.join(opt5.outdir, opt5.save_path + '.' + opt5.mesh_format)
            run_phase5(clothed_obj)
