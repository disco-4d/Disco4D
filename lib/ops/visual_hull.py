"""Visual hull + batch mesh update.

Produces ``logs/output/{case_dir}/{FID}_seg_model_dens4.ply``,
consumed downstream by ``lib/gs_utils/main_smplx.py``.

For every yaml under ``configs/{case_dir}/`` this runs, in order:

  1. Per-FID segmentation of the LGM Gaussian splat (``GUI.train``).
  2. SDF-offset the LGM mesh off the SMPL-X body and densify Gaussians
     into the newly-opened shell (``update_with_FID``).

Self-contained: nothing here imports from sibling pipeline scripts.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
import trimesh
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
from pysdf import SDF
from scipy.spatial import KDTree, cKDTree
from transformers import (
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Bare imports (`gs_renderer`, `mesh`) inside the renderer/mesh modules expect
# their own directories on sys.path.
for _p in (REPO_ROOT, REPO_ROOT / "lib" / "gs_utils", REPO_ROOT / "lib" / "mesh_utils"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from utils.cam_utils import orbit_camera, OrbitCamera  # noqa: E402
from utils.grid_put import mipmap_linear_grid_put_2d  # noqa: E402
from gs_renderer import Renderer, MiniCam  # noqa: E402
from mesh import Mesh, safe_normalize  # noqa: E402
from lib import seg_config  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1 — segmentation training of the LGM Gaussian splat.
# ---------------------------------------------------------------------------


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

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
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.opt.use_seg:
            seg_cfg = seg_config.get()
            self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.num_classes = seg_cfg.num_classes

            self.transform = T.ToPILImage()
            self.seg_processor = SegformerImageProcessor.from_pretrained(seg_cfg.model_name)
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_cfg.model_name)

            self.cat_map = dict(seg_cfg.cat_map)
            self.input_seg_torch = None

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except Exception:
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

        self.renderer.gaussians.training_setup(self.opt)
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            print(f"[INFO] loaded zero123!")

        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        with torch.no_grad():
            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = [], []
                c, v = self.guidance_zero123.get_img_embeds(self.input_img_torch)
                for _ in range(self.opt.batch_size):
                    c_list.append(c)
                    v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]

    def get_seg_inference(self, pred_image):
        image = self.transform(pred_image)
        inputs = self.seg_processor(images=image, return_tensors="pt")

        outputs = self.seg_model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        pseudo_gt_seg = pred_seg.clone()
        for k, v in self.cat_map.items():
            pseudo_gt_seg = torch.where(pred_seg == k, v, pseudo_gt_seg)

        return pseudo_gt_seg

    @staticmethod
    def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
        """Top-k neighborhood KL consistency loss for a 3D point cloud."""
        if features.size(0) > max_points:
            indices = torch.randperm(features.size(0))[:max_points]
            features = features[indices]
            predictions = predictions[indices]

        indices = torch.randperm(features.size(0))[:sample_size]
        sample_features = features[indices]
        sample_preds = predictions[indices]

        dists = torch.cdist(sample_features, features)
        _, neighbor_indices_tensor = dists.topk(k, largest=False)

        neighbor_preds = predictions[neighbor_indices_tensor]

        kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
        loss = kl.sum(dim=-1).mean()

        num_classes = predictions.size(1)
        normalized_loss = loss / num_classes

        return lambda_val * normalized_loss

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            cur_cam = self.fixed_cam
            out = self.renderer.render(cur_cam)
            if self.input_seg_torch is None:
                self.input_seg_torch = self.get_seg_inference(out["image"])

            if self.opt.use_seg and self.input_seg_torch is not None:
                pred_seg = out["segmentation"].unsqueeze(0)

                loss_seg = self.cls_criterion(pred_seg, self.input_seg_torch.unsqueeze(0).cuda()).squeeze(0).mean()
                loss = 1000 * step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))

            render_resolution = 1024
            poses = []
            vers, hors, radii = [], [], []
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = np.random.uniform(-0.5, 0.5)

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                out = self.renderer.render(cur_cam)

                if self.opt.use_seg:
                    pred_seg = out["segmentation"].unsqueeze(0)

                    pseudo_gt_seg = self.get_seg_inference(out["image"])

                    loss_seg = self.cls_criterion(pred_seg, pseudo_gt_seg.unsqueeze(0).cuda()).squeeze(0).mean()
                    loss += 1000 * step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))

                    if self.step % 5 == 0:
                        reg3d_k = 5
                        reg3d_lambda_val = 2
                        reg3d_max_points = 200000
                        reg3d_sample_size = 1000

                        logits3d = self.renderer.gaussians._objects_dc.permute(2, 0, 1)
                        prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1, 0)
                        loss_obj_3d = self.loss_cls_3d(self.renderer.gaussians._xyz.squeeze().detach(), prob_obj3d, reg3d_k, reg3d_lambda_val, reg3d_max_points, reg3d_sample_size)
                        loss += 10000 * step_ratio * loss_obj_3d

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        starter.elapsed_time(ender)

        self.need_update = True

    def load_input(self, file):
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if '_rgba' not in file:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
            file = file.replace('.png', '_rgba.png')
            cv2.imwrite(file, img)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        self.input_img = self.input_img[..., ::-1].copy()

        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )

                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0)

                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f)
                depth = depth.squeeze(0)

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)

                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )

                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def train(self, iters=500):
        if self.gui:
            from visergui import ViserViewer
            self.viser_gui = ViserViewer(device="cuda", viewer_port=8080)
        if iters > 0:
            self.prepare_train()
            if self.gui:
                self.viser_gui.set_renderer(self.renderer, self.fixed_cam)

            for _ in tqdm.trange(iters):
                self.train_step()
                if self.gui:
                    self.viser_gui.update()

            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        self.save_model(mode='model')

        if self.gui:
            while True:
                self.viser_gui.update()


# ---------------------------------------------------------------------------
# Stage 2 — SDF mesh offset & gaussian densification.
# ---------------------------------------------------------------------------


def calculate_sdf_with_pysdf(sdf, points):
    return sdf(points)


def construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape, objects_dc_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(features_dc_shape[1] * features_dc_shape[2]):
        l.append(f'f_dc_{i}')
    for i in range(features_rest_shape[1] * features_rest_shape[2]):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scaling_shape[1]):
        l.append(f'scale_{i}')
    for i in range(rotation_shape[1]):
        l.append(f'rot_{i}')
    for i in range(objects_dc_shape[1] * objects_dc_shape[2]):
        l.append(f'obj_dc_{i}')
    return l


def save_ply(path, xyz, features_dc, features_rest, opacities, scaling, rotation, objects_dc):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc.shape, features_rest.shape, scaling.shape, rotation.shape, objects_dc.shape)]

    elements = np.zeros(xyz.shape[0], dtype=dtype_full)

    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['opacity'] = opacities.flatten()

    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        elements[f'f_dc_{i}'] = features_dc.reshape(-1)[i::features_dc.shape[1] * features_dc.shape[2]]

    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        elements[f'f_rest_{i}'] = features_rest.reshape(-1)[i::features_rest.shape[1] * features_rest.shape[2]]

    for i in range(scaling.shape[1]):
        elements[f'scale_{i}'] = scaling[:, i]

    for i in range(rotation.shape[1]):
        elements[f'rot_{i}'] = rotation[:, i]

    for i in range(objects_dc.shape[1] * objects_dc.shape[2]):
        elements[f'obj_dc_{i}'] = objects_dc.reshape(-1)[i::objects_dc.shape[1] * objects_dc.shape[2]]

    ply_element = PlyElement.describe(elements, 'vertex')
    PlyData([ply_element], text=True).write(path)


def find_nearby_points(xyz, new_points, k=1):
    new_xyz = np.vstack((xyz, np.array(new_points)))
    tree = KDTree(new_xyz)
    _, knn_idxs = tree.query(new_xyz, k=k + 1)
    return new_xyz, knn_idxs[:, 1:]


def load_xyz_from_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata['vertex']['x']),
                    np.asarray(plydata['vertex']['y']),
                    np.asarray(plydata['vertex']['z'])), axis=-1)
    return xyz


def associate_points_to_vertices(mesh_vertices, points):
    tree = cKDTree(mesh_vertices)
    _, indices = tree.query(points, k=1)
    return indices


def offset_associated_points(original_vertices, updated_vertices, points, associated_indices):
    displacements = updated_vertices[associated_indices] - original_vertices[associated_indices]
    offset_points = points + displacements
    return offset_points, displacements


def save_xyz_to_ply(xyz, path, plydata):
    plydata['vertex']['x'] = xyz[:, 0]
    plydata['vertex']['y'] = xyz[:, 1]
    plydata['vertex']['z'] = xyz[:, 2]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plydata.write(path)


def compute_sdf_and_offset_vertices(mesh1, mesh2_vertices):
    """Offset Mesh-2 vertices that fall inside (or close to) Mesh-1's surface."""
    query_points = mesh2_vertices
    sdf_values = mesh1.nearest.signed_distance(query_points)

    updated_vertices = np.copy(mesh2_vertices)
    inside_indices = np.where(sdf_values > -0.002)[0]

    for idx in inside_indices:
        nearest_point, distance, triangle_index = mesh1.nearest.on_surface(query_points[idx].reshape(1, -1))
        surface_normal = mesh1.face_normals[triangle_index[0]]
        offset_vertex = nearest_point + surface_normal * 0.01
        updated_vertices[idx] = offset_vertex

    return updated_vertices, inside_indices


def get_ply(updated_ply_path):
    num_objects = seg_config.get().num_classes
    max_sh_degree = 0

    plydata = PlyData.read(updated_ply_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    objects_dc = np.zeros((xyz.shape[0], num_objects, 1))
    for idx in range(num_objects):
        objects_dc[:, idx, 0] = np.asarray(plydata.elements[0]["obj_dc_" + str(idx)])

    return xyz, opacities, features_dc, features_extra, objects_dc, rots, scales


def update_with_FID(FID, DIR):
    smplx_mesh_path = f'logs/output/{DIR}/{FID}.obj'
    lgm_mesh_path = f'data/{DIR}/lgm/{FID}.obj'
    smplx_mesh = o3d.io.read_triangle_mesh(smplx_mesh_path)
    lgm_mesh = o3d.io.read_triangle_mesh(lgm_mesh_path)

    # Step 1: SDF-offset the LGM mesh off the SMPL-X body.
    mesh1 = trimesh.load_mesh(smplx_mesh_path)
    mesh2_vertices = np.array(lgm_mesh.vertices)

    updated_mesh2_vertices, inside_indices = compute_sdf_and_offset_vertices(mesh1, mesh2_vertices)

    cloned_lgm_mesh = o3d.io.read_triangle_mesh(lgm_mesh_path)
    cloned_lgm_mesh.vertices = o3d.utility.Vector3dVector(updated_mesh2_vertices)
    cloned_path = f'data/{DIR}/lgm/{FID}_cloned.obj'
    o3d.io.write_triangle_mesh(cloned_path, cloned_lgm_mesh)

    # Step 2: densification
    mesh2 = o3d.io.read_triangle_mesh(f'data/{DIR}/lgm/{FID}.obj')
    original_ply_path = f'logs/output/{DIR}/{FID}_seg_model.ply'
    mesh2_updated = o3d.io.read_triangle_mesh(cloned_path)

    mesh2_vertices = np.array(mesh2.vertices)
    mesh2_updated_vertices = np.array(mesh2_updated.vertices)

    x_points = load_xyz_from_ply(original_ply_path)

    nearest_vertex_indices = associate_points_to_vertices(mesh2_vertices, x_points)
    offset_x_points, _ = offset_associated_points(mesh2_vertices, mesh2_updated_vertices, x_points, nearest_vertex_indices)

    updated_ply_path = f'logs/output/{DIR}/{FID}_seg_model_updated.ply'
    plydata = PlyData.read(original_ply_path)
    save_xyz_to_ply(offset_x_points, updated_ply_path, plydata)

    # Densify features from nearest neighbours.
    xyz, opacities, features_dc, features_extra, objects_dc, rots, scales = get_ply(updated_ply_path)

    xs, ys, zs = mesh2_updated_vertices[:, 0], mesh2_updated_vertices[:, 1], mesh2_updated_vertices[:, 2]

    x = np.linspace(xs.min(), xs.max(), int((xs.max() - xs.min()) / 1.5 * 256))
    y = np.linspace(ys.min(), ys.max(), int((ys.max() - ys.min()) / 1.5 * 256))
    z = np.linspace(zs.min(), zs.max(), int((zs.max() - zs.min()) / 1.5 * 256))
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    mesh2_tm = trimesh.load_mesh(smplx_mesh_path)
    mesh2_updated_tm = trimesh.load_mesh(cloned_path)

    mesh2_hull = SDF(mesh2_tm.vertices, mesh2_tm.faces)
    mesh2_updated_hull = SDF(mesh2_updated_tm.vertices, mesh2_updated_tm.faces)

    sdf_mesh2 = calculate_sdf_with_pysdf(mesh2_hull, points)
    sdf_updated_mesh2 = calculate_sdf_with_pysdf(mesh2_updated_hull, points)

    final_points = points[(sdf_mesh2 <= 0) & (sdf_updated_mesh2 >= 0)]
    print(len(final_points))

    smplx_pcd_mesh = o3d.io.read_triangle_mesh(smplx_mesh_path)
    smplx_pcd_mesh.compute_vertex_normals()
    num_pts = 20000
    pcd = smplx_pcd_mesh.sample_points_uniformly(number_of_points=num_pts)
    pcd = smplx_pcd_mesh.sample_points_poisson_disk(number_of_points=num_pts, pcl=pcd)

    query_points = np.vstack((final_points, np.array(pcd.points)))
    print(query_points.shape)

    new_xyz, knn_idxs = find_nearby_points(xyz, query_points, 1)

    original_num_points = xyz.shape[0]
    adjusted_knn_idxs = np.where(knn_idxs < original_num_points, knn_idxs, 0)[:, 0]

    new_opacities = opacities[adjusted_knn_idxs]
    new_features_dc = features_dc[adjusted_knn_idxs]
    new_features_extra = features_extra[adjusted_knn_idxs]
    new_objects_dc = objects_dc[adjusted_knn_idxs]
    new_rots = rots[adjusted_knn_idxs]
    new_scales = scales[adjusted_knn_idxs]

    save_ply(f'logs/output/{DIR}/{FID}_seg_model_dens4.ply', new_xyz, new_features_dc, new_features_extra, new_opacities, new_scales, new_rots, new_objects_dc)


# ---------------------------------------------------------------------------
# Pipeline orchestration.
# ---------------------------------------------------------------------------


def _segment_case(case_dir: str, seg_yaml: str) -> None:
    config_paths = sorted(glob.glob(str(REPO_ROOT / "configs" / case_dir / "*.yaml")))
    for cfg_path in tqdm.tqdm(config_paths):
        opt = OmegaConf.load(seg_yaml)
        FID = Path(cfg_path).stem
        opt.save_path = f"output/{case_dir}/{FID}_seg"
        opt.load = f"data/{case_dir}/lgm/{FID}.ply"
        opt.iters = 20
        os.makedirs(f"logs/output/{case_dir}", exist_ok=True)

        gui = GUI(opt)
        if opt.gui:
            gui.render()
        else:
            gui.train(opt.iters)


def _mesh_case(case_dir: str) -> None:
    config_paths = sorted(glob.glob(str(REPO_ROOT / "configs" / case_dir / "*.yaml")))
    for cfg_path in tqdm.tqdm(config_paths):
        update_with_FID(Path(cfg_path).stem, case_dir)


def run(case_dir: str, seg_yaml: str = "configs/image_seg.yaml") -> None:
    _segment_case(case_dir, seg_yaml)
    _mesh_case(case_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="case directory under data/")
    parser.add_argument("--config", default="configs/image_seg.yaml",
                        help="segmentation yaml config")
    parser.add_argument("--seg_contract", default="configs/segmentation.yaml",
                        help="yaml defining the segmentation model + label space (see lib/seg_config.py)")
    args = parser.parse_args()

    if args.seg_contract and os.path.exists(args.seg_contract):
        seg_config.set_active(args.seg_contract)

    run(args.dir, args.config)
