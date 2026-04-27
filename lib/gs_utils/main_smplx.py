import os
import cv2
import glob
import time
import tqdm
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

import torch
import torch.nn.functional as F

import rembg

from utils.cam_utils import orbit_camera, OrbitCamera
from lib.gs_utils.gs_renderer_4d_canonical_single import Renderer, MiniCam

from utils.grid_put import mipmap_linear_grid_put_2d
from lib.mesh_utils.mesh import safe_normalize
from lib.gs_utils.gaussian_model_4d_canonical_single import load_from_smplx_obj, load_from_smplx_path
from lib import seg_config
import copy

import lpips
from utils.loss_utils import l1_loss, ssim
USE_LPIPS = False

MODEL_NAME = 'disco4d_single'

def get_mesh_dict(smplx_fp, pose_type=None):

    if smplx_fp.endswith(('.obj')):
        verts, faces, face_normals = load_from_smplx_obj(smplx_fp)
    elif smplx_fp.endswith(('.npz')):
        verts, faces, face_normals = load_from_smplx_path(smplx_fp, pose_type=pose_type)
    mesh_dict = {
        'mesh_verts': verts,
        'mesh_norms': face_normals,
    }

    return mesh_dict

def get_cano_mesh(smplx_path, betas=None, pose_type=None):
    if smplx_path.endswith(('.obj')):
        verts, faces, face_normals = load_from_smplx_obj(smplx_path)
    elif smplx_path.endswith(('.npz')):
        verts, faces, face_normals = load_from_smplx_path(smplx_path, pose_type)
    cano_mesh = {
        'mesh_verts': verts,
        'mesh_norms': face_normals,
    }
    return cano_mesh

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        # self.seed = "random"
        self.seed = 888

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None


        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False


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

        self.input_img_list = None
        self.input_mask_list = None
        self.input_img_torch_list = None
        self.input_mask_torch_list = None

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None: # True
            self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs

            self.posed_meshes = []

        self.posed_mesh = get_mesh_dict(self.opt.load_mesh)
        self.posed_meshes.append(self.posed_mesh)


        if self.opt.load_mesh is not None:
            if opt.use_pretrained:
                self.renderer.load_from_pretrained(self.opt.smplx_gaussians, self.opt.load_mesh, self.opt.lgm_ply, reinitialize=True)            
            else:
                self.renderer.initialize_from_smplx_gaussians(self.opt.smplx_gaussians, self.opt.load_mesh, self.opt.lgm_ply, reinitialize=True)            
                cano_mesh = get_cano_mesh(self.opt.load_mesh)
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = cano_mesh)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        self.seed_everything()

        if USE_LPIPS:
            self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))


    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)

        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
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

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != "" # False
        self.enable_zero123 = self.opt.lambda_zero123 > 0 # True
        self.enable_svd = self.opt.lambda_svd > 0 and self.input_img is not None # False

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd: # False
            if self.opt.mvdream: # False
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123: # True
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils_video import Zero123
            self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max])
            print(f"[INFO] loaded zero123!")


        if self.guidance_svd is None and self.enable_svd: # False
            print(f"[INFO] loading SVD...")
            from guidance.svd_utils import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        if self.input_img_list is not None:
            self.input_img_torch_list = [torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device) for input_img in self.input_img_list]
            self.input_img_torch_list = [F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False) for input_img_torch in self.input_img_torch_list]
            
            self.input_mask_torch_list = [torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device) for input_mask in self.input_mask_list]
            self.input_mask_torch_list = [F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False) for input_mask_torch in self.input_mask_torch_list]
        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = [], []
                for _ in range(self.opt.n_views):
                    for input_img_torch in self.input_img_torch_list:
                        c, v = self.guidance_zero123.get_img_embeds(input_img_torch)
                        c_list.append(c)
                        v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
            
            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps): # 1

            # if self.step == (1000-1):
            #     self.renderer.gaussians.reinitialize_positions()
            
            self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_mesh)

            self.step += 1 # self.step starts from 0
            step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0
        
            ### known view
            for b_idx in range(self.opt.batch_size): # 14
                cur_cam = copy.deepcopy(self.fixed_cam)

                cur_cam.time = b_idx
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[b_idx])

                out = self.renderer.render(cur_cam)
                
                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img_torch_list[b_idx]) / self.opt.batch_size

                if USE_LPIPS:
                    gt_image = self.input_img_torch_list[b_idx]
                    lambda_dssim = 1
                    loss = loss + lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss = loss + self.loss_fn_vgg(image, gt_image).reshape(-1).squeeze()
                    
            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.n_views):
                for b_idx in range(self.opt.batch_size):

                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    poses.append(pose)

                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, time=b_idx)
                    self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[b_idx])

                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color)

                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)

                    # enable mvdream training
                    if self.opt.mvdream: # False
                        for view_i in range(1, 4):
                            pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                            poses.append(pose_i)

                            cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                            # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                            out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                            image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                            images.append(image)


            images = torch.cat(images, dim=0)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio) / (self.opt.batch_size * self.opt.n_views)

            if self.enable_svd:
                loss = loss + self.opt.lambda_svd * self.guidance_svd.train_step(images, step_ratio)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    
    def load_input(self, file):
        self.input_img_list, self.input_mask_list = [], []
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
            cv2.imwrite(file.replace('.png', '_rgba.png'), img) 
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        input_mask = img[..., 3:]
        # white bg
        input_img = img[..., :3] * input_mask + (1 - input_mask)
        # bgr to rgb
        input_img = input_img[..., ::-1].copy()
        self.input_img_list.append(input_img)
        self.input_mask_list.append(input_mask)


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, t=0):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = f'logs/{self.opt.save_path}_mesh_{t:03d}.ply'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = f'logs/{self.opt.save_path}_mesh_{t:03d}.obj'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t)

            # perform texture extraction
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
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    time=t
                )
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[t])
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
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

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)
        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_4d_model.ply')
            # self.renderer.gaussians.save_ply(path)
            self.renderer.gaussians.save_ply_fidxs(path)

        print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=500, ui=False):
        image_list =[]
        from PIL import Image
        from diffusers.utils import export_to_video, export_to_gif
        interval = 1
        nframes = iters // interval # 250
        hor = 180
        delta_hor = 4 * 360 / nframes
        time = 0
        delta_time = 1
        if self.gui:
            from visergui import ViserViewer
            self.viser_gui = ViserViewer(device="cuda", viewer_port=8080)
        if iters > 0:
            self.prepare_train()
            if self.gui:
                self.viser_gui.set_renderer(self.renderer, self.fixed_cam)
            
            for i in tqdm.trange(iters): # 500
                self.train_step()
                if self.gui:
                    self.viser_gui.update()

        # render eval
        os.makedirs(f'logs/{self.opt.save_path}', exist_ok=True)
        image_list =[]
        nframes = 14 *5
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=time
            )
            with torch.no_grad():
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[time])
                outputs = self.renderer.render(cur_cam)

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size
            hor = (hor+delta_hor) % 360

        export_to_gif(image_list, f'logs/{self.opt.save_path}.gif')

        # render front view
        os.makedirs(f'logs/{self.opt.save_path}', exist_ok=True)
        image_list =[]
        nframes = self.opt.batch_size # *5
        # delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for idx in range(nframes):
            

            for deg in [0, 90, 180, 270]:
                pose = orbit_camera(0, deg, self.opt.radius)
                cur_cam = MiniCam(
                    pose,
                    512,
                    512,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    time=time
                )

                with torch.no_grad():
                    self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[time])
                    outputs = self.renderer.render(cur_cam)

                out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
                out = Image.fromarray(np.uint8(out*255))

                out_path = f'logs/{self.opt.save_path}/{idx}_{deg}.png'
                out.save(out_path)
                if deg == 0:
                    image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size
            print(time)
            print(idx)
            # hor = (hor+delta_hor) % 360

        export_to_gif(image_list, f'logs/{self.opt.save_path}_1view.gif')

        # save
        self.save_model(mode='model')
        self.renderer.gaussians.save_deformation(self.opt.outdir, self.opt.save_path)


        if self.gui:
            while True:
                self.viser_gui.update()

# =============================================================================
# PLY handling helpers (ported from compile_gaussians)
# =============================================================================

def _construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape, objects_dc_shape):
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


def _construct_list_of_attributes_fidxs(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape, objects_dc_shape):
    return _construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape, objects_dc_shape) + ['sample_fidxs']


class PlyHandler:
    def __init__(self, max_sh_degree, num_objects):
        self.max_sh_degree = max_sh_degree
        self.num_objects = num_objects

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        for i in range(3):
            features_dc[:, i, 0] = np.asarray(plydata.elements[0][f"f_dc_{i}"])

        extra_f_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
                               key=lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
                             key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
                           key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        obj_dc_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc")],
                              key=lambda x: int(x.split('_')[-1]))
        if len(obj_dc_names) > 0:
            objects_dc = np.zeros((xyz.shape[0], len(obj_dc_names)))
            for idx, attr_name in enumerate(obj_dc_names):
                objects_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            objects_dc = np.zeros((xyz.shape[0], self.num_objects))

        objects_dc = objects_dc[:, :, None]
        return xyz, features_dc, features_extra, opacities, scales, rots, objects_dc

    def concat_ply_data(self, data1, data2):
        return [np.concatenate((d1, d2), axis=0) for d1, d2 in zip(data1, data2)]

    def save_ply(self, path, xyz, features_dc, features_extra, opacities, scales, rots, objects_dc):
        normals = np.zeros_like(xyz)
        dtype_full = [(attribute, 'f4') for attribute in _construct_list_of_attributes(features_dc.shape, features_extra.shape, scales.shape, rots.shape, objects_dc.shape)]
        elements = np.zeros(xyz.shape[0], dtype=dtype_full)

        elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
        elements['opacity'] = opacities.flatten()

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            elements[f'f_dc_{i}'] = features_dc.reshape(-1)[i::features_dc.shape[1] * features_dc.shape[2]]
        for i in range(features_extra.shape[1] * features_extra.shape[2]):
            elements[f'f_rest_{i}'] = features_extra.reshape(-1)[i::features_extra.shape[1] * features_extra.shape[2]]
        for i in range(scales.shape[1]):
            elements[f'scale_{i}'] = scales[:, i]
        for i in range(rots.shape[1]):
            elements[f'rot_{i}'] = rots[:, i]
        for i in range(objects_dc.shape[1] * objects_dc.shape[2]):
            elements[f'obj_dc_{i}'] = objects_dc.reshape(-1)[i::objects_dc.shape[1] * objects_dc.shape[2]]

        ply_element = PlyElement.describe(elements, 'vertex')
        PlyData([ply_element], text=True).write(path)

    def save_cleaned_gaussians(self, path, mask, save_path):
        xyz, features_dc, features_extra, opacities, scales, rots, objects_dc = self.load_ply(path)
        self.save_ply(save_path, xyz[mask], features_dc[mask], features_extra[mask], opacities[mask], scales[mask], rots[mask], objects_dc[mask])

    def save_cleaned_gaussians_category(self, path, category, save_path):
        xyz, features_dc, features_extra, opacities, scales, rots, objects_dc = self.load_ply(path)
        cat_labels = np.argmax(objects_dc.squeeze(2), axis=-1)
        mask = np.isin(cat_labels, category)
        self.save_ply(save_path, xyz[mask], features_dc[mask], features_extra[mask], opacities[mask], scales[mask], rots[mask], objects_dc[mask])


def _save_cleaned_gaussians_with_fidxs(gaussians, mask, path):
    """Mirror compile_gaussians.save_cleaned_gaussians: dump masked GS with sample_fidxs."""
    means3D = gaussians._xyz[mask].detach().cpu().numpy()
    opacity = gaussians._opacity[mask].detach().cpu().numpy()
    scales = gaussians._scaling[mask].detach().cpu().numpy()
    rotations = gaussians._rotation[mask].detach().cpu().numpy()
    features_dc = gaussians._features_dc[mask].detach().cpu().numpy()
    features_rest = gaussians._features_rest[mask].detach().cpu().numpy()
    sh_objs = gaussians._objects_dc[mask].detach().cpu().numpy()
    sample_fidxs = gaussians.sample_fidxs[mask].detach().cpu().numpy()

    normals = np.zeros_like(means3D)
    dtype_full = [(attribute, 'f4') for attribute in _construct_list_of_attributes_fidxs(features_dc.shape, features_rest.shape, scales.shape, rotations.shape, sh_objs.shape)]
    elements = np.zeros(means3D.shape[0], dtype=dtype_full)

    elements['x'], elements['y'], elements['z'] = means3D[:, 0], means3D[:, 1], means3D[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['opacity'] = opacity.flatten()
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        elements[f'f_dc_{i}'] = features_dc.reshape(-1)[i::features_dc.shape[1] * features_dc.shape[2]]
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        elements[f'f_rest_{i}'] = features_rest.reshape(-1)[i::features_rest.shape[1] * features_rest.shape[2]]
    for i in range(scales.shape[1]):
        elements[f'scale_{i}'] = scales[:, i]
    for i in range(rotations.shape[1]):
        elements[f'rot_{i}'] = rotations[:, i]
    for i in range(sh_objs.shape[1] * sh_objs.shape[2]):
        elements[f'obj_dc_{i}'] = sh_objs.reshape(-1)[i::sh_objs.shape[1] * sh_objs.shape[2]]
    elements['sample_fidxs'] = sample_fidxs.flatten()

    PlyData([PlyElement.describe(elements, 'vertex')], text=True).write(path)


def _assign_mask_to_points(seg_mask, points):
    """Sample seg labels at projected GS points (clipping out-of-frame to None)."""
    points = points.astype(int)
    height, width = seg_mask.shape
    labels = []
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            labels.append(seg_mask[y, x])
        else:
            labels.append(None)
    return np.array(labels)


def _clean_mask_pass1(labels, objects_dc):
    """compile_gaussians.clean_mask: keeps clothing+hair categories, suppresses limb mislabels."""
    mask1 = np.isin(labels, [4, 5, 6, 9, 10])
    mask2 = np.isin(objects_dc, [2])
    mask3 = np.isin(labels, [12, 13, 14, 15])
    new_condition_mask = ~(mask3 & mask2)
    return (mask1 | mask2) & new_condition_mask, [2, 4, 6, 9, 10]


def _clean_mask_pass2(labels, objects_dc):
    """compile_gaussians_svd.clean_mask2: a tighter filter run on the already-trimmed PLY."""
    mask = np.isin(objects_dc, [2, 4, 5, 6, 7, 9, 10])

    mask5 = ~np.isin(labels, [4])
    mask6 = np.isin(objects_dc, [4])
    nc3 = ~(mask5 & mask6)

    mask1 = np.isin(labels, [9, 10])
    mask2 = np.isin(objects_dc, [4, 5, 6, 7])
    nc1 = ~(mask1 & mask2)

    mask3 = np.isin(labels, [4, 5, 6, 7, 9, 10])
    mask4 = np.isin(objects_dc, [2])
    nc2 = ~(mask3 & mask4)

    return mask & nc1 & nc2 & nc3, [2, 4, 5, 6, 7, 9, 10]


# =============================================================================
# Render-canonical helpers (ported from render_canonical)
# =============================================================================

def _save_images_to_gif(image_list, output_path, duration=500, loop=0):
    if not image_list:
        return
    image_list[0].save(output_path, save_all=True, append_images=image_list[1:], duration=duration, loop=loop, format='GIF')


def _create_cam(hor, opt):
    pose = orbit_camera(opt.elevation, hor, opt.radius)
    cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
    return MiniCam(pose, opt.ref_size, opt.ref_size, cam.fovy, cam.fovx, cam.near, cam.far)


def _render_360(renderer, opt, save_name, render_type='combined', cat=None):
    image_list = []
    for deg in range(0, 360, 10):
        cam = _create_cam(deg, opt)
        if render_type == 'combined':
            out = renderer.render(cam)
        elif render_type == 'smplx':
            out = renderer.render_smplx(cam)
        elif render_type == 'learnable':
            out = renderer.render_learnable(cam)
        elif render_type == 'cat':
            out = renderer.render_learnable_categorical(cam, category=cat)
        else:
            raise ValueError(f'unknown render_type {render_type}')
        image_np = (out['image'].transpose(0, 2).transpose(0, 1).cpu().detach().numpy() * 255).astype('uint8')
        image_list.append(Image.fromarray(image_np))
    _save_images_to_gif(image_list, save_name)


def _render_90(renderer, opt, save_dir):
    for deg in range(0, 360, 90):
        cam = _create_cam(deg, opt)
        out = renderer.render(cam)
        image_np = (out['image'].transpose(0, 2).transpose(0, 1).cpu().detach().numpy() * 255).astype('uint8')
        Image.fromarray(image_np).save(os.path.join(save_dir, f'0_{deg}.png'))


# =============================================================================
# Path conventions (no _square_rgba; flexible FID = filename stem)
# =============================================================================

def _paths(args_dir, FID):
    """Canonical artefact paths derived from {DIR, FID}. Stage outputs are derived
    by appending suffixes to these stems."""
    out = f'logs/output/{args_dir}'
    return {
        'out_dir': out,
        'smplx_gaussians': f'{out}/{FID}_clothed_smplx.ply',
        'load_mesh': f'{out}/{FID}_smplx.npz',
        'lgm_ply': f'{out}/{FID}_seg_model_dens4.ply',
        'refined': f'{out}/{FID}_refined.ply',
        'refined_cat': f'{out}/{FID}_refined_cat.ply',
        'refined_cat_seg_model': f'{out}/{FID}_refined_cat_seg_model.ply',
        'cloth_model': f'{out}/{FID}_refined_cat_cloth_model.ply',
        'cloth_model_trimmed': f'{out}/{FID}_refined_cat_cloth_model_trimmed.ply',
        'cloth_model_trimmed2': f'{out}/{FID}_refined_cat_cloth_model_trimmed2.ply',
        'svd_seg_first_frame': f'data/{args_dir}/svd/{FID}/seg/0000_seg.pt',
    }


# =============================================================================
# Stage configures + runners
# =============================================================================

def _phase1_configure(opt, args_dir, FID, input_path):
    """Stage 1 — refine single gaussians (image + zero123 guidance)."""
    P = _paths(args_dir, FID)
    opt.input = input_path
    opt.fid = FID
    opt.smplx_gaussians = P['smplx_gaussians']
    opt.load_mesh = P['load_mesh']
    opt.lgm_ply = P['lgm_ply']
    opt.use_pretrained = False
    opt.save_path = f'{args_dir}/{MODEL_NAME}/{FID}'
    return P


def run_phase1(opt):
    gui = GUI(opt)
    gui.train(opt.iters)
    del gui
    torch.cuda.empty_cache()


def _phase2_configure(opt, args_dir, FID, input_path):
    """Stage 2 — load trained deformation + dump _refined.ply, then concat with
    SMPLX gaussians into _refined_cat.ply (input for the seg stage)."""
    P = _paths(args_dir, FID)
    opt.input = input_path
    opt.fid = FID
    opt.smplx_gaussians = P['smplx_gaussians']
    opt.load_mesh = P['load_mesh']
    opt.lgm_ply = P['lgm_ply']
    opt.use_pretrained = False
    opt.save_path = f'{args_dir}/{MODEL_NAME}/{FID}'
    opt.ref_size = 1024
    opt.W = 1024
    opt.H = 1024
    return P


def run_phase2(opt, paths):
    if not os.path.exists(opt.smplx_gaussians):
        raise FileNotFoundError(f'[phase2] missing SMPLX GS at {opt.smplx_gaussians}')
    deformation_pth = os.path.join('logs', opt.save_path + '_deformation.pth')
    if not os.path.exists(deformation_pth):
        raise FileNotFoundError(f'[phase2] no stage-1 checkpoint at {deformation_pth}')

    gui = GUI(opt)
    gui.renderer.gaussians.load_model('logs', opt.save_path)
    os.makedirs(paths['out_dir'], exist_ok=True)
    gui.renderer.gaussians.save_ply_deformation_fidxs(paths['refined'])
    del gui
    torch.cuda.empty_cache()

    ply_handler = PlyHandler(max_sh_degree=0, num_objects=seg_config.get().num_classes)
    data1 = ply_handler.load_ply(opt.smplx_gaussians)
    data2 = ply_handler.load_ply(paths['refined'])
    combined = ply_handler.concat_ply_data(data1, data2)
    ply_handler.save_ply(paths['refined_cat'], *combined)
    print(f'[phase2] wrote {paths["refined_cat"]}')


def _phase3_configure(seg_opt, args_dir, FID):
    """Stage 3 — train segmentation GS on _refined_cat.ply via SegFormer pseudo-GT.
    Uses lib.gs_utils.main_seg_batch.GUI (basic 3DGS renderer) on the concat'd PLY."""
    P = _paths(args_dir, FID)
    seg_opt.save_path = f'output/{args_dir}/{FID}_refined_cat_seg'
    seg_opt.load = P['refined_cat']
    # Pin to seg-stage default (CLI overrides for the refine stage shouldn't
    # bleed into seg training, which converges in tens of iters).
    seg_opt.iters = 20
    return P


def run_phase3(seg_opt, paths):
    if not os.path.exists(seg_opt.load):
        raise FileNotFoundError(f'[phase3] missing input PLY at {seg_opt.load} (run phase 2 first)')
    from lib.gs_utils.main_seg_batch import GUI as SegGUI
    gui = SegGUI(seg_opt)
    if seg_opt.gui:
        gui.render()
    else:
        gui.train(seg_opt.iters)
    del gui
    torch.cuda.empty_cache()
    print(f'[phase3] wrote {paths["refined_cat_seg_model"]}')


def _trim_pass(opt, in_ply, out_ply, seg_file, clean_fn):
    """Project learnable GS verts to image coords, sample SVD seg labels, clean by category."""
    renderer = Renderer(sh_degree=opt.sh_degree)
    renderer.initialize_from_smplx_gaussians(opt.smplx_gaussians, opt.load_mesh, lgm_ply=in_ply)

    seg_tensor = torch.load(seg_file)
    verts = renderer.gaussians.get_learnable_xyz.detach().cpu().numpy()
    obj_dc = renderer.gaussians.get_learnable_objects.detach().cpu().numpy()
    objects_dc = np.argmax(obj_dc.squeeze(1), axis=-1)

    cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
    pts_cam = (cam.view @ np.hstack([verts, np.ones((len(verts), 1))]).T).T
    pts_proj = (cam.perspective @ pts_cam.T).T
    pts_cart = pts_proj[:, :3] / pts_proj[:, 3:4]
    pts_img = np.zeros_like(pts_cart)
    pts_img[:, 0] = (0.5 + pts_cart[:, 0] / 2) * cam.W
    pts_img[:, 1] = (0.5 - pts_cart[:, 1] / 2) * cam.H
    pts_img[:, 2] = pts_cart[:, 2]
    points = pts_img[:, :2].copy()
    points[:, 1] = -points[:, 1] + cam.H

    seg_mask = seg_tensor.unsqueeze(0).unsqueeze(0).float()
    seg_mask = F.interpolate(seg_mask, size=(opt.H, opt.W), mode='bilinear', align_corners=False).squeeze().long()
    labels = _assign_mask_to_points(seg_mask, points)

    mask, _ = clean_fn(labels, objects_dc)

    _save_cleaned_gaussians_with_fidxs(renderer.gaussians, mask, out_ply)
    PlyHandler(max_sh_degree=0, num_objects=seg_config.get().num_classes).save_cleaned_gaussians(in_ply, mask, out_ply)
    del renderer
    torch.cuda.empty_cache()


def _phase4_configure(opt, args_dir, FID):
    """Stage 4 — clean: filter by clothing categories, then per-pixel trim using SVD seg."""
    P = _paths(args_dir, FID)
    opt.fid = FID
    opt.smplx_gaussians = P['smplx_gaussians']
    opt.load_mesh = P['load_mesh']
    opt.W = 1024
    opt.H = 1024
    opt.ref_size = 1024
    opt.resolution = 1024
    return P


def run_phase4(opt, paths):
    if not os.path.exists(paths['refined_cat_seg_model']):
        raise FileNotFoundError(f'[phase4] missing seg PLY at {paths["refined_cat_seg_model"]} (run phase 3 first)')

    ph = PlyHandler(max_sh_degree=0, num_objects=seg_config.get().num_classes)
    ph.save_cleaned_gaussians_category(paths['refined_cat_seg_model'], seg_config.get().nonskin_categories, paths['cloth_model'])
    print(f'[phase4] category-filtered -> {paths["cloth_model"]}')

    seg_file = paths['svd_seg_first_frame']
    if not os.path.exists(seg_file):
        print(f'[phase4] skipping per-pixel trim: SVD seg not found at {seg_file}')
        return

    _trim_pass(opt, in_ply=paths['cloth_model'],
               out_ply=paths['cloth_model_trimmed'], seg_file=seg_file, clean_fn=_clean_mask_pass1)
    print(f'[phase4] trim pass 1 -> {paths["cloth_model_trimmed"]}')

    _trim_pass(opt, in_ply=paths['cloth_model_trimmed'],
               out_ply=paths['cloth_model_trimmed2'], seg_file=seg_file, clean_fn=_clean_mask_pass2)
    print(f'[phase4] trim pass 2 -> {paths["cloth_model_trimmed2"]}')


def _phase5_configure(opt, args_dir, FID):
    """Stage 5 — render canonical / 360 GIFs of the cleaned GS."""
    P = _paths(args_dir, FID)
    opt.fid = FID
    opt.smplx_gaussians = P['smplx_gaussians']
    opt.load_mesh = P['load_mesh']
    opt.lgm_ply = P['cloth_model_trimmed2'] if os.path.exists(P['cloth_model_trimmed2']) else P['cloth_model']
    opt.radius = 1.5
    opt.ref_size = 1024
    opt.W = 1024
    opt.H = 1024
    return P


def run_phase5(opt, args_dir, FID, paths):
    if not os.path.exists(opt.lgm_ply):
        raise FileNotFoundError(f'[phase5] missing cleaned PLY at {opt.lgm_ply} (run phase 4 first)')
    from lib.gs_utils.gs_renderer_canonical_edit import Renderer as RendererEdit

    save_dir = f'data/assets/{args_dir}/{FID}'
    os.makedirs(save_dir, exist_ok=True)
    out_img_dir = f'logs/{args_dir}/disco4d_single_refined/{FID}'
    os.makedirs(out_img_dir, exist_ok=True)

    renderer = RendererEdit(sh_degree=opt.sh_degree)
    renderer.initialize_from_smplx_gaussians(opt.smplx_gaussians, opt.load_mesh, lgm_ply=opt.lgm_ply)

    _render_360(renderer, opt, f'{save_dir}/orig_combined.gif', render_type='combined')
    _render_90(renderer, opt, out_img_dir)

    cano_mesh = _get_cano_mesh_da_pose(opt.load_mesh)
    renderer.gaussians.update_to_posed_mesh(posed_mesh=cano_mesh)
    _render_360(renderer, opt, f'{save_dir}/canonical_combined.gif', render_type='combined')
    _render_360(renderer, opt, f'{save_dir}/canonical_learnable.gif', render_type='learnable')
    _render_360(renderer, opt, f'{save_dir}/canonical_smplx.gif', render_type='smplx')

    for tag, cat in seg_config.get().tag_categories.items():
        _render_360(renderer, opt, f'{save_dir}/canonical_learnable_cat_{tag}.gif', render_type='cat', cat=cat)

    del renderer
    torch.cuda.empty_cache()
    print(f'[phase5] wrote renders to {save_dir}')


def _get_cano_mesh_da_pose(smplx_path):
    verts, faces, face_normals = load_from_smplx_path(smplx_path, 'da-pose')
    return {'mesh_verts': verts, 'mesh_norms': face_normals}


# =============================================================================
# Orchestrator
# =============================================================================

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/4d.yaml', help="path to the main yaml config")
    parser.add_argument("--seg_config", default='configs/image_seg.yaml', help="yaml config for the seg stage")
    parser.add_argument("--seg_contract", default='configs/segmentation.yaml',
                        help="yaml defining the segmentation model + label space (see lib/seg_config.py)")
    parser.add_argument("--dir", required=True, help="case dir under data/ and logs/output/")
    parser.add_argument("--phases", default="1,2,3,4,5",
                        help="comma-separated phases: 1=refine, 2=compile, 3=segment, 4=clean, 5=render")
    args, extras = parser.parse_known_args()

    if args.seg_contract and os.path.exists(args.seg_contract):
        seg_config.set_active(args.seg_contract)

    args_dir = args.dir
    phases = {int(p) for p in args.phases.split(',') if p.strip()}

    # Pipeline defaults. CLI extras override these; they in turn override the YAML.
    pipeline_defaults = OmegaConf.create({
        'elevation': 0,
        'force_cuda_rast': True,
        'gui': False,
        'n_views': 8,
        'iters': 1000,
        'use_pretrained': False,
        'batch_size': 1,
        'ref_size': 1024,
    })
    cli_overrides = OmegaConf.from_cli(extras)

    base_opt = OmegaConf.merge(OmegaConf.load(args.config), pipeline_defaults, cli_overrides)
    base_seg_opt = OmegaConf.merge(OmegaConf.load(args.seg_config), pipeline_defaults, cli_overrides) if 3 in phases else None

    input_dir = f'data/{args_dir}/img'
    img_paths = sorted(p for p in os.listdir(input_dir) if p.endswith('.png'))

    for impath in img_paths:
        FID = os.path.splitext(impath)[0]
        input_path = f'{input_dir}/{impath}'
        print(f'\n=== {FID} ===')

        if 1 in phases:
            opt1 = copy.deepcopy(base_opt)
            _phase1_configure(opt1, args_dir, FID, input_path)
            run_phase1(opt1)

        if 2 in phases:
            opt2 = copy.deepcopy(base_opt)
            paths2 = _phase2_configure(opt2, args_dir, FID, input_path)
            run_phase2(opt2, paths2)

        if 3 in phases:
            seg_opt = copy.deepcopy(base_seg_opt)
            paths3 = _phase3_configure(seg_opt, args_dir, FID)
            run_phase3(seg_opt, paths3)

        if 4 in phases:
            opt4 = copy.deepcopy(base_opt)
            paths4 = _phase4_configure(opt4, args_dir, FID)
            run_phase4(opt4, paths4)

        if 5 in phases:
            opt5 = copy.deepcopy(base_opt)
            paths5 = _phase5_configure(opt5, args_dir, FID)
            run_phase5(opt5, args_dir, FID, paths5)
