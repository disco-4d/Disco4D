import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg

from utils.cam_utils import orbit_camera, OrbitCamera, fov2focal 
from lib.gs_utils.gs_renderer_4d_canonical_single import Renderer, load_from_smplx_path, load_from_smplx_obj

from utils.grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
import copy

import lpips
from utils.loss_utils import l1_loss, ssim
import random
import glob

USE_LPIPS = False
USE_NOVEL_VIEW = False
# NUM_FRAMES = 37

def proj_from_intrinsics_opengl(fx, fy, cx, cy, W, H, near=0.01, far=100.0):
    """
    OpenGL-style perspective with w_clip = -z and NDC y-up,
    reproducing: u = fx*x/z + cx, v = fy*y/z + cy (image origin top-left).
    """
    sx =  2.0 * fx / W
    sy = -2.0 * fy / H          # minus for image y-down -> NDC y-up
    ox = 1.0 - 2.0 * (cx / W)
    oy = 2.0 * (cy / H) - 1.0

    A = -(far + near) / (far - near)
    B = -(2.0 * far * near) / (far - near)

    P = np.array([
        [ sx, 0.0,  ox, 0.0],
        [0.0,  sy,  oy, 0.0],
        [0.0, 0.0,   A,   B],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)
    return P


class MiniCamTransforms:
    def __init__(self, c2w, width, height, cam_angle_x, znear, zfar,time=0,  device="cuda"):
        self.image_width  = int(width)
        self.image_height = int(height)
        self.FoVx = float(cam_angle_x)
        self.znear = float(znear)
        self.zfar  = float(zfar)

        # --- ADD THIS: vertical FOV from horizontal FOV ---
        aspect = self.image_width / float(self.image_height)
        self.FoVy = 2.0 * np.arctan(np.tan(self.FoVx * 0.5) / aspect)  # radians

        # Intrinsics (match your overlay math)
        fx = (self.image_width  / 2.0) / np.tan(self.FoVx / 2.0)
        fy = fx * (self.image_width / self.image_height)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0

        c2w[:, 0] *= -1
        w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
        w2c[1:3, :3] *= -1
        w2c[:3, 3]   *= -1
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).t().to(device)

        P = proj_from_intrinsics_opengl(
            fx, fy, cx, cy, self.image_width, self.image_height, self.znear, self.zfar
        )
        self.projection_matrix = torch.tensor(P, dtype=torch.float32).t().to(device)

        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3], dtype=torch.float32).to(device)

        self.time = time


def get_mesh_dict(smplx_fp, pose_type=None):

    if smplx_fp.endswith(('.obj')):
        verts, faces, face_normals = load_from_smplx_obj(smplx_fp)
    else:
        verts, faces, face_normals = load_from_smplx_path(smplx_fp, pose_type=pose_type)
    mesh_dict = {
        'mesh_verts': verts,
        # 'mesh_norms': torch.tensor(obj.vertex_normals),
        'mesh_norms': face_normals,
    }

    return mesh_dict

# SEL_LIST = list(range(14)) 
def get_cano_mesh(smplx_path, betas=None, pose_type=None):
    if smplx_path.endswith(('.obj')):
        verts, faces, face_normals = load_from_smplx_obj(smplx_path)
    else:

        if betas is not None:
            verts, faces, face_normals = load_from_smplx_path_custom(smplx_path, betas=betas)
        else:
            verts, faces, face_normals = load_from_smplx_path(smplx_path, pose_type)
    cano_mesh = {
        'mesh_verts': verts,
        # 'mesh_norms': torch.tensor(obj.vertex_normals),
        'mesh_norms': face_normals,
        # 'mesh_faces':  faces
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
        print(self.opt.input)
        print(os.path.abspath(self.opt.input))
        if self.opt.input is not None: # True
            self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs

            self.posed_meshes = []
        

        self.posed_mesh = get_mesh_dict(self.opt.load_mesh)
        self.posed_meshes.append(self.posed_mesh)

        if self.opt.input_svd is not None:
            self.load_svd_input(self.opt.input_svd)


        if self.opt.load_mesh is not None:
            # lgm_ply = self.opt.lgm_ply if self.opt.lgm_init else None
            if opt.use_pretrained:
                self.renderer.load_from_pretrained(self.opt.smplx_gaussians, self.opt.load_mesh, self.opt.lgm_ply, reinitialize=True)            
            else:
                self.renderer.initialize_from_smplx_gaussians(self.opt.smplx_gaussians, self.opt.load_mesh, self.opt.lgm_ply, reinitialize=True)            
                # cano_mesh = get_cano_mesh(self.opt.load_mesh, pose_type='da-pose')
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
        # cam_angle_x = self.mv_cameras[0].cam_angle_x
        self.fixed_cam = MiniCamTransforms(
            pose,
            1024,
            1024,
            # self.cam.fovy,
            # self.cam.fovx,
            self.cam_angle_x,
            self.cam.near,
            self.cam.far,
            # time=time
        )

        self.enable_sd = False # self.opt.lambda_sd > 0 and self.prompt != "" # False
        self.enable_zero123 = False # self.opt.lambda_zero123 > 0 # True
        self.enable_svd = False # self.opt.lambda_svd > 0 and self.input_img is not None # False

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

            # mv_np_images, self.mv_cameras = self.images_to_video() 
            # self.mv_images, self.mv_masks = self.images_segmentation(mv_np_images)


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
        
            ## known view
            for b_idx in range(self.opt.batch_size): # 14
                cur_cam = copy.deepcopy(self.fixed_cam)

                cur_cam.time = b_idx
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[b_idx])

                out = self.renderer.render(cur_cam)
                

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                # loss = loss + F.mse_loss(image, self.input_img_torch_list[b_idx])
                loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img_torch_list[b_idx]) / self.opt.batch_size

                if USE_LPIPS:
                    gt_image = self.input_img_torch_list[b_idx]
                    lambda_dssim = 1
                    loss = loss + lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss = loss + self.loss_fn_vgg(image, gt_image).reshape(-1).squeeze()
                    


                # # mask loss
                # mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                # loss = loss +  F.mse_loss(mask, self.input_mask_torch_list[b_idx])
                # print(loss.item())

            if USE_NOVEL_VIEW:
                ### novel view (manual batch)
                render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
                # render_resolution = 512
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

            else:
                for _ in range(self.opt.n_views):
                    for b_idx in range(self.opt.batch_size):

                        idx = random.randint(0, len(self.mv_images)-1)
                        cam, gt_image, gt_mask = self.mv_cameras[idx], self.mv_images[idx], self.mv_masks[idx] 
                
                        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
                        gs_out = self.renderer.render(cam, bg_color=bg_color) 
                        # image, alpha, rend_normal, surf_normal = gs_out["image"], gs_out["rend_alpha"], gs_out["rend_normal"], gs_out["surf_normal"]
                        image, alpha = gs_out["image"], gs_out["alpha"] # , gs_out["rend_normal"], gs_out["surf_normal"]
                        
                        # image loss 
                        loss = (1.0 - self.opt.lambda_dssim) * l1_loss(image, gt_image) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    
                        # mask loss  
                        loss  = loss+ self.opt.lambda_mask * (alpha * (1 - gt_mask)).mean()


            if self.opt.anisotropy_regularizer:
                r= 2.0
                scalings = self.renderer.gaussians.get_scaling
                anisotropy_regularizer = torch.mean(torch.max(scalings.max(dim=1)[0] / scalings.min(dim=1)[0], torch.tensor(r)) - r)
                loss = loss + anisotropy_regularizer * 1000

                # ensure it is not too large
                weight = 1.0
                threshold = 0.005
                scalings.mean(dim=1)
                excess = torch.relu(scalings.mean(dim=1) - threshold).mean()
                penalty_loss = weight * torch.pow(excess, 2)
                loss = loss + penalty_loss * 1000


            # # temporal regularization
            # for t in range(self.opt.batch_size):
            #     means3D_final, rotations_final, scales_final, opacity_final = self.renderer.gaussians.get_deformed_everything(t)
            


            # print(loss.item())
            # optimize step
            # print(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    
    def load_input(self, file):
        # file_list = [file.replace('000_rgba.png', f'{x:03d}_rgba.png') for x in range(self.opt.batch_size)]
        # file_list = [file.replace('000_rgba.png', f'{x:03d}_rgba.png') for x in SEL_LIST]
        self.input_img_list, self.input_mask_list = [], []
        # for file in file_list:
        # load image
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

    def load_svd_input(self, input_dir):
        # opt_batch_size = 20
        # if file.endswith('0000.png'):
        #     file_list = [file.replace('0000.png', f'{x:04d}.png') for x in range(opt_batch_size)]
        # else:
        #     file_list = [file.replace('.png', f'_{x:04d}.png') for x in range(self.opt.batch_size)]

        file_list = sorted(glob.glob(f'{input_dir}/*.png'))
        
        self.mv_images, self.mv_masks = [], []
        for file in file_list:
            # load image
            print(f'[INFO] load image from {file}...')
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            # if img.shape[-1] == 3:
            #     if self.bg_remover is None:
            #         self.bg_remover = rembg.new_session()
            #     img = rembg.remove(img, session=self.bg_remover)
            #     cv2.imwrite(file.replace('.jpg', '_rgba.png'), img) 
            img = cv2.resize(img, (self.opt.ref_size, self.opt.ref_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            input_mask = img[..., 3:]
            input_image = img[..., :3] 
            # # white bg
            # input_img = img[..., :3] * input_mask + (1 - input_mask)
            # # bgr to rgb
            input_image = input_image[..., ::-1].copy()

            input_mask = torch.from_numpy(input_mask).float().permute(2, 0, 1).to(self.device) 
            input_image = torch.from_numpy(input_image).float().permute(2, 0, 1).to(self.device) 
            input_image = input_image * input_mask   
            if self.opt.white_background:
                input_image += (1 - input_mask) 



            self.mv_images.append(input_image)
            self.mv_masks.append(input_mask)

        if NUM_FRAMES == 80:
            indices = [ 0,  1,  3,  4,  6,  8,  9, 11, 13, 14, 16, 18, 19, 21, 23, 24, 26,
            27, 29, 31, 32, 34, 36, 37, 39, 41, 42, 44, 46, 47, 49, 51, 52, 54,
            55, 57, 59, 60, 62, 64, 65, 67, 69, 70, 72, 74, 75, 77, 79]
            # azimuths_deg = list(np.linspace(0, 360, 80) % 360) # [1:]  

            ang_list = []
            frame_length = 80   
            ang = 0.     
            step = 360 / frame_length
            for i in range(frame_length):
                ang_list.append(ang)
                ang = ang - step
            azimuths_deg = list(np.array(ang_list)[indices])
        elif NUM_FRAMES == 40:

            ang_list = []
            frame_length = 40   
            ang = 0.     
            step = 360 / frame_length
            for i in range(frame_length):
                ang_list.append(ang)
                ang = ang + step
            azimuths_deg = list(np.array(ang_list))

        elif NUM_FRAMES == 37:
            print('enter')
            indices = list(range(0, 37))
            ang_list = []
            frame_length = 40   
            ang = 0.     
            step = 360 / frame_length
            for i in range(frame_length):
                ang_list.append(ang)
                ang = ang + step
            azimuths_deg = list(np.array(ang_list)[indices])
            
        if NUM_FRAMES == 20 and self.opt.transforms: # 4ddress multiview
            mv_cameras = []
            import json
            with open(self.opt.transforms, "r") as f:
                meta = json.load(f)

            cam_angle_x = float(meta["camera_angle_x"])
            frames = meta["frames"]
            # proj = self.renderer.proj_from_fovx()

                # c2w comes in as numpy; ensure dtype/shape
            

            # build projection from horizontal FOV to match OrbitCamera.perspective conventions
            # proj = self.renderer.proj_from_fovx(float(cam_angle_x), self.opt.ref_size, self.opt.ref_size).astype(np.float32)

            # for i in range(NUM_FRAMES):
            #     c2w = frames[i]['transform_matrix']
            #     c2w = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
            #     mv_cameras.append((c2w, proj))

            #     get_transforms_cameras

            transforms = [np.array(fr["transform_matrix"], dtype=np.float32) for fr in frames]
            print(len(transforms))
            cam_angle_x = float(meta["camera_angle_x"])  # radians

            mv_cameras = self.get_transforms_cameras(transforms, cam_angle_x)
            self.cam_angle_x = float(cam_angle_x)


        else:
            mv_cameras = self.get_cameras(self.opt.elevation, azimuths_deg, self.opt.radius) # [:-1]
        self.mv_cameras = mv_cameras


    @torch.no_grad()
    def get_cameras(self, elevation, azimuths_deg, radius):
        """
        elevation: int
        azimuths_deg: list of int
        radius: float
        """
        cameras = []
        camera_setting = dict(
            width=self.opt.ref_size,
            height=self.opt.ref_size,
            fovy=self.cam.fovy,
            fovx=self.cam.fovx,
            znear=self.cam.near,
            zfar=self.cam.far,
        )
        for azimuth in azimuths_deg:
            pose = orbit_camera(elevation, azimuth, radius)
            cameras.append(MiniCam(pose, **camera_setting))
        return cameras
    
    @torch.no_grad()
    def get_transforms_cameras(self, transforms, cam_angle_x):
        """
        transforms: list/iterable of (4,4) c2w matrices (NeRF style)
        cam_angle_x: float (radians), from NeRF meta["camera_angle_x"]
        Uses self.opt.ref_size for (W,H) and self.cam.{near,far} for depth range.
        """
        W = H = int(self.opt.ref_size)
        znear = float(self.cam.near)
        zfar  = float(self.cam.far)

        cameras = []
        for c2w in transforms:
            cameras.append(
                MiniCamTransforms(
                    c2w=np.asarray(c2w, dtype=np.float32),
                    width=W,
                    height=H,
                    cam_angle_x=float(cam_angle_x),
                    znear=znear,
                    zfar=zfar,
                )
            )
        return cameras
    
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, t=0):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = f'logs/{opt.save_path}_mesh_{t:03d}.ply'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = f'logs/{opt.save_path}_mesh_{t:03d}.obj'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
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

                cur_cam = MiniCamTransforms(
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

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
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
                
                # albedo += cur_albedo
                # cnt += cur_cnt
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

        os.makedirs(f'logs/{opt.save_path}', exist_ok=True)
        image_list =[]
        nframes = 14 *5
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            # cam_angle_x = self.mv_cameras[0].cam_angle_x
            cur_cam = MiniCamTransforms(
                pose,
                1024,
                1024,
                # self.cam.fovy,
                # self.cam.fovx,
                # float(cam_angle_x),
                self.cam_angle_x,
                self.cam.near,
                self.cam.far,
                time=time
            )
            # cur_cam
            with torch.no_grad():
                self.renderer.gaussians.update_to_posed_mesh(posed_mesh = self.posed_meshes[time])
                outputs = self.renderer.render(cur_cam)

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size
            hor = (hor+delta_hor) % 360

        export_to_gif(image_list, f'logs/{opt.save_path}.gif')

        # render front view
        os.makedirs(f'logs/{opt.save_path}', exist_ok=True)
        image_list =[]
        nframes = self.opt.batch_size # *5
        # delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for idx in range(nframes):
            

            for deg in [0, 90, 180, 270]:
                pose = orbit_camera(0, deg, self.opt.radius)
                # cam_angle_x = self.mv_cameras[0].cam_angle_x
                cur_cam = MiniCamTransforms(
                    pose,
                    1024,
                    1024,
                    # self.cam.fovy,
                    # self.cam.fovx,
                    self.cam_angle_x,
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

                out_path = f'logs/{opt.save_path}/{idx}_{deg}.png'
                out.save(out_path)
                if deg == 0:
                    image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size
            print(time)
            print(idx)
            # hor = (hor+delta_hor) % 360

        export_to_gif(image_list, f'logs/{opt.save_path}_1view.gif')

        # save
        self.save_model(mode='model')
        self.renderer.gaussians.save_deformation(self.opt.outdir, self.opt.save_path)


        if self.gui:
            while True:
                self.viser_gui.update()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--batch", type=lambda x: (str(x).lower() == 'true'), required=True, help="True or False for the first stage")
    parser.add_argument("--model", required=False, help="path to the yaml config file")
    parser.add_argument("--dir", required=False, help="path to the main dir")
    parser.add_argument("--num_frames", required=False, help="path to the main dir")
    # parser.add_argument("--use_obj_mesh", type=lambda x: (str(x).lower() == 'true'), required=True, help="True or False for the first stage")


    args, extras = parser.parse_known_args()


    base_opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    NUM_FRAMES = int(args.num_frames)
    print(f"using {NUM_FRAMES}")

    if args.batch:
        args_dir   = args.dir
        data_root  = Path("data") / args_dir
        fids       = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
        model_name = getattr(args, "model", "default")

        for fid in fids:
            fid_dir       = data_root / fid
            train_images  = fid_dir / "train_images"
            input_img     = train_images / "train_0000.png"
            transforms    = fid_dir / "transforms_train.json"

            # quick sanity checks (skip if required files are missing)
            missing = [name for name, ok in [
                ("train_images", train_images.exists()),
                ("input image", input_img.exists()),
                ("transforms", transforms.exists()),
            ] if not ok]
            if missing:
                print(f"[skip] {fid}: missing {', '.join(missing)}")
                continue

            opt = OmegaConf.merge(
                base_opt,
                {
                    "fid": fid,
                    "gui": False,
                    "num_frames": NUM_FRAMES,
                    "input": str(input_img),
                    "input_svd": str(train_images),
                    "load_mesh": f"logs/output/{args_dir}/{fid}_smplx.obj",
                    "smplx_gaussians": f"logs/output/{args_dir}/{fid}_clothed_smplx.ply",
                    "lgm_ply": f"logs/output/{args_dir}/{fid}/{fid}_gs.ply",
                    "use_pretrained": False,
                    "save_path": f"output/{args_dir}/{model_name}/{fid}",
                    "white_background": True,
                    "lambda_dssim": 0.2,
                    "lambda_mask": 1.0,
                    "transforms": str(transforms),
                    "anisotropy_regularizer": False,
                    "n_views": 8,
                },
            )

            gui = GUI(opt)
            try:
                gui.train(opt.iters)
            finally:
                del gui
                torch.cuda.empty_cache()

    else:
        opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
        gui = GUI(opt)
        gui.render() if opt.gui else gui.train(opt.iters)
