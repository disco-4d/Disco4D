import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg

from utils.cam_utils import orbit_camera, OrbitCamera
from lib.gs_utils.gs_renderer import Renderer, MiniCam

from utils.grid_put import mipmap_linear_grid_put_2d
from lib.mesh_utils.mesh import Mesh, safe_normalize
from lib import seg_config

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T



class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
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

            # define torch tensor to PIL image transform
            self.transform = T.ToPILImage()

            self.seg_processor = SegformerImageProcessor.from_pretrained(seg_cfg.model_name)
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_cfg.model_name)

            self.cat_map = dict(seg_cfg.cat_map)
            self.input_seg_torch = None

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

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
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

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
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

        # make predictions
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


        # Remap the tensor
        pseudo_gt_seg = pred_seg.clone()
        for k, v in self.cat_map.items():
            pseudo_gt_seg = torch.where(pred_seg == k, v, pseudo_gt_seg)
                        
        return pseudo_gt_seg

    @staticmethod
    def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
        """
        Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
        and the KL divergence.
        
        :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
        :param predictions: Tensor of shape (N, C), where C is the number of classes.
        :param k: Number of neighbors to consider.
        :param lambda_val: Weighting factor for the loss.
        :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
        :param sample_size: Number of points to randomly sample for computing the loss.
        
        :return: Computed loss value.
        """
        # Conditionally downsample if points exceed max_points
        if features.size(0) > max_points:
            indices = torch.randperm(features.size(0))[:max_points]
            features = features[indices]
            predictions = predictions[indices]


        # Randomly sample points for which we'll compute the loss
        indices = torch.randperm(features.size(0))[:sample_size]
        sample_features = features[indices]
        sample_preds = predictions[indices]

        # Compute top-k nearest neighbors directly in PyTorch
        dists = torch.cdist(sample_features, features)  # Compute pairwise distances
        _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

        # Fetch neighbor predictions using indexing
        neighbor_preds = predictions[neighbor_indices_tensor]

        # Compute KL divergence
        kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
        loss = kl.sum(dim=-1).mean()

        # Normalize loss into [0, 1]
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

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            cur_cam = self.fixed_cam
            out = self.renderer.render(cur_cam)
            ### known view
            if self.input_seg_torch is None:
                self.input_seg_torch = self.get_seg_inference(out["image"])
                # self.renderer.gaussians.initialize_from_segmask(self.input_seg_torch, self.cam)


            if self.opt.use_seg and self.input_seg_torch is not None:
                pred_seg = out["segmentation"].unsqueeze(0)

                # pred_obj = torch.argmax(out["segmentation"],dim=0)
                # colormap = plt.cm.get_cmap('viridis', 16)
                # colored_seg = colormap(pred_obj.detach().cpu().numpy())
                # plt.imsave(f'logs/debug/{self.step}.png', colored_seg)

                loss_seg = self.cls_criterion(pred_seg, self.input_seg_torch.unsqueeze(0).cuda()).squeeze(0).mean()
                loss = 1000* step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))  # normalize to (0,1)


            ### novel view (manual batch)
            # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            render_resolution = 1024
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # radius = 0
                radius = np.random.uniform(-0.5, 0.5) 

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam)

                # image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                # images.append(image)


                if self.opt.use_seg:
                    pred_seg = out["segmentation"].unsqueeze(0)

                    # make predictions
                    pseudo_gt_seg = self.get_seg_inference(out["image"])

                    loss_seg = self.cls_criterion(pred_seg, pseudo_gt_seg.unsqueeze(0).cuda()).squeeze(0).mean()
                    loss += 1000* step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))  # normalize to (0,1)
                

                    loss_obj_3d = None
                    if self.step % 5 == 0: # reg3d_interval
                        # regularize at certain intervals
                        reg3d_k = 5
                        reg3d_lambda_val = 2
                        reg3d_max_points = 200000
                        reg3d_sample_size = 1000

                        logits3d = self.renderer.gaussians._objects_dc.permute(2,0,1)
                        prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                        loss_obj_3d = self.loss_cls_3d(self.renderer.gaussians._xyz.squeeze().detach(), prob_obj3d, reg3d_k, reg3d_lambda_val, reg3d_max_points, reg3d_sample_size)
                        loss +=  10000* step_ratio *loss_obj_3d

                        
            # print(loss.item())
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # # densify and prune
            # if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
            #     viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            #     self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if self.step % self.opt.densification_interval == 0:
            #         # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
            #         self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
            #     if self.step % self.opt.opacity_reset_interval == 0:
            #         self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    
    def load_input(self, file):
        # load image
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
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
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
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")


    # no gui mode
    def train(self, iters=500, ui=False):
        image_list =[]
        from PIL import Image
        from diffusers.utils import export_to_video, export_to_gif
        interval = 2
        nframes = iters // interval # 250
        hor = 180
        delta_hor = 4 * 360 / nframes

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

                if i % interval == 0:
                    pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                    cur_cam = MiniCam(
                        pose,
                        256,
                        256,
                        self.cam.fovy,
                        self.cam.fovx,
                        self.cam.near,
                        self.cam.far,
                    )
                    with torch.no_grad():
                        outputs = self.renderer.render(cur_cam)

                    out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                    out = np.transpose(out, (1, 2, 0))
                    out = Image.fromarray(np.uint8(out*255))
                    image_list.append(out)
                    
                    hor = (hor+delta_hor) % 360
            export_to_gif(image_list, f'logs/{self.opt.save_path}_static.gif')
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        # self.save_model(mode='geo+tex')

        image_list =[]
        from PIL import Image
        from diffusers.utils import export_to_video, export_to_gif
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
            )

            outputs = self.renderer.render(cur_cam)

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % 14
            hor = (hor+delta_hor) % 360

        export_to_gif(image_list, f'logs/{self.opt.save_path}_static.gif')

        for cat_idx in seg_config.get().render_categories:
            image_list, seg_list =[], []
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
                )

                outputs = self.renderer.render_categorical(cur_cam, category=cat_idx)

                out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
                out = Image.fromarray(np.uint8(out*255))
                image_list.append(out)

                time = (time + delta_time) % 14
                hor = (hor+delta_hor) % 360

                pred_obj = torch.argmax(outputs["segmentation"],dim=0)
                colormap = plt.cm.get_cmap('viridis', seg_config.get().num_classes)
                colored_img = colormap(pred_obj.detach().cpu().numpy())
                out_seg = Image.fromarray(np.uint8(colored_img[..., :3]*255))
                seg_list.append(out_seg)
                
            export_to_gif(image_list, f'logs/{self.opt.save_path}_cat{cat_idx}__static.gif')
            export_to_gif(seg_list, f'logs/{self.opt.save_path}_cat{cat_idx}__static_segmentation.gif')

        if self.gui:
            while True:
                self.viser_gui.update()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--dir", required=True, help="path to the main dir")
    # parser.add_argument("--first_stage", action="store_true", help="path to the main dir")
    parser.add_argument("--first_stage", type=lambda x: (str(x).lower() == 'true'), required=True, help="True or False for the first stage")
    parser.add_argument("--batch", type=lambda x: (str(x).lower() == 'true'), required=True, help="True or False for the first stage")
    args, extras = parser.parse_known_args()

    DIR = args.dir

    use_batch = args.batch
    
    if use_batch:
        # override default config from cli
        for file in tqdm.tqdm(glob.glob(f'configs/{DIR}/*.yaml')):
            print(file)

            opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

            FID = file.replace(f'configs/{DIR}/', '').replace('.yaml', '')

            if args.first_stage:
                opt.save_path = f"output/{DIR}/{FID}_seg"
                opt.load = f'data/{DIR}/lgm/{FID}.ply'
                os.makedirs(f"logs/output/{DIR}", exist_ok=True)
            else:
                opt.save_path = f"output/{DIR}/{FID}_refined_cat_seg"
                opt.load = f'logs/output/{DIR}/{FID}_refined_cat.ply'
            print(os.path.exists(opt.load), opt.load)
            opt.iters = 20

            gui = GUI(opt)

            if opt.gui:
                gui.render()
            else:
                gui.train(opt.iters)
    else:
        opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
        FID = os.path.splitext(os.path.basename(args.config))[0]

        if args.first_stage:
            opt.save_path = f"output/{DIR}/{FID}_seg"
            opt.load = f'data/{DIR}/lgm/{FID}.ply'
        else:
            opt.save_path = f"output/{DIR}/{FID}_refined_cat_seg"
            opt.load = f'logs/output/{DIR}/{FID}_refined_cat.ply'
        os.makedirs(f"logs/output/{DIR}", exist_ok=True)
        opt.iters = 20

        gui = GUI(opt)

        if opt.gui:
            gui.render()
        else:
            gui.train(opt.iters)