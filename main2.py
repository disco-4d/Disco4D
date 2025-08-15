import os
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
from mesh_renderer import Renderer
from utils.grid_put import mipmap_linear_grid_put_2d, linear_grid_put_2d, nearest_grid_put_2d

# from kiui.lpips import LPIPS
from PIL import Image
import kiui
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
import torchvision.transforms.functional as TF
# import torchvision.transforms as transforms
import torchvision.transforms as T
import random
SD_INPAINT = True
pil_to_tensor = T.ToTensor()
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt

def remap_obj(obj1_path, save_path):
    from mesh_renderer import Renderer
    from omegaconf import OmegaConf
    
    config = 'configs/image.yaml'
    extras = ['input=data/thuman/0001/render/images/150_000.png', \
            'save_path=output/mesh_seg', \
                    'elevation=0', 'force_cuda_rast=True', 'gui=True']
    
    opt = OmegaConf.merge(OmegaConf.load(config), OmegaConf.from_cli(extras))
    opt.radius = 1.5
    device = 'cuda'
    opt.resolution = 1024
    opt.ref_size = 1024

    opt.mesh = obj1_path
    renderer = Renderer(opt).to(device)

    opt.mesh = 'data/smplx_uv/smplx_tex.obj'
    renderer2 = Renderer(opt).to(device)
    
    renderer.mesh.ft = renderer2.mesh.ft
    renderer.mesh.vt = renderer2.mesh.vt

    renderer.export_mesh(save_path)

def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int

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


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = 1024 # opt.W
        self.H = 1024  # opt.H
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
        self.renderer = Renderer(opt).to(self.device)

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
        # self.lpips_loss = LPIPS(net='vgg').to(self.device)
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
            self.initialize(keep_ori_albedo=False)
        
        if self.opt.back_input is not None:
            self.load_input_back(self.opt.back_input)

        if self.opt.svd_input is not None:
            self.load_svd_input(self.opt.svd_input)
            print('loaded')

        if self.opt.train_seg:
            self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.num_classes = 16

            # load in the segmentation model

            # define torch tensor to PIL image transform
            self.transform = T.ToPILImage()

            # define processor
            self.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

            clothing_dict = {
                0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", \
                    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", \
                        14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
            } 


            self.cat_map = {
                    16:0, 
                    17: 0
            }
            self.input_seg_torch = None

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        if SD_INPAINT:
            self.prepare_sd_guidance()
        else:
            self.prepare_guidance()
        self.opt.texture_size = 1024
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
        self.face_inpainting_mask = torch.zeros(20908).float().cuda()  # Initialize with zeros, assuming the number of faces in the mesh is known
    
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


    def prepare_sd_guidance(self):
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            # revision="fp16",
            # torch_dtype=torch.float16,
            device='cuda'
        )
        # prompt = ""
        self.pipe = pipe.to("cuda")


    def prepare_guidance(self):
        
        self.opt.control_mode = ['depth_inpaint']
        self.opt.model_key= 'philz1337/revanimated'
        self.opt.lora_keys= []
        self.guidance = None


        self.opt.posi_prompt= "masterpiece, high quality"
        self.negative_prompt = "bad quality, worst quality, shadows"
        self.prompt = self.opt.posi_prompt + ', ' + self.opt.prompt

        if self.guidance is None:
            print(f'[INFO] loading guidance model...')
            from guidance.sd_inpaint_utils import StableDiffusion
            self.guidance = StableDiffusion(self.device, control_mode=self.opt.control_mode, model_key=self.opt.model_key, lora_keys=self.opt.lora_keys)
            print(f'[INFO] loaded guidance model!')

        print(f'[INFO] encoding prompt...')
        nega = self.guidance.get_text_embeds([self.negative_prompt])

        # if not self.opt.text_dir:
        #     posi = self.guidance.get_text_embeds([self.prompt])
        #     self.guidance_embeds = torch.cat([nega, posi], dim=0)
        # else:
        self.guidance_embeds = {}
        posi = self.guidance.get_text_embeds([self.prompt])
        self.guidance_embeds['default'] = torch.cat([nega, posi], dim=0)
        for d in ['front', 'side', 'back', 'top', 'bottom']:
            posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
            self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)


        
        
    def prepare_train(self):

        self.step = 0

        # setup training
        if not self.opt.train_seg:
            self.renderer.trainable = True
            self.renderer.init_params(self.face_inpainting_mask)
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

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def get_seg_inference(self, pred_image):

        # make predictions
        image = self.transform(pred_image)
        inputs = self.seg_processor(images=image, return_tensors="pt")

        outputs = self.seg_model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = torch.nn.functional.interpolate(
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
    
    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):

        # self.prepare_guidance()
        
        h = w = 1024 # int(self.opt.texture_size)

        self.albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        self.cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        self.viewcos_cache = - torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        # # keep original texture if using ip2p
        # if 'ip2p' in self.opt.control_mode:
        #     self.renderer.mesh.ori_albedo = self.renderer.mesh.albedo.clone()

        if keep_ori_albedo:
            self.albedo = self.renderer.mesh.albedo.clone()
            self.cnt += 1 # set to 1
            self.viewcos_cache *= -1 # set to 1
        
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
        
        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()

        self.albedo = dilate_image(self.albedo, mask, iterations=int(h*0.2))
        self.cnt = dilate_image(self.cnt, mask, iterations=int(h*0.2))
        
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.opt.texture_size)

        self.backup()

        # overall deblur by LR then SR
        # kiui.vis.plot_image(self.albedo)
        cur_albedo = self.renderer.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        # kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        # enhance quality by SD refine...
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())
        # text_embeds = self.guidance_embeds if not self.opt.text_dir else self.guidance_embeds['default']
        # albedo = self.guidance.refine(text_embeds, albedo.permute(2,0,1).unsqueeze(0).contiguous(), strength=0.8).squeeze(0).permute(1,2,0).contiguous()
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())

        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def generate(self):

        # self.initialize(keep_ori_albedo=False)

        # vers = [0,]
        # hors = [0,]
        self.opt.camera_path = 'front'

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

        # better to generate a top-back-view earlier
        # vers = [0, -45, -45,  0,   0, -89.9,  0,   0, 89.9,   0,    0]
        # hors = [0, 180,   0, 45, -45,     0, 90, -90,    0, 135, -135]

        start_t = time.time()

        # image inpaint the first one
        pose = orbit_camera(0, 0, self.cam.radius)
        self.image_inpaint(pose)

        # # train step to use zero123 to fill in the rest of the view
        # iters = 200
        # self.train_steps = 200
        # for i in tqdm.trange(iters):
        #     self.train_step()
        
        print(f'[INFO] start generation...')
        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)
            self.inpaint_view(pose)

            # preview
            self.need_update = True
            self.test_step()

        self.dilate_texture()
        self.deblur()

        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    @torch.no_grad()
    def inpaint_view(self, pose):

        h = w = 1024 # int(self.opt.texture_size)
        H = W = 1024 # int(self.opt.render_resolution)
        self.opt.refine_strength = 0.6
        self.opt.vis = False
        self.opt.cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        # valid crop region with fixed aspect ratio
        valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()
        
        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

        # trimap: generate, refine, keep
        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1 # [1, 1, H, W]

        viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

        mask_keep = (~mask_generate & ~mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()

        # mask_refine = mask_generate.float()
        # mask_keep = mask_generate.float()

        # dilate and blur mask
        # blur_size = 9
        # mask_generate_blur = dilation(mask_generate, kernel=torch.ones(blur_size, blur_size, device=mask_generate.device))
        # mask_generate_blur = gaussian_blur(mask_generate_blur, kernel_size=blur_size, sigma=5) # [1, 1, H, W]
        # mask_generate[mask_generate > 0.5] = 1 # do not mix any inpaint region
        mask_generate_blur = mask_generate

        # weight map for mask_generate
        # mask_weight = (mask_generate > 0.5).float().cpu().numpy().squeeze(0).squeeze(0)
        # mask_weight = ndimage.distance_transform_edt(mask_weight)#.clip(0, 30) # max pixel dist hardcoded...
        # mask_weight = (mask_weight - mask_weight.min()) / (mask_weight.max() - mask_weight.min() + 1e-20)
        # mask_weight = torch.from_numpy(mask_weight).to(self.device).unsqueeze(0).unsqueeze(0)

        # kiui.vis.plot_matrix(mask_generate, mask_refine, mask_keep)

        if not (mask_generate > 0.5).any():
            return

        control_images = {}

        # construct normal control
        if 'normal' in self.opt.control_mode:
            rot_normal = out['rot_normal'] # [H, W, 3]
            rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
            control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
        
        # construct depth control
        if 'depth' in self.opt.control_mode:
            depth = out['depth']
            control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1) # [1, 3, H, W]
        
        # construct ip2p control
        if 'ip2p' in self.opt.control_mode:
            ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512)) # [1, 3, H, W]
            control_images['ip2p'] = ori_image

        # construct inpaint control
        if 'inpaint' in self.opt.control_mode:
            image_generate = image.clone()
            image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            control_images['inpaint'] = image_generate

            # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            control_images['latents_mask'] = latents_mask
            control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep
            control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
        
        # construct depth-aware-inpaint control
        if 'depth_inpaint' in self.opt.control_mode:

            image_generate = image.clone()

            # image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
            image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1 # -1 is inpaint region

            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1) # [1, 3, H, W]
            control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1) # [1, 6, H, W]

            # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            control_images['latents_mask'] = latents_mask
            control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep

            # image_fill = image.clone()
            # image_fill = dilate_image(image_fill, mask_generate_blur, iterations=int(H*0.2))

            control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
        
        
        # if not self.opt.text_dir:
        #     text_embeds = self.guidance_embeds
        # else:
        # pose to view dir
        ver, hor, _ = undo_orbit_camera(pose)
        if ver <= -60: d = 'top'
        elif ver >= 60: d = 'bottom'
        else:
            if abs(hor) < 30: d = 'front'
            elif abs(hor) < 90: d = 'side'
            else: d = 'back'
        text_embeds = self.guidance_embeds[d]

        # prompt to reject & regenerate
        rgbs = self.guidance(text_embeds, height=512, width=512, control_images=control_images, refine_strength=self.opt.refine_strength).float()

        # performing upscaling (assume 2/4/8x)
        if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
            scale = W // rgbs.shape[-1]
            rgbs = rgbs.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
            rgbs = (rgbs * 255).astype(np.uint8)
            rgbs = kiui.sr.sr(rgbs, scale=scale)
            rgbs = rgbs.astype(np.float32) / 255
            rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        
        # apply mask to make sure non-inpaint region is not changed
        rgbs = rgbs * (1 - mask_keep) + image * mask_keep
        # rgbs = rgbs * mask_generate_blur + image * (1 - mask_generate_blur)

        if self.opt.vis:
            if 'depth' in control_images:
                kiui.vis.plot_image(control_images['depth'])
            if 'normal' in control_images:
                kiui.vis.plot_image(control_images['normal'])
            if 'ip2p' in control_images:
                kiui.vis.plot_image(ori_image)
            # kiui.vis.plot_image(mask_generate)
            if 'inpaint' in control_images:
                kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
            if 'depth_inpaint' in control_images:
                kiui.vis.plot_image(control_images['depth_inpaint'][:, :3].clamp(0, 1))
                kiui.vis.plot_image(control_images['depth_inpaint'][:, 3:].clamp(0, 1))
            kiui.vis.plot_image(rgbs)

        # grid put

        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.opt.cos_thresh)  # [H, W, 1]
        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
        
        # print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')

        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]

        # update mesh texture for rendering
        self.update_mesh_albedo()

        # update viewcos cache
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)


    @torch.no_grad()
    def inpaint_sd_view(self, pose):

        h = w = 1024 # int(self.opt.texture_size)
        H = W = 1024 # int(self.opt.render_resolution)
        cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        # get mask
        proj_mask = (out['alpha'] > 0)
        face_indices = out['face_indices'][0].unsqueeze(-1) # .view(-1) # [0].unsqueeze(-1) * proj_mask
        # face_mask = self.face_inpainting_mask[face_indices].bool()
        inpainted_values = self.face_inpainting_mask[face_indices].bool() * proj_mask
        # inpainted_values = inpainted_values.float()
        inpainted_mask = (~inpainted_values) * out['alpha'].bool()
        mask_image = inpainted_mask.repeat(1, 1, 3).float()

        prompt = ""
        mask_image = TF.to_pil_image(mask_image.permute(2, 0, 1))
        image = TF.to_pil_image(out['image'].permute(2, 0, 1))
        image = self.pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
        image.save('data/debug/ip_out.png')


        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > cos_thresh) & inpainted_mask # [H, W, 1]
        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
        proj_mask = proj_mask.view(1, 1, H, W).float().view(-1).bool()
        uvs = out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous()

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        # rgbs = self.input_img_torch.clone()
        ref_size = 1024
        rgbs = F.interpolate(pil_to_tensor(image).cuda().unsqueeze(0), (ref_size, ref_size), mode="bilinear", align_corners=False)
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
        
        # print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]


        # update mesh texture for rendering
        self.update_mesh_albedo()

        # update viewcos cache
        viewcos = out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    @torch.no_grad()
    def image_inpaint(self, pose):

        h = w = 1024 # int(self.opt.texture_size)
        H = W = 1024 # int(self.opt.render_resolution)
        cos_thresh = 0

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > cos_thresh)  # [H, W, 1]
        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
        proj_mask = proj_mask.view(1, 1, H, W).float().view(-1).bool()
        uvs = out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous()

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = self.input_img_torch.clone()
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
        
        # print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]

        # update mesh texture for rendering
        self.update_mesh_albedo()

        # update viewcos cache
        viewcos = out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)


    @staticmethod
    def squeeze_hw(x: torch.Tensor) -> torch.Tensor:
        """
        Make tensor shape exactly (H, W) by removing leading/trailing singleton dims.
        Works for (H,W), (1,H,W), (H,W,1), (1,H,W,1).
        """
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            # (1,H,W) or (H,W,1)
            if x.size(0) == 1:
                return x[0]
            if x.size(-1) == 1:
                return x[..., 0]
        if x.dim() == 4:
            # (1,H,W,1)
            if x.size(0) == 1 and x.size(-1) == 1:
                return x[0, ..., 0]
        raise ValueError(f"Unexpected tensor shape for squeeze_hw: {tuple(x.shape)}")


    @torch.no_grad()
    def image_inpaint_input(self, cam, torch_img, torch_mask):

        h = w = 1024  # atlas size
        H = W = 1024  # render size

        # Render this view
        out = self.renderer.render(*cam, H, W)

        # --- Build robust geometry mask ---
        # Use only solid interior of triangles (avoid AA edges pulling white bg)
        alpha = out['alpha'].squeeze(-1).clamp(0, 1)          # (H,W)
        alpha_core = alpha > 0.999

        # Gate by view angle (expects signed viewcos > 0 for front-facing;
        # if your render still uses abs(), this still just rejects grazing angles)
        viewcos = out['viewcos'].squeeze(-1)                  # (H,W)
        cos_thresh = 0.25
        cos_mask = viewcos > cos_thresh

        geom_mask_2d = alpha_core & cos_mask                  # (H,W)

        # --- Erode the INPUT mask to avoid white boundary bleed ---
        m = torch_mask.view(1, 1, H, W).float()
        eroded = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - eroded, kernel_size=3, stride=1, padding=1)
        img_mask_interior_2d = (eroded.view(H, W) > 0.5)

        # --- Reject near-white / near-black / low-info pixels (extra safety) ---
        img_hw3 = torch_img.squeeze(0).permute(1, 2, 0).contiguous()  # (H,W,3) in [0,1]
        cmax, _ = img_hw3.max(dim=-1); cmin, _ = img_hw3.min(dim=-1)
        sat = (cmax - cmin) / (cmax + 1e-6)
        near_white = (img_hw3 > 0.98).all(dim=-1)
        near_black = (img_hw3 < 0.02).all(dim=-1)
        gray = 0.299*img_hw3[..., 0] + 0.587*img_hw3[..., 1] + 0.114*img_hw3[..., 2]
        k = torch.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
        lap = F.conv2d(gray.unsqueeze(0).unsqueeze(0), k, padding=1).squeeze(0).squeeze(0)
        low_contrast = lap.abs() < 0.002
        color_good_2d = ~(near_white | near_black | ((sat < 0.08) & low_contrast))

        # --- Final per-pixel sampling mask (H*W,) ---
        proj_mask_2d = geom_mask_2d & img_mask_interior_2d & color_good_2d
        proj_mask = proj_mask_2d.view(-1)

        # --- Gather UVs/RGBs from accepted pixels ---
        # (keep your original UV layout/order)
        uvs_hw2 = out['uvs']  # (H,W,2)
        uvs = uvs_hw2.view(-1, 2)[proj_mask]                       # (N,2)
        rgbs = img_hw3.view(-1, 3)[proj_mask]                      # (N,3)

        # --- Faces corresponding to these pixels; drop background (-1) before indexing ---
        face_indices_full = out['face_indices'].view(-1)           # (H*W,)
        fi = face_indices_full[proj_mask]                          # (N,)
        fi_valid_mask = fi >= 0
        if fi_valid_mask.any():
            fi_valid = fi[fi_valid_mask]
            # Skip faces already inpainted
            already = self.face_inpainting_mask[fi_valid].bool()
            keep_valid = fi_valid_mask.clone()
            keep_valid[fi_valid_mask] = ~already  # only drop those valid pixels whose face is already inpainted
        else:
            fi_valid = fi.new_zeros((0,), dtype=torch.long)
            keep_valid = torch.zeros_like(fi, dtype=torch.bool)

        # Apply keep mask to uvs/rgbs + faces
        if keep_valid.any():
            uvs = uvs[keep_valid]
            rgbs = rgbs[keep_valid]
            face_indices = fi[keep_valid]
        else:
            return  # nothing to write this pass

        # --- Write to atlas (same accumulation style as your original) ---
        # (If your grid writer expects (y,x) in [-1,1], keep the swap)
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True
        )

        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]
        # mark faces touched this pass
        if face_indices.numel() > 0:
            self.face_inpainting_mask[face_indices.unique()] = 1

        # --- Update mesh texture for rendering (unchanged) ---
        self.update_mesh_albedo()

        # --- Update viewcos cache (only for kept pixels) ---
        viewcos_hw1 = out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,1,H,W)
        # Rebuild a flat mask aligned with 'keep_valid' over the previously accepted proj_mask locations
        viewcos_flat = viewcos_hw1.view(-1, 1)[proj_mask][keep_valid]            # (Nk,1)
        cur_viewcos = mipmap_linear_grid_put_2d(
            h, w, uvs[..., [1, 0]] * 2 - 1, viewcos_flat, min_resolution=256
        )
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)



    def get_known_view_loss(self): 
        loss = 0
        for _ in range(10):
            idx = random.randint(0, len(self.mv_images)-1)
            cam, gt_image, gt_mask = self.mv_cameras[idx], self.mv_images[idx], self.mv_masks[idx] 
    
            ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            out = self.renderer.render(*cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa) 
            # image, alpha, rend_normal, surf_normal = gs_out["image"], gs_out["rend_alpha"], gs_out["rend_normal"], gs_out["surf_normal"]
            image = out["image"]
            valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach() # , gs_out["rend_normal"], gs_out["surf_normal"]
            new_mask = (valid_mask & gt_mask.permute(1, 2, 0).bool()).float()
            # image loss 
            loss += F.mse_loss(image * new_mask, gt_image.permute(1, 2, 0) * new_mask)
 
        return loss
    


    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()



        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0
            losses = {}


            if self.opt.train_seg:


                ### novel view (manual batch)
                render_resolution = 512
                images = []
                poses = []
                vers, hors, radii = [], [], []
                # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
                min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
                max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
                for _ in range(self.opt.batch_size):

                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    poses.append(pose)

                    # random render resolution
                    ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                    out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                    image = out["image"] # [H, W, 3] in [0, 1]
                    image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                    images.append(image)

                    # enable mvdream training
                    if self.opt.mvdream or self.opt.imagedream:
                        for view_i in range(1, 4):
                            pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                            poses.append(pose_i)

                            out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                            image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                            images.append(image)

                    # if self.opt.use_seg:
                    pred_seg = out["segmentation"].permute(2,0,1).contiguous().unsqueeze(0)

                    # make predictions
                    pseudo_gt_seg = self.get_seg_inference(out["image"].permute(2,0,1).contiguous())

                    # print(pred_seg.shape, pseudo_gt_seg.shape, out['image'].shape)
                    loss_seg = self.cls_criterion(pred_seg, pseudo_gt_seg.unsqueeze(0).cuda()).squeeze(0).mean()
                    loss += 1000* step_ratio * loss_seg / torch.log(torch.tensor(self.num_classes))  # normalize to (0,1)


            else:


                loss += self.get_known_view_loss() 

                if self.step > 100:


                    render_resolution = 512
                    images = []
                    poses = []
                    vers, hors, radii = [], [], []
                    # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
                    min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
                    max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
                    for _ in range(self.opt.batch_size):

                        # render random view
                        ver = np.random.randint(min_ver, max_ver)
                        hor = np.random.randint(-180, 180)
                        radius = 0

                        vers.append(ver)
                        hors.append(hor)
                        radii.append(radius)

                        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                        poses.append(pose)

                        # random render resolution
                        ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                        out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                        image = out["image"] # [H, W, 3] in [0, 1]
                        image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                        images.append(image)

                        # enable mvdream training
                        if self.opt.mvdream or self.opt.imagedream:
                            for view_i in range(1, 4):
                                pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                                poses.append(pose_i)

                                out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                                image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                                images.append(image)

                    images = torch.cat(images, dim=0)
                    poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

                    # import kiui
                    # kiui.lo(hor, ver)
                    # kiui.vis.plot_image(image)

                    # guidance loss
                    strength = step_ratio * 0.15 + 0.8
                    if self.enable_sd:
                        if self.opt.mvdream or self.opt.imagedream:
                            # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                            refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                            refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                            losses['sd_guidance_loss'] = self.opt.lambda_sd * F.mse_loss(images, refined_images)
                        else:
                            # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                            refined_images = self.guidance_sd.refine(images, strength=strength).float()
                            refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                            losses['sd_guidance_loss'] = self.opt.lambda_sd * F.mse_loss(images, refined_images)

                    if self.enable_zero123:
                        # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                        refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                        refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                        losses['zero123_guidance_loss'] = self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                        # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

                    for k in losses:
                        loss += losses[k]

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
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

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            if self.mode not in ['occ']:
                buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            if self.mode in ['segmentation']:
                pred_obj = torch.argmax(buffer_image,dim=-1)
                colormap = plt.cm.get_cmap('viridis', 16)
                colored_img = colormap(pred_obj.detach().cpu().numpy())
                buffer_image = torch.tensor(colored_img[..., :3], dtype=torch.float32) # .permute(2, 0, 1)


            if self.mode in ['occ']:
                # proj_mask = (out['alpha'] > 0)
                proj_mask = (out['alpha'] > 0)
                face_indices = out['face_indices'][0].unsqueeze(-1) # .view(-1) # [0].unsqueeze(-1) * proj_mask
                # face_mask = self.face_inpainting_mask[face_indices].bool()
                inpainted_values = self.face_inpainting_mask[face_indices].bool() * proj_mask
                inpainted_values = inpainted_values.float()

                buffer_image = inpainted_values.repeat(1, 1, 3) 

                # self.face_inpainting_mask = 

                # buffer_image = face_mask
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
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(
            img, (self.W, self.H), interpolation=cv2.INTER_AREA
        )
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # # load prompt
        # file_prompt = file.replace("_rgba.png", "_caption.txt")
        # if os.path.exists(file_prompt):
        #     print(f'[INFO] load prompt from {file_prompt}...')
        #     with open(file_prompt, "r") as f:
        #         self.prompt = f.read().strip()
    
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            ref_size = 1024
            self.input_img_torch = F.interpolate(self.input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)


    def load_svd_input(self, input_dir):
        file_list = sorted(glob.glob(f'{input_dir}/*.png'))
        
        self.mv_images, self.mv_masks = [], []

        for file in file_list:
            # load image
            print(f'[INFO] load image from {file}...')
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (self.opt.ref_size, self.opt.ref_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            input_mask = img[..., 3:]
            input_image = img[..., :3] 
            input_image = input_image[..., ::-1].copy()

            input_mask = torch.from_numpy(input_mask).float().permute(2, 0, 1).to(self.device) 
            input_image = torch.from_numpy(input_image).float().permute(2, 0, 1).to(self.device) 
            input_image = input_image * input_mask   
            if self.opt.white_background:
                input_image += (1 - input_mask) 


            self.mv_images.append(input_image)
            self.mv_masks.append(input_mask)

            # print(self.mv_images, self.mv_masks)

            if self.opt.num_frames == 20 and self.opt.transforms: # 4ddress multiview
                mv_cameras = []
                import json
                with open(self.opt.transforms, "r") as f:
                    meta = json.load(f)

                cam_angle_x = float(meta["camera_angle_x"])
                frames = meta["frames"]

                # build projection from horizontal FOV to match OrbitCamera.perspective conventions
                proj = self.renderer.proj_from_fovx(float(cam_angle_x), self.opt.ref_size, self.opt.ref_size).astype(np.float32)

                for i in range(self.opt.num_frames):
                    c2w = frames[i]['transform_matrix']
                    c2w = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
                    mv_cameras.append((c2w, proj))

            self.mv_cameras = mv_cameras
        # print(self.mv_cameras, self.mv_images, self.mv_masks)

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

        # img = Image.open(file)
        # img = img.resize((1024, 1024))
        # img = np.array(img)/ 255.0

        self.input_back_mask = img[..., 3:]
        # white bg
        self.input_back_img = img[..., :3] * self.input_back_mask + (1 - self.input_back_mask)
        # bgr to rgb
        self.input_back_img = self.input_back_img[..., ::-1].copy() 

        if self.input_back_img is not None:
            self.input_back_img_torch = torch.from_numpy(self.input_back_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            ref_size = 1024
            self.input_back_img_torch = F.interpolate(self.input_back_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

            self.input_back_mask_torch = torch.from_numpy(self.input_back_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_back_mask_torch = F.interpolate(self.input_back_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        if self.opt.train_seg:
            file_path = path.replace('.obj', '_vclass.npy')
            np.save(file_path, self.renderer.vclass.detach().cpu().numpy())

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
                        label="",                        default_value=self.opt.save_path,
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

                    # TODO: For image inpaint
                    def callback_image_inpaint(sender, app_data):
                        # # inpaint current view
                        # self.image_inpaint(self.cam.pose)

                        # inpaint front and back view 
                        front_pose = self.mv_cameras[0] # orbit_camera(0, 0, self.opt.radius)
                        self.image_inpaint_input(front_pose, self.input_img_torch, self.input_mask_torch)
                        back_pose = self.mv_cameras[10] # orbit_camera(0, 180, self.opt.radius)
                        self.image_inpaint_input(back_pose, self.input_back_img_torch, self.input_back_mask_torch)
                        
                        self.training = False

                    dpg.add_button(
                        label="Image inpaint",
                        tag="_button_image_inpaint",
                        callback=callback_image_inpaint,
                    )
                    dpg.bind_item_theme("_button_image_inpaint", theme_button)

                    def callback_inpaint(sender, app_data):
                        # inpaint current view
                        self.inpaint_view(self.cam.pose)
                        self.need_update = True

                    dpg.add_button(
                        label="inpaint",
                        tag="_button_inpaint",
                        callback=callback_inpaint,
                    )
                    dpg.bind_item_theme("_button_inpaint", theme_button)


                    def callback_sd_inpaint(sender, app_data):
                        # inpaint current view
                        self.inpaint_sd_view(self.cam.pose)
                        self.need_update = True

                    dpg.add_button(
                        label="sd_inpaint",
                        tag="_button_sd_inpaint",
                        callback=callback_sd_inpaint,
                    )
                    dpg.bind_item_theme("_button_sd_inpaint", theme_button)
                    # dpg.add_text("Generate: ")

                    def callback_generate(sender, app_data):
                        self.generate()
                        self.need_update = True

                    dpg.add_button(
                        label="auto",
                        tag="_button_generate",
                        callback=callback_generate,
                    )
                    dpg.bind_item_theme("_button_generate", theme_button)

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
                    ("image", "depth", "alpha", "normal", "occ", "segmentation"),
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
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--dir", required=True, help="path to the main dir")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args, extras = parser.parse_known_args()


    DIR = args.dir
    input_dir = f'data/{args.dir}/'
    fidxs = os.listdir(input_dir)

    # args_stage = int(args.stage)
    base_opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    def run_stage1(FID):
        opt = OmegaConf.merge(base_opt, {
            "fid": FID,
            "front_dir": "+z",
            "mesh": f"logs/output/{DIR}/{FID}_smplx.obj",
            "input": f"data/{DIR}/{FID}/images/train_0000.png",
            "svd_input": f"data/{DIR}/{FID}/train_images",
            "back_input": f"data/{DIR}/{FID}/images/train_0010.png",
            "save_path": f"output/{DIR}/{FID}_clothed_smplx",
            "iters_refine": 750,
            "ref_size": 1024,
            "transforms": f"data/{DIR}/{FID}/transforms_train.json",
            "num_frames": 20,
            "white_background": True,
            "train_seg": False,
        })
        gui = GUI(opt)
        if opt.gui:
            gui.render()
        else:
            front_pose = gui.mv_cameras[0]
            gui.image_inpaint_input(front_pose, gui.input_img_torch, gui.input_mask_torch)
            back_pose = gui.mv_cameras[10]
            gui.image_inpaint_input(back_pose, gui.input_back_img_torch, gui.input_back_mask_torch)
            gui.train(opt.iters_refine)

    def run_stage2(FID):
        opt = OmegaConf.merge(base_opt, {
            "fid": FID,
            "front_dir": "+z",
            "mesh": f"logs/output/{DIR}/{FID}_clothed_smplx.obj",
            "save_path": f"output/{DIR}/{FID}_clothed_smplx",
            "radius": 1.5,
            "ref_size": 1024,
            "iters_refine": 150,
            "train_seg": True,
            "back_input": None,
            "svd_input": None,
            "seg_lr": 0.2,
        })
        gui = GUI(opt)
        if opt.gui:
            gui.render()
        else:
            gui.train(opt.iters_refine)

    for fid in fidxs:
        if args.stage in ("1", "both"):
            run_stage1(fid)
        if args.stage in ("2", "both"):
            run_stage2(fid)