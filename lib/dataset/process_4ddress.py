import os
import tqdm
import pickle
import trimesh
import numpy as np
from PIL import Image

import sys
# sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from omegaconf import OmegaConf
from utils.cam_utils import orbit_camera, OrbitCamera
from lib.mesh_utils.mesh_renderer_smplx import Renderer

# load data from pkl_dir
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# save data to pkl_dir
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# get xyz rotation matrix
def rotation_matrix(angle, axis='x'):
    # get cos and sin from angle
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # get totation matrix
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

def save_mesh_to_obj(file_path, vertices, faces, vertex_colors=None):
    with open(file_path, 'w') as file:
        # Write vertices
        for i, vertex in enumerate(vertices):
            if vertex_colors is not None:
                color = vertex_colors[i]
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
            else:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces
        for face in faces:
            file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def preprocess_scan_mesh(vertices, mcentral=False, bbox=True, bcenter=None, rotation=None, offset=None, scale=1.0):
    # get scan vertices mass center
    mcenter = np.mean(vertices, axis=0)
    # get scan vertices bbox center
    if bcenter is None:
        bmax = np.max(vertices, axis=0)
        bmin = np.min(vertices, axis=0)
        bcenter = (bmax + bmin) / 2
    # centralize scan data around mass center
    if mcentral:
        vertices -= mcenter
    # centralize scan data around bbox center
    elif bbox:
        vertices -= bcenter
    # scale scan vertices
    vertices /= scale
    # rotate scan vertices
    if rotation is not None:
        vertices = np.matmul(rotation, vertices.T).T
    # offset scan vertices
    if offset is not None:
        vertices += offset
    # return scan data, centers, scale
    return vertices, {'mcenter': mcenter, 'bcenter': bcenter}, scale

            
if __name__ == '__main__':

    fid_list = ['00134_Inner/Inner/Take4',
    '00156_Inner/Inner/Take1',
    '00147_Inner/Inner/Take1',
    '00163_Inner_2/Inner/Take9',
    '00129_Inner/Inner/Take3',
    '00149_Inner_1/Inner/Take4',
    '00136_Inner/Inner/Take5',
    '00151_Inner/Inner/Take5']

    dataset_dir = '/mnt/sda/4D-Dress'
    out_dir = 'data/4ddress_video'
    out_img_dir = 'data/4ddress_img/img'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    for fid in fid_list:
        subj, outfit, seq = fid.split('/')

        OUTDIR = f'{out_dir}/{subj}_{seq}'
        os.makedirs(OUTDIR, exist_ok=True)
        os.makedirs(f'{OUTDIR}/meshes', exist_ok=True)

        subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
        scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
        smpl_dir = os.path.join(subj_outfit_seq_dir, 'SMPL')
        smplx_dir = os.path.join(subj_outfit_seq_dir, 'SMPLX')
        label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
        cloth_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'clothes')

        n_start = 0
        n_stop = -1

        # locate scan_frames from basic_info
        basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
        print('# # ============ Compact View Subj_Outfit_Seq: {}_{}_{} // Frames: {}'.format(subj, outfit, seq, len(scan_frames)))

        SLICE =  len(scan_frames)//14


        loop = tqdm.tqdm(range(n_start, len(scan_frames), SLICE))
        for idx, n_frame in enumerate(loop):


            # check stop frame
            if 0 <= n_stop < n_frame: break
            frame = scan_frames[n_frame]
            loop.set_description('## Loading Frame for {}_{}_{}: {}/{}'.format(subj, outfit, seq, frame, scan_frames[-1]))

            # locate scan, smpl, smplx files
            scan_mesh_fn = os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame))
            smpl_mesh_fn = os.path.join(smpl_dir, 'mesh-f{}_smpl.ply'.format(frame))

            # smplx_frname = update_smplx_fname(frame, leading_idx)
            smplx_mesh_fn = os.path.join(smplx_dir, 'mesh-f{}_smplx.ply'.format(frame))
            scan_label_fn = os.path.join(label_dir, 'label-f{}.pkl'.format(frame))
            scan_cloth_fn = os.path.join(cloth_dir, 'cloth-f{}.pkl'.format(frame))

            # load scan_mesh with vertex colors
            scan_mesh = load_pickle(scan_mesh_fn)
            scan_mesh['uv_path'] = scan_mesh_fn.replace('mesh-f', 'atlas-f')
            if 'colors' not in scan_mesh:
                # load atlas data
                atlas_data = load_pickle(scan_mesh['uv_path'])
                # load scan uv_coordinate and uv_image as TextureVisuals
                uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
                texture_visual = trimesh.visual.texture.TextureVisuals(uv=scan_mesh['uvs'], image=uv_image)
                # pack scan data as trimesh
                scan_trimesh = trimesh.Trimesh(
                    vertices=scan_mesh['vertices'],
                    faces=scan_mesh['faces'],
                    vertex_normals=scan_mesh['normals'],
                    visual=texture_visual,
                    process=False,
                )
                scan_mesh['colors'] = scan_trimesh.visual.to_color().vertex_colors

            if scan_rotation is not None: scan_mesh['vertices'] = np.matmul(scan_rotation, scan_mesh['vertices'].T).T

            smplx_trimesh = trimesh.load_mesh(smplx_mesh_fn)
            if scan_rotation is not None: smplx_trimesh.vertices = np.matmul(scan_rotation, smplx_trimesh.vertices.T).T

            if idx == 0:
                _, center, scale = preprocess_scan_mesh(scan_mesh['vertices'].copy(), mcentral=False, bbox=True, scale=2.0)

            # save normalized meshes
            new_mesh_verts, _, _ = preprocess_scan_mesh(scan_mesh['vertices'], mcentral=False, bbox=True, bcenter=center['bcenter'], scale=scale)
            save_mesh_to_obj(f'{OUTDIR}/meshes/{idx}_norm.obj', new_mesh_verts, scan_mesh['faces'], scan_mesh['colors']/ 255.)
            new_mesh_verts, _, _ = preprocess_scan_mesh(smplx_trimesh.vertices, mcentral=False, bbox=True, bcenter=center['bcenter'], scale=scale)
            save_mesh_to_obj(f'{OUTDIR}/meshes/{idx}_norm_smplx.obj', new_mesh_verts, smplx_trimesh.faces)



        # render frames from mesh
        device = 'cuda'
        args_config = 'configs/mesh_seg.yaml'
        opt = OmegaConf.load(args_config)

        ssaa = 2.0
        opt.radius = 1.5
        opt.ref_size = 1024

        for fidx in range(14):

            out_img_folder = f'{OUTDIR}/img' 
            os.makedirs(out_img_folder, exist_ok=True)
            opt.mesh = f'{OUTDIR}/meshes/{fidx}_norm.obj'
            renderer = Renderer(opt).to(device)

            pose = orbit_camera(opt.elevation, 0, opt.radius)
            cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
            fixed_cam = (pose, cam.perspective)

            out = renderer.render(*fixed_cam, opt.ref_size, opt.ref_size, ssaa=ssaa)
            img = out['image'].detach().cpu().numpy()

            img = ((out['image'].cpu().detach().numpy()) * 255).astype('uint8')
            mask_np = ((out['alpha'].cpu().detach().numpy()) * 255).astype('uint8')
            rgba = np.dstack((img, mask_np[..., 0]))  # Shape: (1024, 1024, 4)
            image = Image.fromarray(rgba, mode='RGBA')

            str_idx = str(fidx).zfill(3)
            dst_file = f'{out_img_folder}/{str_idx}_rgba.png'
            image.save(dst_file)

            if fidx == 0:
                img_dst_file = f'{out_img_dir}/{subj}_{seq}_square_rgba.png'
                image.save(img_dst_file)


        break

