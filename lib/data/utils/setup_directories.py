# lib/data/utils/setup_directories.py

import os

def setup_directories(path, process_rgba=False):
    """
    Given a path to raw or image folder, construct all required output directories.
    Returns a dictionary of key working paths.
    """
    if os.path.isdir(path):
        main_dir = path
        dir_name = os.path.basename(os.path.dirname(path))
        img_dir = main_dir.replace('/raw', '/img') if process_rgba else path
    else:
        raise ValueError("Expected a directory path, got a file path instead.")

    os.makedirs(img_dir, exist_ok=True)

    def make(subdir): 
        out = img_dir.replace('/img', subdir)
        os.makedirs(out, exist_ok=True)
        return out

    return {
        'main_dir': main_dir,
        'dir_name': dir_name,
        'img_dir': img_dir,
        'svd_dir': make('/svd'),
        # 'svd_mmpose_dir': make('/svd/mmpose'),
        # 'svd_rgba_dir': make('/svd/rgba'),
        # 'svd_seg_dir': make('/svd/seg'),
        'smplx_dir': make('/smplerx'),
        'seg_dir': make('/seg'),
        'mmpose_dir': make('/mmpose'),
        'configs_dir': f'configs/{dir_name}',
        'clothless_img_dir': make('/clothless_img'),
        'skin_img_dir': make('/skin_img'),
        'lgm_dir': make('/lgm'),
    }