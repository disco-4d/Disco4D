# lib/data/utils/config_generator.py

import os
import yaml

def generate_configs(img_dir, configs_dir):
    os.makedirs(configs_dir, exist_ok=True)
    dir_name = os.path.basename(os.path.dirname(img_dir))
    template_path = 'configs/smplx_opt.yaml'

    name_ids = [f.replace('.png', '') for f in os.listdir(img_dir)]

    for name_id in name_ids:
        with open(template_path, 'r') as f:
            data = yaml.safe_load(f)

        for key in ['smplx_params_path', 'save_path', 'input', 'input_back', 'input_svd', 'lgm_mesh', 'smplx_gaussians', 'load_mesh', 'lgm_ply']:
            data[key] = data[key].replace('example', name_id).replace('/compile/', f'/{dir_name}/')

        with open(os.path.join(configs_dir, f'{name_id}.yaml'), 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
