import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict, coco_classes, variance_5_class_index_sorted

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen

def replace(latent_source, latent_target, bounding_box_latent_source, bounding_box_latent_target):
    """_summary_

    Args:
        latent_source (_type_): _description_
        latent_target (_type_): _description_
        bounding_box_latent_source (_type_): _description_
        bounding_box_latent_target (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    latent_target[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = latent_source[:, :, y_s:y_s+height_s, x_s:x_s+width_s]
    return latent_target


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
id = 19901
seed = seeds_plus[variance_index_sorted[id]]
os.makedirs(f'pics/compete/duplicate/right/{id}', exist_ok=True)
for i, coco_class in enumerate(coco_classes):
    prompt = f"a {coco_class} on the right side"
    prompt_ = f"a {coco_class}"
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout()
    with torch.no_grad():
        out = pipe(prompt=prompt, generator=set_seed(seed))
        ax[0].imshow(out.images[0])
        ax[0].axis('off')
        out = pipe(prompt=prompt_, generator=set_seed(seed))
        ax[1].imshow(out.images[0])
        ax[1].axis('off')
    fig.savefig(f'pics/compete/duplicate/right/{id}/{coco_class}.png')