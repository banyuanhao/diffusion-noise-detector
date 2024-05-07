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
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict

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

name = 'ball_position'
# seed_source = seeds_plus[variance_index_sorted[0]]
# seed_target = seeds_plus[variance_index_sorted[19040]]

# prompt_source = "A sports ball is caught in a fence."
# prompt_target = "A sports ball is caught in a fence."

# bounding_box_latent_source = [40,20,24,30]
# bounding_box_latent_target = [10,10,24,30]

# seed_source = seeds_plus[variance_index_sorted[0]]
# seed_target = seeds_plus[variance_index_sorted[19040]]

# prompt_source = "A grizzly bear fishes in a rushing river."
# prompt_target = "A grizzly bear fishes in a rushing river."

# bounding_box_latent_source = [40,20,24,30]
# bounding_box_latent_target = [10,10,24,30]

seed_source = seeds_plus[variance_index_sorted[0]]
seed_target = seeds_plus[variance_index_sorted[19040]]

prompt_source = "A sports ball is caught in a fence."
prompt_target = "A sports ball is caught in a fence."

bounding_box_latent_source = [40,20,24,30]
bounding_box_latent_target = [40,10,24,30]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

latents_source = torch.randn((1,4,64,64), generator=set_seed(seed_source), device='cuda', dtype=torch.float32)
latents_target = torch.randn((1,4,64,64), generator=set_seed(seed_target), device='cuda', dtype=torch.float32)


with torch.no_grad():

    bounding_box_image_source = [value * 8 for value in bounding_box_latent_source]
    bounding_box_image_target = [value * 8 for value in bounding_box_latent_target]
    
    
    out_target, _ = pipe(prompt=prompt_target, generator=set_seed(seed_target), latents = latents_target, output_type = "latent and pil")
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(out_target.images[0])
    # add patch in image
    plt.gca().add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='red', lw=5))
    plt.savefig(f'pics/paper/fig1/{name}_ori.png')
    
    latents_target = replace(latents_source, latents_target, bounding_box_latent_source, bounding_box_latent_target)
    
    out_target, _ = pipe(prompt=prompt_target, generator=set_seed(seed_target), latents = latents_target, output_type = "latent and pil")
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(out_target.images[0])
    plt.gca().add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='red', lw=5))
    plt.savefig(f'pics/paper/fig1/{name}_mod.png')
    
    
    latents_target = latents_target.cpu().squeeze(0).permute(1,2,0).numpy()
    # scale to 0-1
    latents_target = (latents_target - latents_target.min()) / (latents_target.max() - latents_target.min())
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().add_patch(plt.Rectangle((bounding_box_latent_target[0], bounding_box_latent_target[1]), bounding_box_latent_target[2], bounding_box_latent_target[3], fill=None, edgecolor='red', lw=5))
    plt.imshow(latents_target)
        
    plt.savefig(f'pics/paper/fig1/{name}_latent.png')