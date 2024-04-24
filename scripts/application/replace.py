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
    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    latent_target[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = latent_source[:, :, y_s:y_s+height_s, x_s:x_s+width_s]
    return latent_target

def self_replace(latent, bounding_box_latent_source, bounding_box_latent_target):
    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    tmp = latent[:, :, y_t:y_t+height_t, x_t:x_t+width_t].clone()
    latent[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = latent[:, :, y_s:y_s+height_s, x_s:x_s+width_s]
    latent[:, :, y_s:y_s+height_s, x_s:x_s+width_s] = tmp
    return latent

def self_paste(latent, bounding_box_latent_source, bounding_box_latent_target):
    """_summary_

    Args:
        latent (_type_): latent
        bounding_box_latent_source (_type_): paste source red
        bounding_box_latent_target (_type_): paste target blue

    Returns:
        _type_: _description_
    """
    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    tmp = latent[:, :, y_s:y_s+height_s, x_s:x_s+width_s].clone()
    latent[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = tmp
    return latent

# print(variance_index_sorted[0])
# print(seeds_plus[variance_index_sorted[0]])
seed = seeds_plus[variance_index_sorted[0]]
prompt = "A sports ball is caught in a fence."
bounding_box_latent_source = [40,20,24,30]
bounding_box_latent_target = [0,20,24,30]

# bounding_box_latent_source = [40,0,24,30]
# bounding_box_latent_target = [0,34,24,30]
# seed = seeds_plus[variance_index_sorted[1]]
# seed = seeds_plus[variance_index_sorted[0]]
# prompt = "The baseball glove waits by the fence."
# bounding_box_latent_source = [30,15,34,40]
# bounding_box_latent_target = [00,20,34,40]

# seed = seeds_plus[variance_index_sorted[0]]
# prompt = "A red stop sign halts traffic at an intersection."
# bounding_box_latent_source = [30,15,34,40]
# bounding_box_latent_target = [00,20,34,40]


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

latents = pipe.get_latents(prompt='nothing', generator=set_seed(seed))
latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=latents.dtype)

with torch.no_grad():
    fig, axs = plt.subplots(2, 2, figsize=(5*2, 5*2))
    bounding_box_image_source = [value * 8 for value in bounding_box_latent_source]
    bounding_box_image_target = [value * 8 for value in bounding_box_latent_target]
    
    out, latents_final = pipe(prompt=prompt, generator=set_seed(seed), latents = latents, output_type = "latent and pil")
    axs[0][0].imshow(out.images[0])
    axs[0][0].add_patch(plt.Rectangle((bounding_box_image_source[0], bounding_box_image_source[1]), bounding_box_image_source[2], bounding_box_image_source[3], fill=None, edgecolor='red', lw=2))
    axs[0][0].add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='blue', lw=2))
    
    latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
    # scale to 0-1
    latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
    axs[0][1].add_patch(plt.Rectangle((bounding_box_latent_source[0], bounding_box_latent_source[1]), bounding_box_latent_source[2], bounding_box_latent_source[3], fill=None, edgecolor='red', lw=2))
    axs[0][1].add_patch(plt.Rectangle((bounding_box_latent_target[0], bounding_box_latent_target[1]), bounding_box_latent_target[2], bounding_box_latent_target[3], fill=None, edgecolor='blue', lw=2))
    axs[0][1].imshow(latents_final)
    
    # flip the latent
    # latents = latents.flip(3)
    latents = self_replace(latents, bounding_box_latent_source, bounding_box_latent_target)
    
    out, latents_final = pipe(prompt=prompt, generator=set_seed(seed), latents = latents, output_type = "latent and pil")
    axs[1][0].imshow(out.images[0])
    axs[1][0].add_patch(plt.Rectangle((bounding_box_image_source[0], bounding_box_image_source[1]), bounding_box_image_source[2], bounding_box_image_source[3], fill=None, edgecolor='red', lw=2))
    axs[1][0].add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='blue', lw=2))
    
    latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
    # scale to 0-1
    latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
    axs[1][1].add_patch(plt.Rectangle((bounding_box_latent_source[0], bounding_box_latent_source[1]), bounding_box_latent_source[2], bounding_box_latent_source[3], fill=None, edgecolor='red', lw=2))
    axs[1][1].add_patch(plt.Rectangle((bounding_box_latent_target[0], bounding_box_latent_target[1]), bounding_box_latent_target[2], bounding_box_latent_target[3], fill=None, edgecolor='blue', lw=2))
    axs[1][1].imshow(latents_final)
    
    fig.savefig(f'pics/flip.png')