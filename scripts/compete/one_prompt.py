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


for i in range(0,100):
    seed_source = seeds_plus[variance_index_sorted[i]]
    seed_target = seeds_plus[variance_index_sorted[i]]

    prompt_source = "a sports ball on the right"
    prompt_target = "A sports ball"


    model_id = 'stabilityai/stable-diffusion-2-base'
    device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

    latents_source = torch.randn((1,4,64,64), generator=set_seed(seed_source), device='cuda', dtype=torch.float32)
    latents_target = torch.randn((1,4,64,64), generator=set_seed(seed_target), device='cuda', dtype=torch.float32)

    with torch.no_grad():
        fig, axs = plt.subplots(3, 2, figsize=(5*2, 5*3))
        # for ax in axs.flatten():
        #     ax.axis('off')
        # fig.tight_layout()
        
        out_source, latents_source_final = pipe(prompt=prompt_source, generator=set_seed(seed_source), latents = latents_source, output_type = "latent and pil")
        axs[0][0].imshow(out_source.images[0])
        
        latents_source_final = latents_source_final.cpu().squeeze(0).permute(1,2,0).numpy()
        # scale to 0-1
        latents_source_final = (latents_source_final - latents_source_final.min()) / (latents_source_final.max() - latents_source_final.min())
        axs[0][1].imshow(latents_source_final)
        
        out_target, latents_target_final = pipe(prompt=prompt_target, generator=set_seed(seed_target), latents = latents_target, output_type = "latent and pil")
        axs[1][0].imshow(out_target.images[0])
        
        latents_target_final = latents_target_final.cpu().squeeze(0).permute(1,2,0).numpy()
        # scale to 0-1
        latents_target_final = (latents_target_final - latents_target_final.min()) / (latents_target_final.max() - latents_target_final.min())
        axs[1][1].imshow(latents_target_final)
        
        fig.savefig(f'pics/compete/0/output_{i}.png')