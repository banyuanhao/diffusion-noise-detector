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

def regenerate(latent, bounding_box, regenerate_seed):
    x, y, width, height = bounding_box
    latents_box = torch.randn((1,4,height,width), generator=set_seed(regenerate_seed), device='cuda', dtype=latent.dtype)
    latent[:, :, y:y+height, x:x+width] = latents_box
    return latent

seed = 537
regenerate_seed = 125
prompt = 'Paris Street on a rainy day'
### xy, width, height
bounding_box_latent = [20,44,30,20]
# 2723
# image = image.cpu().permute(0, 2, 3, 1).float().numpy()

# seed = 16464
# regenerate_seed = 125
# prompt = 'Office wall above a wooden desk'
# bounding_box_latent = [40,30,10,12]

# seed = 1646125
# regenerate_seed = 6236
# prompt = 'Office wall above a wooden desk'
# bounding_box_latent = [25,22,10,20]

# seed = 1646
# regenerate_seed = 1123
# prompt = 'Office wall above a wooden desk'
# bounding_box_latent = [15,43,20,17]

# seed = 1646
# regenerate_seed = 1123
# prompt = 'Office wall above a wooden desk'
# bounding_box_latent = [15,43,20,17]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
latents = pipe.get_latents(prompt='nothing', generator=set_seed(seed))
latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=latents.dtype)


with torch.no_grad():
    fig, axs = plt.subplots(2, 2, figsize=(5*2, 5*2))
    bounding_box_image = [value * 8 for value in bounding_box_latent]
    
    out, latents_final = pipe(prompt=prompt, generator=set_seed(seed), latents = latents, output_type = "latent and pil")
    axs[0][0].imshow(out.images[0])
    axs[0][0].add_patch(plt.Rectangle((bounding_box_image[0], bounding_box_image[1]), bounding_box_image[2], bounding_box_image[3], fill=None, edgecolor='red', lw=2))
    latents_final = regenerate(latents_final, bounding_box_latent, regenerate_seed)
    latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
    # scale to 0-1
    latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
    axs[0][1].imshow(latents_final)
    
    # flip the latent
    # latents = latents.flip(3)
    latents = regenerate(latents, bounding_box_latent, regenerate_seed)
    out, latents_final = pipe(prompt=prompt, generator=set_seed(seed),latents = latents, output_type = "latent and pil")
    axs[1][0].imshow(out.images[0])
    axs[1][0].add_patch(plt.Rectangle((bounding_box_image[0], bounding_box_image[1]), bounding_box_image[2], bounding_box_image[3], fill=None, edgecolor='red', lw=2))
    latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
    # scale to 0-1
    latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
    print(latents_final.shape)
    axs[1][1].imshow(latents_final)
    fig.savefig(f'pics/flip.png')