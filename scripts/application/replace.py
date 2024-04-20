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


# print(variance_index_sorted[0])
# print(min(variance_index_sorted))
print(seeds_plus[variance_index_sorted[0]])
seed = seeds_plus[variance_index_sorted[0]]
prompt = "A sports ball is caught in a fence."


# bounding_box_latent_source = [15,43,20,17]
# bounding_box_latent_target = [15,43,20,17]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
out = pipe(prompt=prompt, generator=set_seed(seed), num_inference_steps=30)
fig, axs = plt.subplots(2, 3, figsize=(5*2, 5*3))
axs[0][0].imshow(out.images[0])
fig.savefig(f'pics/regenerate.png')

# latents = pipe.get_latents(prompt='nothing', generator=set_seed(seed))
# latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=latents.dtype)


# with torch.no_grad():
#     fig, axs = plt.subplots(3, 2, figsize=(5*3, 5*2))
#     bounding_box_image = [value * 8 for value in bounding_box_latent_source]
    
#     out, latents_final = pipe(prompt=prompt, generator=set_seed(seed), latents = latents, output_type = "latent and pil")
#     axs[0][0].imshow(out.images[0])
#     axs[0][0].add_patch(plt.Rectangle((bounding_box_image[0], bounding_box_image[1]), bounding_box_image[2], bounding_box_image[3], fill=None, edgecolor='red', lw=2))
#     latents_final = replace(latents_final, bounding_box_latent_source, regenerate_seed)
#     latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
#     # scale to 0-1
#     latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
#     axs[0][1].imshow(latents_final)
    
#     # flip the latent
#     # latents = latents.flip(3)
#     latents = replace(latents, bounding_box_latent, regenerate_seed)
#     out, latents_final = pipe(prompt=prompt, generator=set_seed(seed),latents = latents, output_type = "latent and pil")
#     axs[1][0].imshow(out.images[0])
#     axs[1][0].add_patch(plt.Rectangle((bounding_box_image[0], bounding_box_image[1]), bounding_box_image[2], bounding_box_image[3], fill=None, edgecolor='red', lw=2))
#     latents_final = latents_final.cpu().squeeze(0).permute(1,2,0).numpy()
#     # scale to 0-1
#     latents_final = (latents_final - latents_final.min()) / (latents_final.max() - latents_final.min())
#     print(latents_final.shape)
#     axs[1][1].imshow(latents_final)
#     fig.savefig(f'pics/flip.png')