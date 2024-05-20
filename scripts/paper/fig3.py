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
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict, coco_classes

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


seed_source = seeds_plus[variance_index_sorted[19910]]
seed_target = seeds_plus[variance_index_sorted[19910]]

prompt_source = "A sports ball appearing in the left"
prompt_target = "A sports ball"

# seed_source = seeds_plus[variance_index_sorted[0]]
# seed_target = seeds_plus[variance_index_sorted[0]]

# prompt_source = "A sports ball appearing in the left"
# prompt_target = "A sports ball"


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

latents_source = torch.randn((1,4,64,64), generator=set_seed(seed_source), device='cuda', dtype=torch.float32)
latents_target = torch.randn((1,4,64,64), generator=set_seed(seed_target), device='cuda', dtype=torch.float32)

with torch.no_grad():
    
    out_source, _ = pipe(prompt=prompt_source, generator=set_seed(seed_source), latents = latents_source, output_type = "latent and pil")
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(out_source.images[0])
    plt.savefig(f'pics/paper/fig3/low_ori.png')
    
    
    out_target, _ = pipe(prompt=prompt_target, generator=set_seed(seed_target), latents = latents_target, output_type = "latent and pil")
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(out_target.images[0])
    plt.savefig(f'pics/paper/fig3/low_mod.png')
    