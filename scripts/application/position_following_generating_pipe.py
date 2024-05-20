import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample_pos
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed
import numpy as np
import torch

prompts_left= [
    'a sports ball in the left',
    'a cow in the left',
    'an apple in the left',
    'a bicycle in the left',
    'a vase in the left'
]
prompts_right= [
    'a sports ball in the right',
    'a cow in the right',
    'an apple in the right',
    'a bicycle in the right',
    'a vase in the right'
]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp2'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/position_following/' + class_name + '/' + exp + '/'
os.makedirs(path+f'control',exist_ok=True)
os.makedirs(path+f'control/images',exist_ok=True)
os.makedirs(path+f'control/noises',exist_ok=True)
os.makedirs(path+f'rejection',exist_ok=True)
os.makedirs(path+f'rejection/images',exist_ok=True)
os.makedirs(path+f'rejection/noises',exist_ok=True)


for i in range(1000):
    print(i)
    seed = torch.randint(0,1000000,(1,)).item()
    
    latent = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    torch.save(latent, path+f'control/noises/seed_{i}.pt')
    for j, prompt in enumerate(prompts_left):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'control/images/seed_{i}_prompt_left_{j}.png')
    for j, prompt in enumerate(prompts_right):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'control/images/seed_{i}_prompt_right_{j}.png')
    
    latent = reject_sample_pos(therhold=0.80, position='left')
    torch.save(latent, path+f'rejection/noises/seed_{i}_left.pt')
    for j, prompt in enumerate(prompts_left):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'rejection/images/seed_{i}_prompt_left_{j}.png')
        
    latent = reject_sample_pos(therhold=0.80,position='right')
    torch.save(latent, path+f'rejection/noises/seed_{i}_right.pt')
    for j, prompt in enumerate(prompts_right):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'rejection/images/seed_{i}_prompt_right_{j}.png')
    