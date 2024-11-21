import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample_pos_finegrained
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed
import numpy as np
import torch

prompts_left_down= [
    'a sports ball in the left down corner',
    'a cow in the left down corner',
    'an apple in the left down corner',
    'a bicycle in the left down corner',
    'a vase in the left down corner',
]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp2'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/position_following_finegrained/' + class_name + '/' + exp + '/'
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
    for j, prompt in enumerate(prompts_left_down):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'control/images/seed_{i}_prompt_left_down_{j}.png')
    
    latent = reject_sample_pos_finegrained(therhold=0.80, position='left_down')
    torch.save(latent, path+f'rejection/noises/seed_{i}.pt')
    for j, prompt in enumerate(prompts_left_down):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'rejection/images/seed_{i}_prompt_left_down_{j}.png')
    