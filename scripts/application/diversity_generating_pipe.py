import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample,accept_sample,accept_sample_var,reject_sample_var,reject_sample_con
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed,auto_device,variance_5_class_index_sorted,seeds_plus
import numpy as np
import torch


prompts = [
    'An apple hangs from the tree branch.',
    'The solitary apple rests on a weathered windowsill.',
    'On the wooden kitchen table is a crisp apple',
    'An apple falls and lands with a thud.',
    'Sunlight falls on the apple.'
]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp1'
group = 'rejection'
labels = 32
class_name = 'apple'


path = '/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/'
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
    for j, prompt in enumerate(prompts):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'control/images/seed_{i}_prompt_{j}.png')
    
    latent = reject_sample_con(therhold=0.6,seed=seed)
    torch.save(latent, path+f'rejection/noises/seed_{i}.pt')
    for j, prompt in enumerate(prompts):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'rejection/images/seed_{i}_prompt_{j}.png')
    