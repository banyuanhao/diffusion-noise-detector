import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample,accept_sample,accept_sample_var,reject_sample_var,reject_sample_con
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed,auto_device,variance_5_class_index_sorted,seeds_plus
import numpy as np
import torch


# prompts = [
#     'An apple hangs from the tree branch.',
#     'The solitary apple rests on a weathered windowsill.',
#     'On the wooden kitchen table is a crisp apple',
#     'An apple falls and lands with a thud.',
#     'Sunlight falls on the apple.'
# ]

prompts= [
    'The golden sunlight filters through the dense canopy of the forest, casting dappled shadows on the moss-covered ground.',
    'A red bicycle leans against a gnarled oak tree, its wheels slightly caked with mud from the morning’s ride.',
    'Nearby, a picnic table is set with a checkered cloth, and atop it rests a basket filled with fresh fruit and sandwiches.',
    'A frisbee lies forgotten on the grass, a few feet away from a sleeping dog with its fur glistening in the sun.',
    'In the background, a kite dances in the sky, its bright colors a stark contrast against the blue expanse above.',
    'A laptop is open on the table, displaying vibrant images of nature, momentarily abandoned for the allure of the outdoors.',
    'A baseball glove and ball sit on the bench, remnants of a game played in the spirit of friendly competition.',
    'A traffic cone marks the end of a nearby trail, signaling caution to the cyclists and hikers passing by.',
    'A fire hydrant stands at the edge of the clearing, its red paint chipped but vibrant, a silent guardian of safety.',
    'As the day wanes, the street lights begin to flicker on, their glow adding a soft luminescence to the tranquil scene.'
]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp1'
class_name = 'various'


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
    