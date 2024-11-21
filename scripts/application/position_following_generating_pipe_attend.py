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

from diffusers import StableDiffusionAttendAndExcitePipeline

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, torch_dtype=torch.float16,requires_safety_checker=False).to("cuda")
pipe = pipe.to("cuda")

exp = 'exp2'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/position_following/' + class_name + '/' + exp + '/'
os.makedirs(path+f'attend',exist_ok=True)
os.makedirs(path+f'attend/images',exist_ok=True)
os.makedirs(path+f'attend/noises',exist_ok=True)


for i in range(1000):
    print(i)
    seed = torch.randint(0,1000000,(1,)).item()

    for j, prompt in enumerate(prompts_left):
        out = pipe(prompt=prompt,  token_indices=[2,6],max_iter_to_alter=25,)
        out.images[0].save(path+f'attend/images/seed_{i}_prompt_left_{j}.png')
    for j, prompt in enumerate(prompts_right):
        out = pipe(prompt=prompt,token_indices=[2,6],max_iter_to_alter=25,)
        out.images[0].save(path+f'attend/images/seed_{i}_prompt_right_{j}.png')
    
    