import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.models.diffuserpipeline import StableDiffusionPipeline
from initno.initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline
import os
from scripts.utils.utils_odfn import set_seed
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
pipe_initno = StableDiffusionInitNOPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp1'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/position_following/' + class_name + '/' + exp + '/'
os.makedirs(path+f'initno',exist_ok=True)
os.makedirs(path+f'initno/images',exist_ok=True)
os.makedirs(path+f'initno/noises',exist_ok=True)


for i in range(1000):
    print(i)
    seed = torch.randint(0,1000000,(1,)).item()
    
    latent = pipe_initno.obtain_noise(prompt='An object in the left', token_indices=[2, 5], guidance_scale=7.5, num_inference_steps=50, K=1, seed=seed).to(device)

    torch.save(latent, path+f'initno/noises/seed_{i}_left.pt')
    for j, prompt in enumerate(prompts_left):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'initno/images/seed_{i}_prompt_left_{j}.png')
    
    latent = pipe_initno.obtain_noise(prompt='An object in the right', token_indices=[2, 5], guidance_scale=7.5, num_inference_steps=50, K=1, seed=seed).to(device)

    torch.save(latent, path+f'initno/noises/seed_{i}_right.pt')
    for j, prompt in enumerate(prompts_right):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'initno/images/seed_{i}_prompt_right_{j}.png')
    