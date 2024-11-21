import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.models.diffuserpipeline import StableDiffusionPipeline
from initno.initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline
import os
from scripts.utils.utils_odfn import set_seed
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
pipe_initno = StableDiffusionInitNOPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp1'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/position_following_finegrained/' + class_name + '/' + exp + '/'
os.makedirs(path+f'initno',exist_ok=True)
os.makedirs(path+f'initno/images',exist_ok=True)
os.makedirs(path+f'initno/noises',exist_ok=True)


for i in range(200):
    print(i)
    seed = torch.randint(0,1000000,(1,)).item()
    
    latent = pipe_initno.obtain_noise(prompt='An object in the left down corner', token_indices=[2, 5, 6], guidance_scale=7.5, num_inference_steps=50, K=1, seed=seed).to(device)

    torch.save(latent, path+f'initno/noises/seed_{i}_left_down.pt')
    for j, prompt in enumerate(prompts_left_down):
        out = pipe_initno(prompt=prompt, latents = latent, token_indices=[2, 5, 6])
        out.images[0].save(path+f'initno/images/seed_{i}_prompt_left_down_{j}.png')
    