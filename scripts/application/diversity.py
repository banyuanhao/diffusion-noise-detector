import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample,accept_sample,accept_sample_var,reject_sample_var,reject_sample_con
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed,auto_device,variance_5_class_index_sorted,seeds_plus

prompts_class_path = '/nfs/data/yuanhaoban/ODFN/version_2/prompts_10000_5_5'
# prompts_class_path = '/nfs/data/yuanhaoban/ODFN/version_1/prompts_brief_strict'
classes = os.listdir(prompts_class_path)
prompts_path = prompts_class_path + '/' + 'sports_ball.txt'
# open the prompt file
prompts = open(prompts_path, 'r').readlines()
prompts = [prompt.strip() for prompt in prompts[1:]]

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

# seed = reject_sample_var(150)
seed = reject_sample_con(0.55)
# seed = seeds_plus[variance_5_class_index_sorted[18800]]

for prompt in prompts:
    out = pipe(prompt=prompt, latents = seed)
    out.images[0].save(f'pics/diversity/reject/{prompt}.png')
    
# # seed = accept_sample_var(20)
# seed = accept_sample(0.85)
# # seed = seeds_plus[variance_5_class_index_sorted[1]]

# for prompt in prompts:
#     out = pipe(prompt=prompt, generator=set_seed(seed))
#     out.images[0].save(f'pics/diversity/accept/{prompt}.png')