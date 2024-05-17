import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.models.diffuserpipeline import StableDiffusionPipeline
from initno.initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline
import os
import torch


prompts = [
    'An apple hangs from the tree branch.',
    'The solitary apple rests on a weathered windowsill.',
    'On the wooden kitchen table is a crisp apple',
    'An apple falls and lands with a thud.',
    'Sunlight falls on the apple.'
]

# prompts= [
#     'The golden sunlight filters through the dense canopy of the forest, casting dappled shadows on the moss-covered ground.',
#     'A red bicycle leans against a gnarled oak tree, its wheels slightly caked with mud from the morning’s ride.',
#     'Nearby, a picnic table is set with a checkered cloth, and atop it rests a basket filled with fresh fruit and sandwiches.',
#     'A frisbee lies forgotten on the grass, a few feet away from a sleeping dog with its fur glistening in the sun.',
#     'In the background, a kite dances in the sky, its bright colors a stark contrast against the blue expanse above.',
#     'A laptop is open on the table, displaying vibrant images of nature, momentarily abandoned for the allure of the outdoors.',
#     'A baseball glove and ball sit on the bench, remnants of a game played in the spirit of friendly competition.',
#     'A traffic cone marks the end of a nearby trail, signaling caution to the cyclists and hikers passing by.',
#     'A fire hydrant stands at the edge of the clearing, its red paint chipped but vibrant, a silent guardian of safety.',
#     'As the day wanes, the street lights begin to flicker on, their glow adding a soft luminescence to the tranquil scene.'
# ]

# token_ids = [[2,10],
#              [2,8],
#              [4,16],
#              [2,14],
#              [5,15],
#              [1,21],
#              [2,8],
#              [2,15],
#              [2,20],
#              [7,22]]
model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
pipe_initno = StableDiffusionInitNOPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

exp = 'exp1'
class_name = 'apple'


path = '/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/'
os.makedirs(path+f'initno',exist_ok=True)
os.makedirs(path+f'initno/images',exist_ok=True)
os.makedirs(path+f'initno/noises',exist_ok=True)


for i in range(1000):
    print(i)
    seed = torch.randint(0,1000000,(1,)).item()
    latent = pipe_initno.obtain_noise(prompt='A red bench and a yellow clock', token_indices=[3, 7], guidance_scale=7.5, num_inference_steps=50, K=1, seed=seed).to(device)
    torch.save(latent, path+f'initno/noises/seed_{i}.pt')
    for j, prompt in enumerate(prompts):
        out = pipe(prompt=prompt, latents = latent)
        out.images[0].save(path+f'initno/images/seed_{i}_prompt_{j}.png')