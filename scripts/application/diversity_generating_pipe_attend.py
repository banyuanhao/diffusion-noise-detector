import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.detector.detector_resampling import reject_sample,accept_sample,accept_sample_var,reject_sample_var,reject_sample_con
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import os
from scripts.utils.utils_odfn import set_seed,auto_device,variance_5_class_index_sorted,seeds_plus
import numpy as np
import torch


from diffusers import StableDiffusionAttendAndExcitePipeline

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, torch_dtype=torch.float16,requires_safety_checker=False).to("cuda")
pipe = pipe.to("cuda")

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

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_id, torch_dtype=torch.float16,requires_safety_checker=False).to("cuda")
pipe = pipe.to("cuda")

exp = 'exp1'
class_name = 'various'


path = '/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/'
os.makedirs(path+f'attend/images',exist_ok=True)


for i in range(1000):
    for j, prompt in enumerate(prompts):
        out = pipe(prompt=prompt,  token_indices=[2,6],max_iter_to_alter=25,)
        out.images[0].save(path+f'attend/images/seed_{i}_prompt_{j}.png')
    