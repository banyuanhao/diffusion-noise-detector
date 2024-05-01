from diffusers import StableDiffusionPipeline
import sys
sys.path.append('/home/banyh2000/odfn/scripts/why/daam')
sys.path.append('/home/banyh2000/odfn')
from daam import set_seed, trace

from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict, coco_classes

from matplotlib import pyplot as plt
import torch

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

for i, name in enumerate(coco_classes):
    prompt = f"an object"

    gen = set_seed(seeds_plus[variance_index_sorted[0]])  # for reproducibility

    with torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(prompt, num_inference_steps=50, generator=gen)
            heat_map = tc.compute_global_heat_map(time_idx=[0,1,2,3,4])
            heat_map = heat_map.compute_word_heat_map('object')
            heat_map.plot_overlay(out.images[0])
            plt.savefig(f'pics/heatmap/ .png')