from diffusers import StableDiffusionPipeline
from daam import set_seed, trace


from matplotlib import pyplot as plt
import torch

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
prompt = 'A dog runs across the field'
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=50, generator=gen)
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('dog')
        heat_map.plot_overlay(out.images[0])
        plt.savefig('pics/heat_map.png')