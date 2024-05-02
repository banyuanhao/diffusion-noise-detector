from diffusers import StableDiffusionPipeline
import sys
sys.path.append('/home/banyh2000/odfn/scripts/why/daam')
sys.path.append('/home/banyh2000/odfn')
from daam import set_seed, trace
import os
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict, coco_classes, set_seed, auto_device

from matplotlib import pyplot as plt
import torch




def replace(latent_source, latent_target, bounding_box_latent_source, bounding_box_latent_target):
    """_summary_

    Args:
        latent_source (_type_): _description_
        latent_target (_type_): _description_
        bounding_box_latent_source (_type_): _description_
        bounding_box_latent_target (_type_): _description_

    Returns:
        _type_: _description_
    """

    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    latent_target[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = latent_source[:, :, y_s:y_s+height_s, x_s:x_s+width_s]
    return latent_target


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

for i, name in enumerate(coco_classes):
    
    seed_source = seeds_plus[variance_index_sorted[0]]
    seed_target = seeds_plus[variance_index_sorted[19900]]
    
    bounding_box_latent_source = [40,24,24,24]
    bounding_box_latent_target = [40,24,24,24]
    
    bounding_box_image_source = [value * 8 for value in bounding_box_latent_source]
    bounding_box_image_target = [value * 8 for value in bounding_box_latent_target]
    
    latents_source = torch.randn((1,4,64,64), generator=set_seed(seed_source), device='cuda', dtype=torch.float32)
    latents_target = torch.randn((1,4,64,64), generator=set_seed(seed_target), device='cuda', dtype=torch.float32)
    prompt_target = f"a {name} on the table"

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    with torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(prompt_target, num_inference_steps=50, latents = latents_target, generator=set_seed(seed_target))
            axs[0][0].imshow(out.images[0])
            axs[0][0].add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='blue', lw=2))
            
            heat_map = tc.compute_global_heat_map(time_idx=[0,1,2,3,4])
            heat_map = heat_map.compute_word_heat_map(f'{name}')
            heat_map.plot_overlay(out.images[0], ax=axs[0][1])
            
        latents_target = replace(latents_source, latents_target, bounding_box_latent_source, bounding_box_latent_target)
        with trace(pipe) as tc:
            out = pipe(prompt_target, num_inference_steps=50, latents = latents_target, generator=set_seed(seed_target))
            axs[1][0].imshow(out.images[0])
            axs[1][0].add_patch(plt.Rectangle((bounding_box_image_target[0], bounding_box_image_target[1]), bounding_box_image_target[2], bounding_box_image_target[3], fill=None, edgecolor='blue', lw=2))
            
            heat_map = tc.compute_global_heat_map(time_idx=[0,1,2,3,4])
            heat_map = heat_map.compute_word_heat_map(f'{name}')
            heat_map.plot_overlay(out.images[0], ax=axs[1][1])
            
            path = 'pics/heatmap/seed_0_pos_40_24_24_24_layer_0_5'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f'{path}/{name}.png')