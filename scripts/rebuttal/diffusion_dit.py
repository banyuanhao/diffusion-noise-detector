# finetuned, unCLIP, DiffusionXL
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableUnCLIPPipeline, DiffusionPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus,set_seed, variance_5_class_index_sorted


import json

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

def generate_patch_sin(size):
    s_x, s_y, s_z = size
    x, y, z = torch.meshgrid(torch.linspace(-1, 1, s_x), torch.linspace(-1, 1, s_y), torch.linspace(-1, 1, s_z))
    mat_x = torch.sin( 1 * torch.pi * x)
    mat_y = torch.sin( 1 * torch.pi * y)
    mat_z = torch.sin( 1 * torch.pi * z)
    result = mat_x + mat_y + mat_z
    return result.unsqueeze(0).cuda()

def generate_patch_gaussian(size, mean = 0, std = 1, seed = None):
    if seed is None:
        seed = torch.randint(0,1000000,(1,)).item()
    s_x, s_y, s_z = size
    result = torch.randn((1, s_x, s_y, s_z), generator=set_seed(seed), device='cuda') * std + mean
    return result.unsqueeze(0).cuda()

# compute if the p percent of the generated bounding box is in the original bounding box
def Con50(bounding_box_1,bounding_box_2):
    bounding_box_2 = [bounding_box_2[0],bounding_box_2[1],bounding_box_2[2]-bounding_box_2[0],bounding_box_2[3]-bounding_box_2[1]]
    x1 = max(bounding_box_1[0], bounding_box_2[0])
    y1 = max(bounding_box_1[1], bounding_box_2[1])
    x2 = min(bounding_box_1[0] + bounding_box_1[2], bounding_box_2[0] + bounding_box_2[2])
    y2 = min(bounding_box_1[1] + bounding_box_1[3], bounding_box_2[1] + bounding_box_2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    iou = intersection / (bounding_box_1[2] * bounding_box_1[3])
    return iou
    

def IoU50(bounding_box_1,bounding_box_2):
    # compute IoU 50 between the generated bounding box and the original bounding box
    bounding_box_2 = [bounding_box_2[0],bounding_box_2[1],bounding_box_2[2]-bounding_box_2[0],bounding_box_2[3]-bounding_box_2[1]]
    x1 = max(bounding_box_1[0], bounding_box_2[0])
    y1 = max(bounding_box_1[1], bounding_box_2[1])
    x2 = min(bounding_box_1[0] + bounding_box_1[2], bounding_box_2[0] + bounding_box_2[2])
    y2 = min(bounding_box_1[1] + bounding_box_1[3], bounding_box_2[1] + bounding_box_2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    union = bounding_box_1[2] * bounding_box_1[3] + bounding_box_2[2] * bounding_box_2[3] - intersection
    iou = intersection / union
    return iou


def get_patch_natural(num=0):
    bounding_box = [40,27,24,24]
    
    # resize bounding_box to a fixed width and height 24x24
    seed = seeds_plus[variance_index_sorted[num]]
    latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    patch = latents[:, :,bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]].clone()
    return patch

num = 0
device = 'cuda'

mode = ['resample', 'shift gaussian', 'functional', 'natural']
mode = mode[0]

from diffusers import DiffusionPipeline, DiTPipeline
model = 'DiT-XL-2-512'
model_id = "facebook/DiT-XL-2-512"
pipe = DiffusionPipeline.from_pretrained(model_id)

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

prompt = "A sports ball is caught in a fence."
bounding_box = [10,30,24,24]
x_t, y_t, width_t, height_t = bounding_box
theta = 8
theta = theta / 100 * np.pi / 2
mean = 0
std = 0.9

values = []
for i in range(200):
    print(i)
    seed = i

    latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    bounding_box_image = [value * 8 for value in bounding_box]

    with torch.no_grad():
        
        if mode == 'resample':
            patch = generate_patch_gaussian((4,height_t, width_t), std = 1.0, mean = 0)
            latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = patch
        elif mode == 'shift gaussian':
            patch = generate_patch_gaussian((4,height_t, width_t), std = std, mean = 0)
            latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = patch
        elif mode == 'functional':
            patch = generate_patch_sin((4,height_t, width_t))
            latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = patch * np.sin(theta) + np.cos(theta) * latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t]
        elif mode == 'natural':
            patch = get_patch_natural(num)
            latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = patch
        else:
            raise ValueError('mode not recognized')
        
        # out = pipe(latents = latents)
        out = pipe(latents = latents, class_labels = [0])
        image = np.array(out.images[0])
        results = inferencer(image)
        bounding_box_generated = results['predictions'][0]['bboxes'][0]       
        # compute IoU 50 between the generated bounding box and the original bounding box
        fig,ax = plt.subplots()
        ax.imshow(image)
        rect = plt.Rectangle((bounding_box_generated[0],bounding_box_generated[1]),bounding_box_generated[2]-bounding_box_generated[0],bounding_box_generated[3]-bounding_box_generated[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        rect = plt.Rectangle((bounding_box_image[0],bounding_box_image[1]),bounding_box_image[2],bounding_box_image[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        plt.savefig(f'/home/banyh2000/odfn/scripts/rebuttal/imgs/{i}.png')
        iou = Con50(bounding_box_image,bounding_box_generated)
        print(iou)
        values.append(iou)
    import json
    with open(f'/home/banyh2000/odfn/scripts/rebuttal/data/generalization/model_{model}_{mode}.json','w') as f:
        json.dump(values,f)
