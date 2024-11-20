import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import json




inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')
prompt = "A sports ball is caught in a fence."
bounding_box = [10,30,12,12]
x_t, y_t, width_t, height_t = bounding_box
values = []
areas = []

image_path = 

for i in range(200):

    bounding_box_image = [value * 8 for value in bounding_box]

    with torch.no_grad():
        
        out = pipe(prompt=prompt, latents = latents)
        image = np.array(out.images[0])
        results = inferencer(image)
        bounding_box_generated = results['predictions'][0]['bboxes'][0]       
        # compute IoU 50 between the generated bounding box and the original bounding box
        fig,ax = plt.subplots()
        ax.imshow(image)
        rect = plt.Rectangle((bounding_box_generated[0],bounding_box_generated[1]),bounding_box_generated[2]-bounding_box_generated[0],bounding_box_generated[3]-bounding_box_generated[1],linewidth=1,edgecolor='r',facecolor='none')
        areas.append((bounding_box_generated[2]-bounding_box_generated[0])*(bounding_box_generated[3]-bounding_box_generated[1])/64)
        ax.add_patch(rect)
        rect = plt.Rectangle((bounding_box_image[0],bounding_box_image[1]),bounding_box_image[2],bounding_box_image[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        # plt.savefig(f'/home/banyh2000/odfn/scripts/rebuttal/imgs/{i}.png')
        iou = Con50(bounding_box_image,bounding_box_generated)
        print(iou)
        values.append(iou)
    import json
    with open(f'/home/banyh2000/odfn/scripts/rebuttal/data/generalization/scheduler_{mode}_{width_t}_{height_t}.json','w') as f:
        json.dump(values,f)
    with open(f'/home/banyh2000/odfn/scripts/rebuttal/data/generalization/scheduler_{mode}_{width_t}_{height_t}_areas.json','w') as f:
        json.dump(areas,f)