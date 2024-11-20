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
import os


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
    

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')
prompt = "A sports ball is caught in a fence."
bounding_box = [10,30,12,12]
x_t, y_t, width_t, height_t = bounding_box
values = []
areas = []

mode = 'resample'
base_image_path = f'/nfs/data/yuanhaoban/imgs/{mode}'
image_names = os.listdir(base_image_path)

for image_name in image_names:

    bounding_box_image = [value * 8 for value in bounding_box]

    out = Image.open(os.path.join(base_image_path, image_name))
    image = np.array(out)
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
    with open(f'/home/banyh2000/odfn/scripts/rebuttal/data/generalization/sd3_{mode}.json','w') as f:
        json.dump(values,f)