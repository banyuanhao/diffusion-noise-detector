from mmdet.apis import DetInferencer
import numpy as np
import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus,set_seed,seeds_plus_spilt,return_seeds_plus_spilt
import json
from pathlib import Path
import os
path = Path('/nfs/data/yuanhaoban/ODFN/version_2/')
import torch
import matplotlib.pyplot as plt

for j in range(0,20000,100):
    image_id = variance_index_sorted[j]
    seed = seeds_plus[image_id]
    spilt = return_seeds_plus_spilt(seed)

    array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    array = array[0].cpu().numpy().transpose(1,2,0)


    annotations = []
    for spilt_ in ['train', 'val', 'test']:
        base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
        path = base_path + spilt_ + '/annotations/' + spilt_ + '_for_1_category_5_class_npy.json'
        data = json.load(open(path, 'r'))
        annotations += data['annotations']


    # Initialize the DetInferencer
    inferencer = DetInferencer(model='/nfs/data/yuanhaoban/ODFN/model/data_augmentation.py', weights='/nfs/data/yuanhaoban/ODFN/model/weights.pth',device='cuda')

    results = inferencer(array)
    results = results['predictions'][0]
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    # scale array to 0-1
    array = (array - array.min())/(array.max()-array.min())

    # axs[0].imshow(array)
    axs[0].set_title(spilt)
    
    place_holder = np.zeros((64,64,3))
    for i in range(5):
        scores = results['scores'][i]
        bbox = results['bboxes'][i]
        # axs[0].add_patch(plt.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,edgecolor='r'))
        place_holder[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:] += scores * (5-i)
    place_holder = (place_holder - place_holder.min())/(place_holder.max()-place_holder.min())
    axs[0].imshow(place_holder)
    
    place_holder = np.zeros((64,64,3))
    annotations_true = []
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            annotations_true.append(annotation)
            
    print(len(annotations_true))
            
    # axs[1].imshow(array)
    for ann in annotations_true:
        bbox = ann['bbox']
        # axs[1].add_patch(plt.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,edgecolor='r'))
        place_holder[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:] += 1
    place_holder = (place_holder - place_holder.min())/(place_holder.max()-place_holder.min())
    axs[1].imshow(place_holder)
    
    fig.savefig(f'pics/detector/5cate/{j}.png')

    print(image_id, seed, spilt)