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
import json
from tqdm import tqdm

annotations = []
for spilt_ in ['train', 'val', 'test']:
    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt_ + '/annotations/' + spilt_ + '_for_1_category_5_class_npy.json'
    data = json.load(open(path, 'r'))
    annotations += data['annotations']

# Initialize the DetInferencer
inferencer = DetInferencer(model='/nfs/data/yuanhaoban/ODFN/model/data_augmentation.py', weights='/nfs/data/yuanhaoban/ODFN/model/weights.pth',device='cuda')




# results_dict = []
# for j in tqdm(range(20000)):
#     image_id = j
#     seed = seeds_plus[image_id]
#     spilt = return_seeds_plus_spilt(seed)

#     array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
#     array = array[0].cpu().numpy().transpose(1,2,0)


#     results = inferencer(array)
#     results = results['predictions'][0]
#     results_dict.append(results)
    
# with open('scripts/utils/detector_results.json', 'w') as f:
#     json.dump(results_dict, f)


