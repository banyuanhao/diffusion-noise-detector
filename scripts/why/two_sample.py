import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus,set_seed

annotations = []
for spilt in ['train', 'val', 'test']:
    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + spilt + '_for_1_category_1_class_npy.json'
    data = json.load(open(path, 'r'))
    annotations += data['annotations']

    data = json.load(open(path, 'r'))

# do normal test on latents_source and latents_target

import numpy as np
from scipy import stats
from tqdm import tqdm

value = []
# for i in tqdm(range(0,20000,100)):
image_id = variance_index_sorted[19999]
image_seed = seeds_plus[image_id]
img = torch.randn((1,4,64,64), generator=set_seed(image_seed), device='cuda', dtype=torch.float32)
img = img[0].cpu().numpy().transpose(1,2,0)
ann_count = 0
count = 0
for ann in annotations:
    if ann['image_id'] == image_id:
        ann_count += 1
        bbox = ann['bbox']
        bbox = [int(i) for i in bbox]
        sample_trigger = img[bbox[1]:bbox[3],bbox[0]:bbox[2]].flatten()
        #sample_trigger = img[bbox[0]:bbox[2],bbox[1]:bbox[3]].flatten()
        
        # sample_random = np.random.randn(bbox[2]-bbox[0],bbox[3]-bbox[1]).flatten()
        
        for i in range(1000):
            # random choose a patch in img with the shape the same as sample_trigger
            sample_random = np.random.randn(bbox[3]-bbox[1],bbox[2]-bbox[0]).flatten()
            # first = np.random.randint(0,64-bbox[3]+bbox[1])
            # second = np.random.randint(0,64-bbox[2]+bbox[0])
            # sample_random = img[first:first+bbox[3]-bbox[1],second:second+bbox[2]-bbox[0],:].flatten()
            
            p_value = stats.anderson_ksamp([sample_trigger, sample_random]).significance_level
            if p_value < 0.1:
                count += 1
print(count/ann_count)
# value.append(count/ann_count)
    
# import json
# with open('ks.json', 'w') as f:
#     json.dump(value, f)
    
# plt.plot(value)
# plt.savefig('pics/ks.png')