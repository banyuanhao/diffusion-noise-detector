import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import torch
# from hyppo.ksample import KSample
# from hyppo.tools import rot_ksamp
from tqdm import tqdm
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus,set_seed
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

import numpy as np
from scipy.spatial.distance import cdist

annotations = []
for spilt in ['train', 'val', 'test']:
    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + spilt + '_for_1_category_1_class_npy.json'
    data = json.load(open(path, 'r'))
    annotations += data['annotations']

mean_list = []
var_list = []
for i in range(0,20000,2000):
    print(i, i+2000)
    pos_sample = []
    neg_sample = []
    for ann in tqdm(annotations):
        seed = seeds_plus[ann['image_id']]
        if ann['image_id'] not in variance_index_sorted[i:i+2000]:
            continue
        bbox = ann['bbox']
        center_point = [(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2]
        if center_point[0] < 12 or center_point[0] > 52 or center_point[1] < 12 or center_point[1] > 52:
            continue
        bbox = [center_point[0]-12, center_point[1]-12, center_point[0]+12, center_point[1]+12]
        bbox = [int(i) for i in bbox]
        latent = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
        sample_p = latent[0,:,bbox[0]:bbox[2],bbox[1]:bbox[3]].clone().cpu().numpy().transpose(1,2,0).flatten()
        pos_sample.append(sample_p)
        
        neg_x = np.random.randint(12,52)
        neg_y = np.random.randint(12,52)
        sample_n = latent[0,:,neg_x-12:neg_x+12,neg_y-12:neg_y+12].cpu().numpy().transpose(1,2,0).flatten()
        neg_sample.append(sample_n)
        

    # # 初始化KSample测试，可以选择不同的检验方法，如 "Energy" 或 "MGC"
    # ks_test = KSample('mgc')

    # 进行统计测试
    pos_sample = np.array(pos_sample)
    neg_sample = np.array(neg_sample)
    print(pos_sample.shape)
    print(neg_sample.shape)

    pos_sample = np.unique(pos_sample, axis=0)
    neg_sample = np.unique(neg_sample, axis=0)
    pos_sample = pos_sample[:2000]
    neg_sample = neg_sample[:2000]
    print('======================')
    print(np.abs(np.mean(pos_sample)))
    print(np.abs(np.mean(neg_sample)))
    print(np.abs(np.var(pos_sample)-1))
    print(np.abs(np.var(neg_sample)-1))
    mean_list.append(np.abs(np.mean(pos_sample))-np.abs(np.mean(neg_sample)))
    var_list.append(np.abs(np.var(pos_sample)-1)-np.abs(np.var(neg_sample)-1))

plt.plot(mean_list)
plt.savefig('mean.png')
plt.close()
plt.plot(var_list)
plt.savefig('var.png')
plt.close()
