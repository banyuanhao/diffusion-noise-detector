from mmdet.apis import DetInferencer
import numpy as np
import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus,set_seed,seeds_plus_spilt,return_seeds_plus_spilt, detector_results,variance,variance_5_class,seeds_plus_dict
import json
from pathlib import Path
import os
path = Path('/nfs/data/yuanhaoban/ODFN/version_2/')
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# annotations = []
# for spilt_ in ['train', 'val', 'test']:
#     base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
#     path = base_path + spilt_ + '/annotations/' + spilt_ + '_for_1_category_5_class_npy.json'
#     data = json.load(open(path, 'r'))
#     annotations += data['annotations']
    
metric_1 = []
metric_2 = []
for i in range(20000):
    results = detector_results[i]['bboxes']
    results = np.array(results)
    place_holder = np.zeros((results.shape[0], 2))
    place_holder[:, 0] = (results[:, 0] + results[:, 2]) / 2
    place_holder[:, 1] = (results[:, 1] + results[:, 3]) / 2
    results = place_holder[:2,:]
    variances = np.var(results, axis=0)
    variances = np.mean(variances)
    metric_1.append(variances)
    
    results = np.array(detector_results[i]['scores'])
    results = np.mean(results[:1])
    metric_2.append(results)
    



plt.hist(metric_1, bins=100)
plt.savefig('pics/metric_1.png')

plt.hist(metric_2, bins=100)
plt.savefig('pics/metric_2.png')

plt.clf()
plt.hist(variance_5_class, bins=100)
plt.savefig('pics/odfn.png')

metric_1_index_sorted = np.argsort(metric_1)
metric_2_index_sorted = np.argsort(metric_2)
# variance_index_sorted = np.argsort(variance_5_class)

plt.clf()
plt.scatter(metric_2_index_sorted,variance_index_sorted, s=0.1)
plt.savefig('pics/odfn_vs_detector.png')

plt.clf()
plt.scatter(list(range(20000)),metric_2, label='metric_1',s=0.1)
plt.savefig('pics/metric_1.png')

metric_2 = np.array(metric_2)

place_holder = np.zeros(20000)
for i in range(20000):
    place_holder[i] = metric_2[variance_index_sorted[i]]


for i,j in zip(range(0,18000,2000),range(2000,20000,2000)):
    print(np.sum(place_holder[i:j]<0.60))    
    print('=====================')
    
    
# metric_1 = np.array(metric_1)

# place_holder = np.zeros(20000)
# for i in range(20000):
#     place_holder[i] = metric_1[variance_index_sorted[i]]


# for i,j in zip(range(0,18000,2000),range(2000,20000,2000)):
#     print(np.sum(place_holder[i:j]>150))    
#     print('=====================')