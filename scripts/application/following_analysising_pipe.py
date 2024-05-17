import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os
import numpy as np

base_path = '/nfs/data/yuanhaoban/ODFN/following/'
exp = 'exp2'
group = 'control'
labels = 32
class_name = 'sports_ball'


import json
with open(base_path + class_name + '/' + exp + '/' + group + '/bboxes.json', 'r') as f:
    data = json.load(f)
    
variances = []
for key, bboxes in data.items():
    if len(bboxes) == 0:
        continue
    array = np.array(bboxes)
    place_holder = np.zeros((array.shape[0], 2))
    place_holder[:, 0] = (array[:, 0] + array[:, 2]) / 2
    place_holder[:, 1] = (array[:, 1] + array[:, 3]) / 2
    variance = np.var(array, axis=0)
    variance = np.mean(variance)
    variances.append(variance)
    
variances = np.array(variances)
print(np.mean(variances))