import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os
import numpy as np

base_path = '/nfs/data/yuanhaoban/ODFN/diversity/'
exp = 'exp1'
# group = 'llm' 102.097
# group = 'attend' 145.16
# group = 'structure' 133.92

labels = 47
class_name = 'various'


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
print(np.mean(variances)/64)

# 171 135