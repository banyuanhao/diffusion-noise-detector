import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

exp = 'exp1'
group = 'rejection'
labels = 32
class_name = 'sports_ball'

path = '/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/' + group + '/images/'
names = os.listdir(path)

dict_bbox = {}
for i in range(800):
    dict_bbox[i] = []
for name in names:
    seed_id = int(name.split('_')[1])
    if seed_id >= 800:
        continue
    results = inferencer(path + name)
    results = results['predictions'][0]
    for label,score,bbox in zip(results['labels'],results['scores'],results['bboxes']):
        if label == 32 and score > 0.75:
            dict_bbox[int(seed_id)].append(bbox)
            break

import json
with open('/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/' + group + '/bboxes.json', 'w') as f:
    json.dump(dict_bbox, f)