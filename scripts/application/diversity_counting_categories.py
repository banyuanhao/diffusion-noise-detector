import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os

# inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

# exp = 'exp1'
# group = 'initno'
# labels = 47
# class_name = 'various'

# path = '/nfs/data/yuanhaoban/ODFN/diversity/' + class_name + '/' + exp + '/' + group + '/images/'
# names = os.listdir(path)

# dict_bbox = {}
# labels = [0] * 80
# for i in range(300):
#     dict_bbox[i] = []
# for name in names:
#     seed_id = int(name.split('_')[1])
#     if seed_id >= 300:
#         continue
#     results = inferencer(path + name)
#     results = results['predictions'][0]
#     for label,score,bbox in zip(results['labels'],results['scores'],results['bboxes']):
#         if score > 0.75:
#             labels[label] += 1

# print(labels)

# import json
# with open('scripts/rebuttal/data/count.json', 'w') as f:
#     json.dump(labels, f)

import json
with open('scripts/rebuttal/data/count.json', 'r') as f:
    labels = json.load(f)

from scripts.utils.utils_odfn import coco_classes

class_list = [] 
for i in range(80):
    if labels[i] > 0:
        class_list.append(coco_classes[i])
print(class_list)