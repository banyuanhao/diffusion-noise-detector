import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

exp = 'exp1'
group = 'control'
class_name = 'various'
class_id_dict = {0:32,1:19,2:47,3:1,4:75}
path = '/nfs/data/yuanhaoban/ODFN/following/' + class_name + '/' + exp + '/' + group + '/images/'
names = os.listdir(path)

count = 0
for name in names:
    left = 'left' in name
    seed_id = int(name.split('_')[1])
    if seed_id >= 100:
        continue
    prompt_id = int(name.split('_')[-1].split('.')[0])
    label_true = class_id_dict[prompt_id]
    
    results = inferencer(path + name)
    results = results['predictions'][0]
    
    for label,score,bbox in zip(results['labels'],results['scores'],results['bboxes']):
        if  score > 0.4 and label == label_true:
            if bbox[0]/2 + bbox[2]/2 < 256 and left:
                count += 1
            if bbox[0]/2 + bbox[2]/2 > 256 and not left:
                count += 1
            break
print(group)
print(count)

# rejection 373