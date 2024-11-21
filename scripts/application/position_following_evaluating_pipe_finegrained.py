import sys
sys.path.append('/home/banyh2000/odfn')
from mmdet.apis import DetInferencer
import os

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

exp = 'exp2'
group = 'attend'
class_name = 'various'
class_id_dict = {0:32,1:19,2:47,3:1,4:75}
path = '/nfs/data/yuanhaoban/ODFN/position_following_finegrained/' + class_name + '/' + exp + '/' + group + '/'



names = os.listdir(path)

count = 0
total = 0
for name in names:
    seed_id = int(name.split('_')[1])
    if seed_id >= 500:
        continue
    prompt_id = int(name.split('_')[-1].split('.')[0])
    label_true = class_id_dict[prompt_id]
    
    results = inferencer(path + name)
    results = results['predictions'][0]
    total+=1
    for label,score,bbox in zip(results['labels'],results['scores'],results['bboxes']):
        if  score > 0.75 and label == label_true:
            if bbox[0]/2 + bbox[2]/2 < 256 and bbox[1]/2 + bbox[3]/2 > 256:
                count += 1
print(group)
print(count/total)
print(count)

# attend
# 0.3226022803487592
# 481

# rejection 2341 0.4 2091 0.75
# control 1596 0.4  1427 0.75
# initno  0.4  1527 0.75

# structure
# 0.3870967741935484
# 144

# llm
# 0.8983050847457628
# 53