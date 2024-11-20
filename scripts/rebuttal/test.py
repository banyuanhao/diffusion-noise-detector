import json
with open('/nfs/data/yuanhaoban/ODFN/version_2/train/annotations/train_for_5_category_5_class_1_prompt.json') as f:
    data = json.load(f)
    
print(data['categories'])
print(data['annotations'][0])
mapping = {}
for key in data['categories']:
    mapping[key['id']] = []
    
for annotation in data['annotations']:
    mapping[annotation['category_id']].append(annotation['area'])

import numpy as np
for key in mapping:
    
    mapping[key] = np.mean(mapping[key])

for key in mapping.keys():
    print(key, mapping[key]*64)