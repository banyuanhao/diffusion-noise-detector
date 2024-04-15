import numpy as np
import json
import matplotlib.pyplot as plt

# spilt = 'val'
# mode = '1_category_1_class_npy'
# name = spilt + '_for_' + mode + '.json'

# base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
# path = base_path + spilt + '/annotations/' + name

### 1. Check the number of annotations in each image
imageid_to_ann_num = np.zeros(20000, dtype=int)

for spilt in ['train', 'val', 'test']:
    mode = '1_category_1_class_npy'
    name = spilt + '_for_' + mode + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)
        
    annotatinos = data['annotations']

    for ann in annotatinos:
        imageid_to_ann_num[ann['image_id']] += 1


### 2. store the centorid of each annotations
dict_annotations = {}
for i in range(20000):
    dict_annotations[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_1_class_npy'
    name = spilt + '_for_' + mode + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        dict_annotations[ann['image_id']].append(ann['bbox'])

for key, value in dict_annotations.items():
    if len(value) == 0:
        print(key)
        dict_annotations[key] = 0
        continue
    value = np.array(value)
    value = np.var(value, axis=0)
    value = np.mean(value)
    dict_annotations[key] = value

### 3. plot the histogram
plt.hist(list(dict_annotations.values()), bins=100)
plt.savefig('variance.png')