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
    mode = '1_category_5_class_npy'
    name = spilt + '_for_' + mode + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)
    annotatinos = data['annotations']

    for ann in annotatinos:
        imageid_to_ann_num[ann['image_id']] += 1


### 2. store the centorid of each annotations
dict_variance = {}
dict_mean = {}
for i in range(20000):
    dict_variance[i] = []
    dict_mean[i] = []

for spilt in ['train', 'val','test']:
    mode = '1_category_1_class_npy'
    name = spilt + '_for_' + mode + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        dict_variance[ann['image_id']].append([ann['bbox'][0]/2 + ann['bbox'][2]/2, ann['bbox'][1]/2 + ann['bbox'][3]/2])
        #dict_variance[ann['image_id']].append([ann['bbox']])

for key, value in dict_variance.items():
    if len(value) == 0:
        dict_variance[key] = 0
        continue
    value = np.array(value)
    mean = np.mean(value, axis=0)
    value = np.var(value, axis=0)
    value = np.mean(value)
    dict_variance[key] = value
    dict_mean[key] = mean

left_upper = 0
right_upper = 0
left_lower = 0
right_lower = 0

for i in range(20000):
    if dict_variance[i] > 1 or dict_variance[i] == 0:
        continue
    if dict_mean[i][0] < 32 and dict_mean[i][1] < 32:
        left_upper += 1
    elif dict_mean[i][0] > 32 and dict_mean[i][1] < 32:
        right_upper += 1
    elif dict_mean[i][0] < 32 and dict_mean[i][1] > 32:
        left_lower += 1
    else:
        right_lower += 1
total = left_upper + right_upper + left_lower + right_lower
print(left_upper, right_upper, left_lower, right_lower)
print(left_upper/total, right_upper/total, left_lower/total, right_lower/total)