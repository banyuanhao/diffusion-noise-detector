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
for i in range(20000):
    dict_variance[i] = []

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
    value = np.var(value, axis=0)
    value = np.mean(value)
    dict_variance[key] = value

with open('dict_variance_5_class.json', 'w') as f:
    json.dump(dict_variance, f)
### 3. plot the histogram
print(len(list(dict_variance.values())))
sort = sorted(list(dict_variance.values()))

bar = 64*64*0.1*0.1
for i in range(20000):
    if sort[i] > bar:
        print(i/20000)
        break
print(sort[10000])

plt.hist(list(dict_variance.values()), bins=100)

plt.savefig('pics/variance_5.png')