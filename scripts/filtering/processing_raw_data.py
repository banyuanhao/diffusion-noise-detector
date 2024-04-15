import numpy as np
import json
import matplotlib.pyplot as plt

### 2. store the centorid of each annotations
dict_annotations_1 = {}
for i in range(20000):
    dict_annotations_1[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_1_class_npy'
    name = spilt + '_for_' + mode + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        dict_annotations_1[ann['image_id']].append(ann['bbox'])

for key, value in dict_annotations_1.items():
    if len(value) == 0:
        print(key)
        dict_annotations_1[key] = 0
        continue
    value = np.array(value)
    value = np.var(value, axis=0)
    value = np.mean(value)
    dict_annotations_1[key] = value

anno_1 = np.zeros(20000)
for key, value in dict_annotations_1.items():
    anno_1[key] = value

### 2. store the centorid of each annotations
dict_annotations = {}
for i in range(20000):
    dict_annotations[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_5_class_npy'
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
    
anno_5 = np.zeros(20000)
for key, value in dict_annotations.items():
    anno_5[key] = value
    
    
index_1 = np.argsort(anno_1)
index_5 = np.argsort(anno_5)
print(index_1[:100])
print(index_5[:100])
list_value = []
for i in range(20):
    print(len(set(index_1[:(i+1)*1000]) & set(index_5[:(i+1)*1000]))/1000/(i+1))
    
    list_value.append(len(set(index_1[:i*1000]) & set(index_5[:i*1000]))/1000/(i+1))

plt.plot(list_value)
plt.savefig('test.png')