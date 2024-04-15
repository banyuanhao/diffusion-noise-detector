import numpy as np
import json
import matplotlib.pyplot as plt
import math

dict_annotations = {}
for i in range(20000):
    dict_annotations[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_1_class_npy'
    name = spilt + '_for_' + mode 

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        dict_annotations[ann['image_id']].append(ann['bbox'])

for key, value in dict_annotations.items():
    if len(value) == 0:
        dict_annotations[key] = 0
        continue
    value = np.array(value)
    value = np.var(value, axis=0)
    value = np.mean(value)
    dict_annotations[key] = value

### get the index
anno = np.zeros(20000)
for key, value in dict_annotations.items():
    anno[key] = value
anno_index = np.argsort(anno)

x_data = []
y_data = []
print(math.sqrt(anno[anno_index[-1]]))
for i in range(999,20000,1000):
    print(i)
    print(math.sqrt(anno[anno_index[i]]))
    x_data.append(i)
    y_data.append(math.sqrt(anno[anno_index[i]]))
plt.plot(x_data, y_data)
plt.savefig('plot.png')

# for i in range(19):
#     index_value = anno_index[:1000*i+1000]
    
#     for spilt in ['train', 'val', 'test']:
#         mode = '1_category_1_class_npy'
#         name = spilt + '_for_' + mode

#         base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
#         path = base_path + spilt + '/annotations/' + name + '.json'
#         with open(path, 'r') as f:
#             data = json.load(f)
        
#         data_ = {}
#         data_['categories'] = data['categories']
#         data_['annotations'] = []
#         data_['images'] = data['images']
                
#         print(len(data_['images']))
        
#         for ann in data['annotations']:
#             if ann['image_id'] in index_value:
#                 data_['annotations'].append(ann)
        
#         print(len(data_['annotations']))
        
#         with open(base_path + spilt + '/annotations/' + name + '_' + str(1000*i+1000) + '.json', 'w') as f:
#             json.dump(data_, f)