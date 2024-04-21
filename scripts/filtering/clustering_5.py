import numpy as np
import json
import matplotlib.pyplot as plt

dict_num = {}
for i in range(100):
    if i < 40:
        dict_num[i] = i
    elif i>= 80 and i < 85:
        dict_num[i] = i - 40
    elif i>= 90 and i < 95:
        dict_num[i] = i - 45
    else:
        continue

### 1. Check the number of annotations in each image
imageid_to_ann_num = np.zeros(20000, dtype=int)

for spilt in ['train', 'val', 'test']:
    mode = '1_category_5_class_npy'
    name = spilt + '_for_1_category' + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_1/'
    path = base_path + spilt + '/annotations/' + name
    with open(path, 'r') as f:
        data = json.load(f)
        
    annotatinos = data['annotations']

    for ann in annotatinos:
        imageid_to_ann_num[ann['image_id']] += 1


### 2. store the centorid of each annotations
dict_variance = {}
for i in range(50):
    dict_variance[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_5_class_npy'
    name = spilt + '_for_1_category' + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_1/'
    path = base_path + spilt + '/annotations/' + name
    
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        # dict_variance[ann['image_id']].append([ann['bbox'][0] + ann['bbox'][2]/2, ann['bbox'][1] + ann['bbox'][3]/2])
        dict_variance[dict_num[ann['image_id']]].append(ann['bbox'])

# value = dict_variance[34]
value = dict_variance[9]

# place_holder = np.zeros((64,64),dtype=np.float32)
# for bbox in value:
#     bbox = [int(i) for i in bbox]
#     place_holder[bbox[1]:bbox[3],bbox[0]:bbox[2]] += 1
    
# place_holder = place_holder / np.max(place_holder)
# plt.imshow(place_holder)
# plt.savefig('heatmap.png')

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import OPTICS

# 假设value是已经定义的四维向量数组
value = np.array(value)
value = value.reshape(-1, 4)
print(value)
new_value = np.zeros((value.shape[0], 2))
new_value[:, 0] = value[:, 0]/2 + value[:, 2]/2
new_value[:, 1] = value[:, 1]/2 + value[:, 3]/2
# value = new_value
print(max(value[:, 0]))

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
labels = clusterer.fit_predict(value)

# # 使用OPTICS进行聚类
# optics = OPTICS(min_samples=30)
# labels = optics.fit_predict(value)

# # 训练模型并预测聚类标签
# labels = optics.fit_predict(value)

# 直接使用数据的前两个维度来可视化
plt.scatter(new_value[:, 0], new_value[:, 1], c=labels, cmap='viridis')

# 保存可视化结果
plt.savefig('cluster.png')
