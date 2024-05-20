import numpy as np
import json
import matplotlib.pyplot as plt


### 2. store the centorid of each annotations
dict_bbox = {}
dict_label = {}

for i in range(20000):
    dict_bbox[i] = []
    dict_label[i] = []

for spilt in ['train', 'val', 'test']:
    mode = '1_category_5_class_npy'
    name = spilt + '_for_1_category_5_class_npy' + '.json'

    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + name
    
    with open(path, 'r') as f:
        data = json.load(f)

    annotatinos = data['annotations']

    for ann in annotatinos:
        # dict_bbox[ann['image_id']].append([ann['bbox'][0] + ann['bbox'][2]/2, ann['bbox'][1] + ann['bbox'][3]/2])
        dict_bbox[ann['image_id']].append(ann['bbox'])
        id = ann['id']
        class_label = id // 10000000
        dict_label[ann['image_id']].append(class_label)

# num = 8
# num = 2000
# num = 4000
# num = 5008
# num = 5093
num = 5104
value = dict_bbox[num]
labels_true = dict_label[num]

print(len(value))

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
new_value = np.zeros((value.shape[0], 2))
new_value[:, 0] = value[:, 0]/2 + value[:, 2]/2
new_value[:, 1] = value[:, 1]/2 + value[:, 3]/2
value = new_value

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
labels = clusterer.fit_predict(value)

# # 使用OPTICS进行聚类
# optics = OPTICS(min_samples=30)
# labels = optics.fit_predict(value)

# # 训练模型并预测聚类标签
# labels = optics.fit_predict(value)

# 直接使用数据的前两个维度来可视化
# 指定绘制范围 64*64

# 
plt.xlim(0, 64)
plt.ylim(0, 64)
plt.scatter(value[:, 0], value[:, 1], c=labels, cmap='viridis')
plt.legend(title="Classes")
plt.savefig('cluster_true.png')


plt.clf()
plt.figure(figsize=(10,10))
plt.xlim(0, 64)
plt.ylim(0, 64)
plt.grid()
plt.scatter(value[:, 0], value[:, 1], c=labels_true, cmap='viridis')
plt.legend(title="Classes")
plt.savefig('cluster_label.png')