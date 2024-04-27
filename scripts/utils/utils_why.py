from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from utils_odfn import seeds_plus, variance_index_sorted,set_seed
from tqdm import tqdm
import json
import torch
import numpy as np

annotations = []
for spilt in ['train', 'val', 'test']:
    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + spilt + '_for_1_category_1_class_npy.json'
    data = json.load(open(path, 'r'))
    annotations += data['annotations']


pos_sample = []
for ann in tqdm(annotations):
    seed = seeds_plus[ann['image_id']]
    if ann['image_id'] not in variance_index_sorted[:10000]:
        continue
    bbox = ann['bbox']
    center_point = [(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2]
    if center_point[0] < 12 or center_point[0] > 52 or center_point[1] < 12 or center_point[1] > 52:
        continue
    bbox = [center_point[0]-12, center_point[1]-12, center_point[0]+12, center_point[1]+12]
    bbox = [int(i) for i in bbox]
    latent = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    sample_p = latent[0,:,bbox[0]:bbox[2],bbox[1]:bbox[3]].clone().cpu().numpy().transpose(1,2,0).flatten()
    pos_sample.append(sample_p)
        
pos_sample = np.array(pos_sample)
print(pos_sample.shape)
# 使用高斯核密度估计拟合二维数据
kde = gaussian_kde(pos_sample)

# save kde
import pickle
with open('kde.pkl', 'wb') as f:
    pickle.dump(kde, f)
    
# load kde


# 指定抽样的样本数量
sample_size = 1

# 从拟合的KDE模型中抽样
samples = kde.resample(sample_size)
print(samples.shape)