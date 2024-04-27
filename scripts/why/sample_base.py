import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import torch
# from hyppo.ksample import KSample
# from hyppo.tools import rot_ksamp
from tqdm import tqdm
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus,set_seed
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

import numpy as np
from scipy.spatial.distance import cdist

def energy_distance(x, y):
    """计算两个数据集的能量距离"""
    x, y = np.asarray(x), np.asarray(y)
    xy_distance = np.mean(cdist(x, y, 'euclidean'))
    xx_distance = np.mean(cdist(x, x, 'euclidean'))
    yy_distance = np.mean(cdist(y, y, 'euclidean'))
    return 2 * xy_distance - xx_distance - yy_distance

def permutation_test(x, y, num_permutations=1000):
    """执行排列检验来计算能量距离的p值"""
    combined = np.vstack([x, y])
    original_distance = energy_distance(x, y)
    count = 0
    
    for _ in tqdm(range(num_permutations)):
        np.random.shuffle(combined)  # 随机打乱数据
        new_x = combined[:len(x)]
        new_y = combined[len(x):]
        permuted_distance = energy_distance(new_x, new_y)
        if permuted_distance >= original_distance:
            count += 1
    
    p_value = count / num_permutations
    return original_distance, p_value

def energy_distance(x, y):
    """ 计算能量距离 """
    x, y = np.asarray(x), np.asarray(y)
    x_distances = np.mean(cdist(x, x))
    y_distances = np.mean(cdist(y, y))
    xy_distances = np.mean(cdist(x, y))
    return 2 * xy_distances - x_distances - y_distances

annotations = []
for spilt in ['train', 'val', 'test']:
    base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
    path = base_path + spilt + '/annotations/' + spilt + '_for_1_category_1_class_npy.json'
    data = json.load(open(path, 'r'))
    annotations += data['annotations']



pos_sample = []
neg_sample = []
for ann in tqdm(annotations):
    seed = seeds_plus[ann['image_id']]

    latent = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)

    pos_x = np.random.randint(12,52)
    pos_y = np.random.randint(12,52)
    sample_p = latent[0,:,pos_x-12:pos_x+12,pos_y-12:pos_y+12].cpu().numpy().transpose(1,2,0).flatten()
    pos_sample.append(sample_p)
    
    neg_x = np.random.randint(12,52)
    neg_y = np.random.randint(12,52)
    sample_n = latent[0,:,neg_x-12:neg_x+12,neg_y-12:neg_y+12].cpu().numpy().transpose(1,2,0).flatten()
    neg_sample.append(sample_n)
    

# # 初始化KSample测试，可以选择不同的检验方法，如 "Energy" 或 "MGC"
# ks_test = KSample('mgc')

# 进行统计测试
pos_sample = np.array(pos_sample)
neg_sample = np.array(neg_sample)
print(pos_sample.shape)
print(neg_sample.shape)

pos_sample = np.unique(pos_sample, axis=0)
neg_sample = np.unique(neg_sample, axis=0)
pos_sample = pos_sample[:1000]
neg_sample = neg_sample[:1000]

print(pos_sample.shape)
print(neg_sample.shape)

# # 示例数据
# np.random.seed(0)
# x = np.random.normal(0, 1, (100, 3))
# y = np.random.normal(0, 1, (100, 3))

# 计算能量距离和p值
distance, p_value = permutation_test(neg_sample, pos_sample, 1000)
print(f"Energy Distance: {distance}")
print(f"P-value: {p_value}")
    
