import numpy as np
import json
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.utils.utils_odfn import variance_5_class_index_sorted, seeds_plus, variance_index_sorted

spilt = 'train'
mode = '1_category_5_class_npy'
name = spilt + '_for_' + mode + '.json'

base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
path = base_path + spilt + '/annotations/' + name

data = json.load(open(path, 'r'))

images = data['images']
annotations = data['annotations']

image = images[15070]
image_path = image['file_name']
image_id = image['id']

# image_id = variance_5_class_index_sorted[15]
image_id = variance_index_sorted[19930]
# image_id = variance_5_class_index_sorted[610]
place_holder = np.zeros((64,64),dtype=np.float32)
for ann in annotations:
    if ann['image_id'] == image_id:
        print(ann['bbox'])
        bbox = ann['bbox']
        bbox = [int(i) for i in bbox]
        place_holder[bbox[1]:bbox[3],bbox[0]:bbox[2]] += 1
        
place_holder = place_holder / np.max(place_holder)
plt.figure(figsize=(64,64))
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.imshow(place_holder)
plt.savefig('heatmap.png')

# read .npy file as nummpy
print(base_path+image_path)
img = np.load(base_path+image_path)

# do normality test on each small 4*4 patch of img
from scipy.stats import shapiro
from scipy.stats import anderson

# do normal test on latents_source and latents_target

place_holder = np.zeros((64,64),dtype=np.float32)
for i in range(0,64):
    for j in range(0,64):
        if i < 4:
            i = 4
        if j < 4:
            j = 4
        if i > 59:
            i = 59
        if j > 59:
            j = 59
        place_holder[i,j] = shapiro(img[i-4:i+4,j-4:j+4].flatten()).pvalue

place_holder = place_holder / np.max(place_holder)
plt.imshow(place_holder)
plt.savefig('place_holder.png')
        

