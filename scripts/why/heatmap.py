import numpy as np
import json
import matplotlib.pyplot as plt

spilt = 'train'
mode = '1_category_5_class_npy'
name = spilt + '_for_' + mode + '.json'

base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
path = base_path + spilt + '/annotations/' + name

data = json.load(open(path, 'r'))

images = data['images']
annotations = data['annotations']

image = images[10]
image_path = image['file_name']
image_id = image['id']

place_holder = np.zeros((64,64),dtype=np.float32)
for ann in annotations:
    if ann['image_id'] == image_id:
        print(ann['bbox'])
        bbox = ann['bbox']
        bbox = [int(i) for i in bbox]
        place_holder[bbox[1]:bbox[3],bbox[0]:bbox[2]] += 1
        
place_holder = place_holder / np.max(place_holder)
plt.imshow(place_holder)
plt.savefig('heatmap.png')

# read .npy file as nummpy
print(base_path+image_path)
img = np.load(base_path+image_path)

# do normality test on each small 4*4 patch of img
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import chisquare

place_holder = np.zeros((64,64),dtype=np.float32)   
for i in range(0,64,8):
    for j in range(0,64,8):
        print(i,j)
        print(anderson(img[i:i+8,j:j+8].flatten()))
        place_holder[i:i+8,j:j+8] = shapiro(img[i:i+8,j:j+8].flatten()).pvalue

place_holder = place_holder / np.max(place_holder)
plt.imshow(place_holder)
plt.savefig('place_holder.png')
        

