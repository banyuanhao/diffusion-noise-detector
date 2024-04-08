spilt = 'val'
mode = '1_category_1_class_npy'
name = spilt + '_for_' + mode + '.json'

base_path = '/nfs/data/yuanhaoban/ODFN/version_2/'
path = base_path + spilt + '/annotations/' + name

import json
with open(path, 'r') as f:
    data = json.load(f)
    
print(data.keys())