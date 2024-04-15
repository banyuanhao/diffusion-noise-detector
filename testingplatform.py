from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from mmcv.transforms import LoadImageFromFile

import json
spilts = ['train', 'val', 'test']
for spilt in spilts:
    path = f'/nfs/data/yuanhaoban/ODFN/version_2/{spilt}/annotations/{spilt}_for_1_category_5_class_npy.json'
    with open(path, 'r') as f:
        data = json.load(f)
        image_data = data['annotations']
        print(len(image_data))
    #     for image in image_data:
    #         image['file_name'] = image['file_name'].replace('/noises/', '/noises_npy/')
        
    # data['images'] = image_data
    
    # with open(path, 'w') as f:
    #     json.dump(data, f)
        
        
# /nfs/data/yuanhaoban/ODFN/version_2/test/annotations/test_for_1_category_5_class.json\
    