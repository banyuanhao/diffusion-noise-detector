import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import init_detector, inference_detector

array = np.random.randn(64,64,3)
# scale to 0,1
array = (array - np.min(array)) / (np.max(array) - np.min(array))
array[:10,:,0] = 1
plt.imshow(array)
plt.savefig('pics/pic.png')


config_file = 'model/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'model/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
img_path = 'pics/diversity/accept/A sports ball is caught in a fence..png'
model = init_detector(config_file, checkpoint_file, device='cuda')
result = inference_detector(model, img_path)
print(result)