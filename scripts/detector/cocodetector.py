from mmdet.apis import DetInferencer
import sys
sys.path.append('/home/banyh2000/odfn')
import numpy as np

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')

# class Detector:
#     def __init__(self, model_name='rtmdet-ins_l_8xb32-300e_coco'):
#         self.inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')
        
#     def __call__(self, image):
#         results = self.inferencer(image)
#         results = results['predictions'][0]
#         return results