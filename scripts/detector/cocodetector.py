from mmdet.apis import DetInferencer
import sys
sys.path.append('/home/banyh2000/odfn')
import numpy as np

inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')
results = inferencer('/home/banyh2000/odfn/pics/diversity/accept/A sports ball is caught in a fence..png',out_dir='/home/banyh2000/odfn/pics/diversity')
print(results)

# class Detector:
#     def __init__(self, model_name='rtmdet-ins_l_8xb32-300e_coco'):
#         self.inferencer = DetInferencer(model='rtmdet-ins_l_8xb32-300e_coco')
        
#     def __call__(self, image):
#         results = self.inferencer(image)
#         results = results['predictions'][0]
#         return results