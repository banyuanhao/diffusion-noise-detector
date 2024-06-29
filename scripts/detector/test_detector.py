import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import init_detector, inference_detector
import cv2
# array = np.random.randn(64,64,3)
# # scale to 0,1
# array = (array - np.min(array)) / (np.max(array) - np.min(array))
# array[:10,:,0] = 1
# plt.imshow(array)
# plt.savefig('pics/pic.png')


# config_file = 'model/rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = 'model/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# img_path = 'pics/diversity/accept/A sports ball is caught in a fence.png'
# model = init_detector(config_file, checkpoint_file, device='cuda')
# result = inference_detector(model, img_path)
# # visualize the results, only draw the bbox with the highest score
# model.show_result(img_path, result, score_thr=0.3, show=False, out_file='pics/demo.png')

num = 8
for i in range(1,9):
    img_path = f'/nfs/data/yuanhaoban/ODFN/diversity/various/exp1/control/images/seed_{num}_prompt_{i}.png'
    # initialize a standard coco detector
    config_file = 'model/rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'model/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda')
    result = inference_detector(model, img_path)
    # visualize the results, only draw the bbox with the highest score
    results = result.pred_instances
    bbox = results.bboxes[0]
    label = results.labels[0]
    image = cv2.imread(img_path)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
    cv2.imwrite(f'pics/pr/{num}_con_{i}.png', image)

for i in range(1,9):
    img_path = f'/nfs/data/yuanhaoban/ODFN/diversity/various/exp1/rejection/images/seed_{num}_prompt_{i}.png'
    # initialize a standard coco detector
    config_file = 'model/rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'model/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda')
    result = inference_detector(model, img_path)
    # visualize the results, only draw the bbox with the highest score
    results = result.pred_instances
    bbox = results.bboxes[0]
    label = results.labels[0]
    image = cv2.imread(img_path)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
    cv2.imwrite(f'pics/pr/{num}_rej_{i}.png', image)