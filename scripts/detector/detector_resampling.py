from mmdet.apis import DetInferencer
import sys
sys.path.append('/home/banyh2000/odfn')
import torch
from scripts.utils.utils_odfn import set_seed,auto_device
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import numpy as np

# Initialize the DetInferencer
inferencer = DetInferencer(model='/nfs/data/yuanhaoban/ODFN/model/data_augmentation.py', weights='/nfs/data/yuanhaoban/ODFN/model/weights.pth',device='cuda')

    
def reject_sample(therhold = 0.6):
    scores = 1
    while scores > therhold:
        seed = torch.randint(0,1000000,(1,)).item()
        
        array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
        array = array[0].cpu().numpy().transpose(1,2,0)

        results = inferencer(array)
        results = results['predictions'][0]
        scores = results['scores'][0]
        print(results['bboxes'][0])
    return seed

def accept_sample(therhold = 0.85):
    scores = 0
    while scores < therhold:
        seed = torch.randint(0,1000000,(1,)).item()
        
        array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
        array = array[0].cpu().numpy().transpose(1,2,0)

        results = inferencer(array)
        results = results['predictions'][0]
        scores = results['scores'][0]
        print(results['bboxes'][0])
    return seed

def reject_sample_var(therhold = 150):
    variances = 50
    while variances < therhold:
        seed = torch.randint(0,1000000,(1,)).item()
        
        array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
        array = array[0].cpu().numpy().transpose(1,2,0)

        results = inferencer(array)
        results = results['predictions'][0]
        
        results = results['bboxes']
        results = np.array(results)
        place_holder = np.zeros((results.shape[0], 2))
        place_holder[:, 0] = (results[:, 0] + results[:, 2]) / 2
        place_holder[:, 1] = (results[:, 1] + results[:, 3]) / 2
        results = place_holder[:4,:]
        variances = np.var(results, axis=0)
        variances = np.mean(variances)
        print(variances)
    return seed

def accept_sample_var(therhold = 10):
    variances = 150
    while variances > therhold:
        seed = torch.randint(0,1000000,(1,)).item()
        
        array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
        array = array[0].cpu().numpy().transpose(1,2,0)

        results = inferencer(array)
        results = results['predictions'][0]
        
        results = results['bboxes']
        results = np.array(results)
        place_holder = np.zeros((results.shape[0], 2))
        place_holder[:, 0] = (results[:, 0] + results[:, 2]) / 2
        place_holder[:, 1] = (results[:, 1] + results[:, 3]) / 2
        results = place_holder[:4,:]
        variances = np.var(results, axis=0)
        variances = np.mean(variances)
    return seed

def reject_sample_con(therhold = 0.6,seed=None):
    scores = 1
    if seed is None:
        seed = torch.randint(0,1000000,(1,)).item()
    array = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    array = array[0].cpu().numpy().transpose(1,2,0)
    count = 0
    while True:
        count += 1
        results = inferencer(array)
        results = results['predictions'][0]
        scores = results['scores'][0]
        bbox = results['bboxes'][0]
        if scores < therhold:
            break
        else:
            # exp2 right
            patch = torch.randn((4, int(bbox[3])-int(bbox[1]), int(bbox[2])-int(bbox[0])), device='cuda', dtype=torch.float32)
            patch = patch.cpu().numpy().transpose(1,2,0)
            array[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:] = patch
            
            # exp1 wrong
            # patch = torch.randn((4, int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1])), device='cuda', dtype=torch.float32)
            # patch = patch.cpu().numpy().transpose(1,2,0)
            # array[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3]),:] = patch
    
    array = torch.tensor(array.transpose(2,0,1), device='cuda', dtype=torch.float32).unsqueeze(0)
    # print(count)
    return array