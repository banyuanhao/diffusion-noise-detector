from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus,set_seed, variance_5_class_index_sorted

def get_patch_natural(num=0):
    bounding_box = [40,27,24,24]
    
    seed = seeds_plus[variance_index_sorted[num]]
    latents = torch.randn((1,4,64,64), generator=set_seed(seed), device='cuda', dtype=torch.float32)
    patch = latents[:, :,bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]].clone()
    return patch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def torchmodify(name) :
    a=name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"
    return '.'.join(a)
import torch.nn as nn
for name, module in pipe.transformer.named_modules() :
    if isinstance(module,nn.GELU) :
        exec('model.'+torchmodify(name)+'=nn.GELU()')

# pick words from Imagenet class labels
pipe.labels  # to print all available words

# pick words that exist in ImageNet
words = ["basketball"]

class_ids = pipe.get_label_ids(words)
bounding_box_ = [6,6,12,12]
bounding_box = [0,0,24,24]
x_t, y_t, width_t, height_t = bounding_box
from diffusers.utils.torch_utils import randn_tensor
mode = 'natural'

for i in range(100):


    latents = randn_tensor(
            shape=(1,4,32,32),
            device=pipe._execution_device,
            dtype=pipe.transformer.dtype,
        )

    with torch.no_grad():
        
        if mode == 'natural':
            patch = get_patch_natural(0)
            latents[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = patch
        else:
            raise ValueError('mode not recognized')
        
    output = pipe(class_labels=class_ids, num_inference_steps=25, generator=torch.manual_seed(i), latents = latents)

    image = output.images[0]  # label 'white shark'
    image.save("white_shark.png")