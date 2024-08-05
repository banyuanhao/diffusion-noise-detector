from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch

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
words = ["white shark", "umbrella"]

class_ids = pipe.get_label_ids(words)

generator = torch.manual_seed(33)
output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

image = output.images[0]  # label 'white shark'