from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from mmcv.transforms import LoadImageFromFile

from scripts.utils.utils_odfn import variance_index_sorted,seeds_plus
print(seeds_plus[variance_index_sorted[19999]])
