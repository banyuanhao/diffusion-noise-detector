mkdir modelpara
cd modelpara

mkdir det
cd dir

http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth

https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py

mim download mmdet --config gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco --dest .


### environment
conda create -n odfn python=3.10
conda install pytorch=1.11 torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install diffusers["torch"] transformers
huggingface-cli login

## training
scripts/generating/dist_train.sh odfn_config/_.py 8 --work-dir /mnt/data0/banyuanhao/ODFN/work_dirs/_

scripts/generating/dist_train.sh odfn_config/ablation/base.py 8 --work-dir /mnt/data0/banyuanhao/ODFN/work_dirs/base

scripts/generating/dist_train.sh odfn_config/ablation/annotations_after.py 8 --work-dir /mnt/data0/banyuanhao/ODFN/work_dirs/ablation_annotations_after
