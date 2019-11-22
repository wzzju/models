#!/bin/bash -ex

export FLAGS_fraction_of_gpu_memory_to_use=0.8

export CUDA_VISIBLE_DEVICES=7

python train.py \
       --model=ResNet50 \
       --data_dir=/work/datasets/ILSVRC2012/ \
       --batch_size=128 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --fp16=True \
       --validate=1 \
       --is_profiler=0 \
       --profiler_path=profile/ \
       --reader_thread=8 \
       --use_dali=True \
       --lr=0.1

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#python -m paddle.distributed.launch train.py \
#       --model=ResNet50 \
#       --data_dir=/work/datasets/ILSVRC2012/ \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --fp16=True \
#       --validate=1 \
#       --is_profiler=0 \
#       --profiler_path=profile/ \
#       --use_dali=True \
#       --reader_thread=1 \
#       --lr=0.1

