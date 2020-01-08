#!/bin/bash -ex

export FLAGS_fraction_of_gpu_memory_to_use=0.8
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=7000 #MB

DATA_FORMAT="NHWC"


#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG=logcudnn.txt
#export FLAGS_benchmark=1
#export FLAGS_profile_compute=1
#export CUDA_VISIBLE_DEVICES=5

#python train.py \
#       --model=ResNet50 \
#       --data_dir=/work/datasets/ILSVRC2012/ \
#       --batch_size=128 \
#       --total_images=1281167 \
#       --print_step=10 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --fp16=True \
#       --use_dynamic_loss_scaling=True \
#       --scale_loss=128.0 \
#       --data_format=${DATA_FORMAT} \
#       --validate=0 \
#       --is_profiler=0 \
#       --profiler_path=profile/ \
#       --reader_thread=10 \
#       --reader_buf_size=8192 \
#       --use_dali=True \
#       --lr=0.1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch train.py \
       --model=ResNet50 \
       --data_dir=/work/datasets/ILSVRC2012/ \
       --batch_size=1024 \
       --total_images=1281167 \
       --print_step=10 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --fp16=True \
       --use_dynamic_loss_scaling=True \
       --scale_loss=128.0 \
       --data_format=${DATA_FORMAT} \
       --validate=0 \
       --is_profiler=0 \
       --profiler_path=profile/ \
       --reader_thread=10 \
       --reader_buf_size=8192 \
       --use_dali=True \
       --lr=0.1

