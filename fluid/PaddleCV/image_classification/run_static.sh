#!/usr/bin/env bash

# 若训练总Pass数(num_epochs)为10，可在第7个Pass后将学习率降低为原来的1/10
ONLY_ONE_CUDA_DEVICE_NUM=0 # 0 is for example
export CUDA_VISIBLE_DEVICES=$ONLY_ONE_CUDA_DEVICE_NUM
# FLAGS_fraction_of_gpu_memory_to_use is set according to your requirement
export FLAGS_fraction_of_gpu_memory_to_use=0.8
#MobileNet v1:
#python train_quant.py \
#       --model=MobileNet \
#       --pretrained_model=MobileNetV1_pretrained \
#       --batch_size=64 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir_prefix=static_ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=10 \
#       --lr=0.00001 \
#       --model_category=models_name \
#       --quant_type=range_abs_max


#ResNet50:
python train_quant.py \
       --model=ResNet50 \
       --pretrained_model=ResNet50_pretrained/ \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir_prefix=static_ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --num_epochs=10 \
       --lr=0.00001 \
       --model_category=models_name \
       --quant_type=range_abs_max
