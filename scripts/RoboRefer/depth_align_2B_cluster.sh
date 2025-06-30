#!/bin/bash

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

export BASE_RUN_NAME="Align"
export STAGE_PATH="./NVILA-Lite-2B-Depth" # Your base model path
export DATA_MIXTURE="2D_choice_qa+2D_choice_qa_RGB+2D_reasoning_template_qa+2D_reasoning_template_qa_RGB+3D_choice_qa+3D_choice_qa_RGB+3D_reasoning_template_qa+3D_reasoning_template_qa_RGB+3D_vacant_qa+3D_vacant_qa_RGB+3D_multi_view_qa+3D_multi_view_qa_RGB+3D_visual_choice_qa+3D_visual_choice_qa_RGB+simulation_dataset+simulation_dataset_RGB"


export OUTPUT_DIR=/share/project/zhouenshen/hpfs/code/VILA/runs/train/RoboRefer-2B-${BASE_RUN_NAME}
mkdir -p $OUTPUT_DIR
touch $OUTPUT_DIR/exp.log

# training config
export GPUS_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=7
export GRADIENT_ACCUMULATION_STEPS=1
export MASTER_PORT=25190


# network config
export NCCL_P2P_LEVEL=NVL
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=4
export ACCELERATE_CPU_AFFINITY=1
export NCCL_IB_HCA=mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108
ulimit -n 1048576



torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --depth_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample_3x3_fix \
    --depth_projector mlp_downsample_3x3_fix \
    --enable_depth True \
    --use_depth_tower True \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model False \
    --tune_depth_tower False \
    --tune_depth_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to wandb >>$OUTPUT_DIR/exp.log 2>&1
    
