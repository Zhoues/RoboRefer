#!/bin/bash

DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=256 # 2(PER_DEVICE_TRAIN_BATCH_SIZE) * 8(GPUS_PER_NODE) * 8(NNODES) * 2(GRADIENT_ACCUMULATION_STEPS)
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2

STAGE_PATH=${1:-"./NVILA-8B-Depth"} # Your base model path
DATA_MIXTURE=${2:-"2D_choice_qa+2D_choice_qa_RGB+2D_reasoning_template_qa+2D_reasoning_template_qa_RGB+3D_choice_qa+3D_choice_qa_RGB+3D_reasoning_template_qa+3D_reasoning_template_qa_RGB+3D_vacant_qa+3D_vacant_qa_RGB+3D_multi_view_qa+3D_multi_view_qa_RGB+3D_visual_choice_qa+3D_visual_choice_qa_RGB+simulation_dataset+simulation_dataset_RGB"}
OUTPUT_DIR=${3:-"runs/train/RoboRefer-8B-Depth-Align"}

source scripts/setups/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --chat_template qwen2 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --depth_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --dynamic_s2 True \
        --s2_scales "448,896,1344" \
        --s2_max_split_size 448 \
        --s2_resize_output_to_scale_idx -1 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --depth_projector mlp_downsample \
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
        --image_aspect_ratio dynamic_s2 \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --report_to wandb
