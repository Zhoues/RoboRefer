
export TORCH_USE_CUDA_DSA=1  # Linux/macOS
export TORCH_USE_CUDA_DSA=1  # Linux/macOS

# image description
export CUDA_VISIBLE_DEVICES=0
vila-infer \
    --model-path /home/zhouenshen/code/VILA/runs/train/NVILA-Lite-2B-depth-sft-2d+3d+sim/model/checkpoint-10000\
    --conv-mode auto \
    --text "Mark the pixel coordinate for the location that is on the front-facing side of the green cup's handle." \
    --media "/home/zhouenshen/code/VILA/test/image.jpg" \
    