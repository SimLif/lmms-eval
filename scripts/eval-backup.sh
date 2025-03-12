CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    --zero_stage 0 \
    -m lmms_eval \
    --model med_moe_phi \
    --model_args pretrained="/mnt/data/haoqiang/workspace/models/Med-MoE/stage3/llavaphi-2.7b-medmoe" \
    --tasks path_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix med_moe_phi_path_vqa \
    --output_path ./logs/

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    --zero_stage 0 \
    -m lmms_eval \
    --model med_moe_phi \
    --model_args pretrained="/mnt/data/haoqiang/workspace/models/Med-MoE/stage3/llavaphi-2.7b-medmoe" \
    --tasks vqa_rad \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix med_moe_phi_vqa_rad \
    --output_path ./logs/


CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    -m lmms_eval \
    --model med_moe_stablelm \
    --model_args pretrained="/mnt/data/haoqiang/workspace/models/Med-MoE/stage3/llavastablelm-1.6b-medmoe" \
    --tasks medmoe_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix med_moe_stablelm_medmoe_vqa \
    --output_path ./logs/

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    -m lmms_eval \
    --model qwen_vl \
    --model_args pretrained="/mnt/data/haoqiang/workspace/models/Qwen-VL" \
    --tasks medmoe_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen_vl_medmoe_vqa \
    --output_path ./logs/

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    -m lmms_eval \
    --model qwen_vl \
    --model_args pretrained="/mnt/data/haoqiang/workspace/models/Qwen-VL" \
    --tasks medmoe_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen_vl_medmoe_vqa \
    --output_path ./logs/