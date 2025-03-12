# conda activate lmms-eval
# bash eval.sh --model med_moe_phi --tasks medmoe_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen --tasks vqa_rad
# bash eval.sh --model moe_llava_qwen --tasks medmoe_vqa
# bash eval.sh --model moe_llava_qwen_med --model_suffix 1epoch --tasks medmoe_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix 9epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_1epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_9epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_k_9epoch --tasks medmoe_vqa --gpu 0


# conda activate lmms-eval-twin
bash eval.sh --model qwen2_vl --model_suffix instruct --tasks medmoe_vqa --gpu 0
# bash eval.sh --model qwen2_vl_adamllm --tasks medmoe_vqa --gpu 0
