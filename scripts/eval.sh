#!/bin/bash

# 默认参数
MODEL="med_moe_phi"
MODEL_SUFFIX=""  # 默认为空
TASKS="vqa_rad"
GPU_ID=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model_suffix)
      MODEL_SUFFIX="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

MODEL_DIR="/mnt/data/haoqiang/workspace/models"
# 模型路径映射字典
declare -A MODEL_PATHS
MODEL_PATHS["med_moe_phi"]="$MODEL_DIR/Med-MoE/stage3/llavaphi-2.7b-medmoe"
MODEL_PATHS["med_moe_stablelm"]="$MODEL_DIR/Med-MoE/stage3/llavastablelm-1.6b-medmoe"
# 可以继续添加更多映射...

# 构建完整的模型标识符，处理MODEL_SUFFIX为空的情况
if [[ -z "$MODEL_SUFFIX" ]]; then
    MODEL_ID="$MODEL"
    LOG_SUFFIX="$MODEL"
else
    MODEL_ID="${MODEL}_${MODEL_SUFFIX}"
    LOG_SUFFIX="${MODEL}_${MODEL_SUFFIX}"
fi

# 检查模型路径是否存在
if [[ -z "${MODEL_PATHS[$MODEL_ID]}" ]]; then
    echo "错误: 未找到模型路径映射: $MODEL_ID"
    echo "可用的模型标识符: ${!MODEL_PATHS[@]}"
    exit 1
fi

# 获取模型路径
MODEL_PATH="${MODEL_PATHS[$MODEL_ID]}"

echo "使用模型: $MODEL"
if [[ -n "$MODEL_SUFFIX" ]]; then
    echo "模型后缀: $MODEL_SUFFIX"
fi
echo "模型标识: $MODEL_ID"
echo "模型路径: $MODEL_PATH"
echo "评估任务: $TASKS"
echo "使用GPU: $GPU_ID"

# 执行评估命令
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --use_deepspeed \
    --zero_stage 0 \
    -m lmms_eval \
    --model $MODEL \
    --model_args pretrained="$MODEL_PATH" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${LOG_SUFFIX}_${TASKS} \
    --output_path ./logs/