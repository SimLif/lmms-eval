#!/usr/bin/env python3
import argparse
import os

import torch

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    raise ImportError("需要安装 safetensors 库，请执行 pip install safetensors")


def load_checkpoint(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == ".safetensors":
        print("检测到 safetensors 格式")
        ckpt = load_file(filepath)
        file_format = "safetensors"
    else:
        print("检测到 PyTorch 格式")
        ckpt = torch.load(filepath, map_location="cpu")
        file_format = "pytorch"
    return ckpt, file_format


def save_checkpoint(ckpt, file_format, filepath):
    if file_format == "safetensors":
        print("保存为 safetensors 格式")
        save_file(ckpt, filepath)
    else:
        print("保存为 PyTorch 格式")
        torch.save(ckpt, filepath)


def process_checkpoint(ckpt, pattern):
    # 获取 state_dict，如果 checkpoint 中包含 "state_dict" 字段
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    updated_count = 0
    for key in state_dict.keys():
        if pattern in key:
            param = state_dict[key]
            print(f"找到参数 {key}，shape: {param.shape}")
            if param.dim() == 0:
                state_dict[key] = param.unsqueeze(0)
                updated_count += 1
                print(f"修改后 {key} 的新 shape: {state_dict[key].shape}")
            else:
                print(f"{key} 无需修改，tensor 维度为 {param.dim()}")
    print(f"总共更新了 {updated_count} 个参数")
    # 将处理后的 state_dict 写回到 checkpoint 中（如果有 "state_dict" 字段）
    if "state_dict" in ckpt:
        ckpt["state_dict"] = state_dict
    else:
        ckpt = state_dict
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="加载 checkpoint，修改匹配指定 pattern 的 0 维参数后保存")
    parser.add_argument("--input", type=str, required=True, help="输入的 checkpoint 路径")
    parser.add_argument("--output", type=str, required=True, help="输出的 checkpoint 路径")
    parser.add_argument("--pattern", type=str, default=".mlp.moe_layer.moe.gate.query_bn.num_batches_tracked", help="匹配参数的子串，默认匹配 query_bn 中的 num_batches_tracked")
    args = parser.parse_args()

    ckpt, file_format = load_checkpoint(args.input)
    ckpt = process_checkpoint(ckpt, args.pattern)
    save_checkpoint(ckpt, file_format, args.output)
    print(f"修改后的 checkpoint 保存在: {args.output}")


if __name__ == "__main__":
    main()


# python process_ckpt.py --input /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-256e8x4-med-nano-ee-smi-5epoch/model-00001-of-00002.safetensors --output /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-256e8x4-med-nano-ee-smi-5epoch/modified_model-00001-of-00002.safetensors


# python process_ckpt.py --input /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-256e8x4-med-nano-ee-smi-5epoch/model-00002-of-00002.safetensors --output /mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-256e8x4-med-nano-ee-smi-5epoch/modified_model-00002-of-00002.safetensors
