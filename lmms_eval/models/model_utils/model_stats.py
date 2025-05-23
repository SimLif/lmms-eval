import argparse

from lmms_eval.models import get_model

import deepspeed
deepspeed.init_distributed(dist_backend='nccl')

def moe_count_parameters_in_billions(model, expert_num, top_k, layer_num):
    """
    统计模型的参数量，并以十亿(Billion)为单位返回
    
    参数:
        model: 模型
        count_moe_activated_only: 是否只计算MoE中被激活的参数
        top_k: MoE中被激活的专家数量
    """
    total_non_moe_params = 0
    total_gate_params = 0
    total_routed_expert_params = 0
    total_shared_expert_params = 0
    
    # 用于存储每个专家层的信息
    expert_layers = {}
    
    for name, param in model.named_parameters():
        if 'moe' in name:
            if 'meta' in name or 'shared' in name:
                total_shared_expert_params += param.numel()
            elif 'expert' in name:
                total_routed_expert_params += param.numel()
            else:
                total_gate_params += param.numel()
        elif ('original_mlp' in name) or ('shared' in name):
            total_shared_expert_params += param.numel()
        else: 
            total_non_moe_params += param.numel()


    total_activated_params = total_non_moe_params + total_gate_params + total_shared_expert_params + total_routed_expert_params / expert_num * top_k
    params_per_expert = total_routed_expert_params / expert_num / layer_num

    print(f'Params per expert: {params_per_expert/1e6:.2f}M')
    print(f'Total gate params: {total_gate_params/1e6:.2f}M')
    print(f'Total activated params: {total_activated_params/1e9:.2f}B')
    print(f'Total shared expert params: {total_shared_expert_params/1e6:.2f}M')
    print(f'Total routed expert params: {total_routed_expert_params/1e6:.2f}M')
    print(f'Total non moe params: {total_non_moe_params/1e9:.2f}B')    

def count_parameters_in_billions(model):
    """
    统计模型的参数量，并以十亿(Billion)为单位返回
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9  # 转换为十亿单位

# 使用示例
def print_model_size(model):
    param_count_billions = count_parameters_in_billions(model)
    print(f"模型参数量: {param_count_billions:.2f}B")
    
# 假设你已经有了一个名为model的模型
# print_model_size(model)

def count_image_tower_parameters_in_billions(model):
    """
    统计模型的参数量，并以十亿(Billion)为单位返回
    """
    total_params = sum(p.numel() for n, p in model.named_parameters() if 'vision' in n and 'projector' not in n)
    if total_params == 0:
        total_params = sum(p.numel() for n, p in model.named_parameters() if 'visual' in n and 'projector' not in n)

    return total_params / 1e9 # 转换为十亿单位


# print parameters of a model
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.shape)


def count_per_mlp_parameters(model):
    per_mlp_params = 0
    for name, param in model.named_parameters():
        if 'model.layers.1.mlp' in name:
            per_mlp_params += param.numel()
            print(name, param.shape)
    print(f'Per mlp params: {per_mlp_params/1e6:.2f}M')

def count_train_parameters(model):
    train_params = 0
    print('\nTrainable params:')
    for name, param in model.named_parameters():
        if (('mlp' in name) or ('moe' in name)) and ('vision' not in name) and ('visual' not in name):
            print(name, param.shape)
            train_params += param.numel()
    print(f'Trainable params: {train_params/1e9:.2f}B')

model_name = 'med_moe_phi'
# model_name = 'qwen2_vl'
# model_name = 'llava'
# model_name = 'moe_qwen2_vl'

# model_path = '/mnt/data/haoqiang/workspace/models/biomed-qwen2-vl-2b-instruct'
# model_path = '/mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct'
# model_path = '/mnt/data/haoqiang/workspace/models/llava-v1.5-7b'
# model_path = '/mnt/data/haoqiang/workspace/models/llava-med-v1.5-mistral-7b'
model_path = '/mnt/data/haoqiang/workspace/models/Med-MoE/stage3/llavaphi-2.7b-medmoe'

# ckpt_dir = "/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/"
# model_dir = 'qwen2-vl-2b-instruct-4e2-med-nano-val-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-3e1-med-share-5eopch'
# model_dir = 'qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch'
# model_dir = 'qwen2-vl-2b-instruct-6e2-nano-s-tok-share-5epoch'

# model_path = ckpt_dir + model_dir

# moe = False
moe = True


if model_path is not None:
    model = get_model(model_name)(model_path).model
else:
    model = get_model(model_name)().model

print_model_parameters(model)
count_train_parameters(model)
print_model_size(model)
image_tower_params = count_image_tower_parameters_in_billions(model)
print(f"Image tower parameters: {image_tower_params:.2f}B")
count_per_mlp_parameters(model)

if moe:
    moe_count_parameters_in_billions(model, expert_num=6, top_k=2, layer_num=14)  # 假设top_k=2

