import argparse

from lmms_eval.models import get_model

def moe_count_parameters_in_billions(model, count_moe_activated_only=False, top_k=None):
    """
    统计模型的参数量，并以十亿(Billion)为单位返回
    
    参数:
        model: 模型
        count_moe_activated_only: 是否只计算MoE中被激活的参数
        top_k: MoE中被激活的专家数量
    """
    total_non_moe_params = 0
    
    # 用于存储每个专家层的信息
    expert_layers = {}
    
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        #     continue
            
        # 检查是否是MoE层的专家参数
        if 'deepspeed_moe.experts.deepspeed_experts' in name:
            try:
                # 提取专家层的标识符
                layer_prefix = name.split('.experts.deepspeed_experts.')[0]
                
                # 提取专家索引
                parts = name.split('.experts.deepspeed_experts.')[1]
                expert_idx = int(parts.split('.')[0])
                
                # 初始化该层信息
                if layer_prefix not in expert_layers:
                    expert_layers[layer_prefix] = {
                        'num_experts': 0,
                        'total_params': 0
                    }
                
                # 更新该层的专家数量
                expert_layers[layer_prefix]['num_experts'] = max(
                    expert_layers[layer_prefix]['num_experts'], 
                    expert_idx + 1
                )
                
                # 累加该层的参数总量
                expert_layers[layer_prefix]['total_params'] += param.numel()
            except (IndexError, ValueError):
                # 如果解析失败，作为普通参数处理
                total_non_moe_params += param.numel()
        else:
            # 非MoE参数
            total_non_moe_params += param.numel()
    
    # 计算总参数量
    total_params = total_non_moe_params
    
    # 计算MoE参数
    for layer_info in expert_layers.values():
        num_experts = layer_info['num_experts']
        total_layer_params = layer_info['total_params']
        
        if num_experts > 0:  # 避免除以零
            # 计算每个专家的平均参数量
            params_per_expert = total_layer_params / num_experts
            
            if count_moe_activated_only and top_k is not None:
                # 只计算激活的top_k个专家
                activated_experts = min(top_k, num_experts)
                total_params += params_per_expert * activated_experts
            else:
                # 计算所有专家的参数
                total_params += total_layer_params
    
    print(f'Params per expert: {params_per_expert/1e6:.2f}M')
    
    return total_params / 1e9  # 转换为十亿单位


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
    total_params = sum(p.numel() for n, p in model.named_parameters() if 'image' in n)
    return total_params / 1e9 # 转换为十亿单位


# print parameters of a model
def print_model_parameters(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.numel())


model_name = 'med_moe_phi'

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default=model_name)
argparser.add_argument('--model_path', type=str, default=None)
argparser.add_argument('--moe', action='store_false', default=True)
args = argparser.parse_args()
model_name = args.model

# get_model(model_name)

if args.model_path is not None:
    model = get_model(model_name)(args.model_path).model
else:
    model = get_model(model_name)().model


if args.moe:
    # 计算所有参数（包括所有专家）
    total_params = moe_count_parameters_in_billions(model)
    print(f"Total parameters: {total_params:.2f}B")
    # 计算前向传播中实际激活的参数
    activated_params = moe_count_parameters_in_billions(model, count_moe_activated_only=True, top_k=2)  # 假设top_k=2
    print(f"Activated parameters: {activated_params:.2f}B")
    # 计算图像塔的参数
    image_tower_params = count_image_tower_parameters_in_billions(model)
    print(f"Image tower parameters: {image_tower_params:.2f}B")
else:
    print_model_parameters(model) 



