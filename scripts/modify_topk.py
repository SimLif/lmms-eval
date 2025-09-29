import json
import argparse
import sys # 用于 sys.exit

def modify_json_file(file_path, new_top_k_value):
    """
    读取一个 JSON 文件，修改 'moe'.'top_k_experts' 字段，
    并将更改写回文件。
    """
    try:
        # 1. 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"错误：无法解码文件 '{file_path}' 中的 JSON。详情: {e}", file=sys.stderr)
                sys.exit(1) # 以错误码退出
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"错误：无法读取文件 '{file_path}'。详情: {e}", file=sys.stderr)
        sys.exit(1)

    original_value_str = "不存在" # 用于记录原始值的字符串

    # 2. 定位到 'moe' 字典并更新/创建 'top_k_experts'
    # 检查 'moe' 键是否存在，如果不存在则创建
    if 'moe' not in data:
        print(f"警告：在 '{file_path}' 中未找到 'moe' 键。将创建 'moe' 对象。")
        data['moe'] = {} # 创建 'moe' 字典
    elif not isinstance(data['moe'], dict):
        # 如果 'moe' 存在但不是一个字典，则报错
        print(f"错误：'{file_path}' 中的 'moe' 键的值不是一个字典（对象）。无法设置 'top_k_experts'。", file=sys.stderr)
        sys.exit(1)

    # 在 'moe' 字典中检查 'top_k_experts' 键
    if 'top_k_experts' in data['moe']:
        original_value_str = str(data['moe']['top_k_experts'])
    else:
        print(f"警告：在 'moe' 对象中未找到 'top_k_experts' 键。将创建该键。")

    # 设置新的 top_k_experts 值
    data['moe']['top_k_experts'] = new_top_k_value

    print(f"\n在文件 '{file_path}' 中成功处理 'moe'.'top_k_experts'：")
    print(f"  旧值: {original_value_str}")
    print(f"  新值: {new_top_k_value}")

    # 3. 将修改后的数据写回 JSON 文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # indent=2 用于美化输出，使其易于阅读
            # ensure_ascii=False 确保非 ASCII 字符（如中文）正确显示而不是被转义
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"文件 '{file_path}' 已成功更新。")
    except IOError as e:
        print(f"错误：无法写入文件 '{file_path}'。详情: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="修改 JSON 文件中 'moe' 对象下的 'top_k_experts' 值。",
        # formatter_class=argparse.RawTextHelpFormatter # 如果帮助文本需要特殊格式，可以使用这个
    )

    # 添加必需的位置参数：JSON 文件路径
    # parser.add_argument(
    #     "json_file",  # 参数名称
    #     help="要修改的 JSON 配置文件的路径。"  # 帮助信息
    # )
    # json_file = '/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/config.json'
    json_file = '/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/config.json'

    # 添加必需的命名参数：--topk
    parser.add_argument(
        "--topk",
        type=int,  # 指定参数类型为整数
        required=True,  # 设置为必需参数
        help="为 'top_k_experts' 设置的新的整数值。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数执行修改操作
    modify_json_file(json_file, args.topk)