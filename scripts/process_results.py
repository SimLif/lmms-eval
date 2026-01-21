import glob
import json  # Added for the test setup
import os
import re
import subprocess


def find_and_process_checkpoints(parent_dir, model_base_name, show_results_script_path="show_results.py"):
    """
    查找并处理给定模型的checkpoint文件夹及其results.json文件。

    Args:
        parent_dir (str): 包含模型checkpoint文件夹的父目录 (例如 "a")。
        model_base_name (str): 模型的基础名称 (例如 "qwen2-vl-2b-instruct-4e2-ada-med-10epoch")。
        show_results_script_path (str): show_results.py 脚本的路径。
    """
    print(f"正在目录 '{parent_dir}' 中搜索模型 '{model_base_name}' 的 checkpoints...\n")

    checkpoints_info = []

    # 构建匹配checkpoint文件夹的模式
    # 文件夹应以 model_base_name 开头并包含 "__checkpoint-"
    # 例如: qwen2-vl-2b-instruct-4e2-ada-med-10epoch__checkpoint-500
    folder_pattern = os.path.join(parent_dir, f"{model_base_name}__checkpoint-*")

    candidate_folders = glob.glob(folder_pattern)

    for folder_path in candidate_folders:
        if not os.path.isdir(folder_path):
            continue

        folder_name = os.path.basename(folder_path)

        # 提取 checkpoint 编号
        # 使用正则表达式从文件夹名称中提取 "__checkpoint-" 后面的数字
        match = re.search(r"__checkpoint-(\d+)", folder_name)
        if not match:
            print(f"警告: 无法从文件夹 '{folder_name}' 中提取 checkpoint 编号。已跳过。")
            continue

        try:
            checkpoint_num = int(match.group(1))
        except ValueError:
            print(f"警告: 文件夹 '{folder_name}' 中的 checkpoint 编号 '{match.group(1)}' 无效。已跳过。")
            continue

        # 在此文件夹中查找 xxx_results.json 文件
        # 例如: 20250507_024617_results.json
        json_file_pattern = os.path.join(folder_path, "*_results.json")
        json_files = glob.glob(json_file_pattern)

        if not json_files:
            print(f"警告: 在文件夹 '{folder_path}' 中未找到 '*_results.json' 文件。已跳过。")
            continue

        if len(json_files) > 1:
            # 如果找到多个匹配的JSON文件，默认使用第一个，并打印警告
            print(f"警告: 在文件夹 '{folder_path}' 中找到多个 '*_results.json' 文件。将使用第一个: '{json_files[0]}'.")

        results_json_path = os.path.abspath(json_files[0])  # 确保是绝对路径

        checkpoints_info.append({"checkpoint": checkpoint_num, "folder_path": folder_path, "json_file_path": results_json_path})

    if not checkpoints_info:
        print(f"在 '{parent_dir}' 中未找到模型 '{model_base_name}' 的 checkpoint 文件夹。")
        return

    # 按 checkpoint 编号排序
    checkpoints_info.sort(key=lambda x: x["checkpoint"])

    print("\n已找到并排序的 checkpoints:")
    for info in checkpoints_info:
        print(f"  Checkpoint: {info['checkpoint']}, JSON 文件: {info['json_file_path']}")

    print("\n按顺序处理 checkpoints:")
    for info in checkpoints_info:
        print(f"\n--- 正在处理 Checkpoint: {info['checkpoint']} ---")

        # 1. 打印 checkpoint
        print(f"Checkpoint: {info['checkpoint']}")

        # 2. 执行 python show_results.py xxxx_results.json 的绝对路径
        command = ["python", show_results_script_path, info["json_file_path"]]

        print(f"将执行命令: {' '.join(command)}")

        try:
            # 执行外部脚本，并捕获其输出
            process = subprocess.run(command, check=True, text=True, capture_output=True, encoding="utf-8")
            print("命令执行成功。")
            if process.stdout:
                print("来自 show_results.py 的输出:")
                print(process.stdout)
            if process.stderr:  # 尽管 check=True, 有些程序可能将警告输出到stderr
                print("来自 show_results.py 的错误信息 (stderr):")
                print(process.stderr)
        except FileNotFoundError:
            print(f"错误: 脚本 '{show_results_script_path}' 未找到。")
            print("请确保 'python' 在您的 PATH 环境变量中，并且脚本路径正确。")
            break  # 如果脚本未找到，则停止进一步处理
        except subprocess.CalledProcessError as e:
            # 当被调用的脚本返回非零退出码时，会抛出此异常
            print(f"执行 checkpoint {info['checkpoint']} 的命令时出错:")
            print(f"命令: {' '.join(e.cmd)}")
            print(f"返回码: {e.returncode}")
            if e.stdout:
                print("标准输出 (Stdout):")
                print(e.stdout)
            if e.stderr:
                print("标准错误 (Stderr):")
                print(e.stderr)
            # 你可以在这里决定是继续处理下一个checkpoint还是停止
            # print("继续处理下一个 checkpoint...")
        except Exception as e:
            print(f"处理 checkpoint {info['checkpoint']} 时发生意外错误: {e}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 父文件夹，例如 "a"
    parent_directory = "/mnt/data/haoqiang/workspace/02-lmms-eval/logs"
    # 你想处理的基础模型名称
    # model_name_to_process = "qwen2-vl-2b-instruct-3e1-med-10epoch"
    # model_name_to_process = "qwen2-vl-2b-instruct-4e2-ada-med-10epoch"
    # model_name_to_process = 'qwen2-vl-2b-instruct-3e1-ada-med-10epoch'
    # model_name_to_process = 'qwen2-vl-2b-instruct-96e32-ada-med-10epoch'
    model_name_to_process = "qwen2-vl-2b-instruct-96e32-ada-med-re-10epoch"

    # 你的 show_results.py 脚本的路径。
    # 如果它与此脚本在同一目录中，则只需 "show_results.py"。
    # 否则，请提供完整路径或相对路径，例如 "scripts/show_results.py"。
    show_results_script = "show_results.py"
    # --- 配置区域结束 ---

    # --- 测试设置 ---
    # # 这部分代码用于在本地创建虚拟的文件夹和文件结构以供测试。
    # # 在实际使用时，你可以删除或注释掉这部分代码。
    # print("--- 开始测试设置 (如果文件夹和文件已存在，则跳过创建) ---")
    # if not os.path.exists(parent_directory):
    #     os.makedirs(parent_directory)
    #     print(f"创建了测试父目录: {parent_directory}")

    # # 定义一些测试用的文件夹和文件
    # test_folders_files_config = {
    #     f"{model_name_to_process}__checkpoint-500": "20250507_024617_results.json",
    #     f"{model_name_to_process}__checkpoint-1000": "20250507_024830_results.json",
    #     f"another-model__checkpoint-100": "20250507_010000_results.json", # 这个应该被忽略
    #     f"{model_name_to_process}__checkpoint-100": "20250507_024000_results.json", # 用于测试排序
    #     f"{model_name_to_process}__checkpoint-malformed": "20250507_025000_results.json", # 测试无效checkpoint名称
    #     f"{model_name_to_process}__checkpoint-200": "no_results_here.txt", # 测试没有results.json文件的情况
    # }

    # # 创建一个虚拟的 show_results.py 脚本用于测试
    # if not os.path.exists(show_results_script):
    #     with open(show_results_script, "w", encoding="utf-8") as f:
    #         f.write("#!/usr/bin/env python\n")
    #         f.write("import sys\n")
    #         f.write("import json\n")
    #         f.write("import os\n")
    #         f.write("print(f'Mock show_results.py: 正在处理文件: {sys.argv[1]}')\n")
    #         f.write("if not os.path.exists(sys.argv[1]):\n")
    #         f.write("    print(f'Mock show_results.py: 错误 - 文件 {sys.argv[1]} 不存在。')\n")
    #         f.write("    sys.exit(1)\n")
    #         f.write("try:\n")
    #         f.write("    with open(sys.argv[1], 'r', encoding='utf-8') as rf:\n")
    #         # 假设它是一个JSON文件
    #         f.write("        data = json.load(rf)\n")
    #         f.write("        print(f'Mock show_results.py: 成功从 {sys.argv[1]} 读取JSON数据。')\n")
    #         f.write("        print(f'Mock show_results.py: 内容: {data}')\n")
    #         f.write("except json.JSONDecodeError:\n")
    #         f.write("    print(f'Mock show_results.py: 文件 {sys.argv[1]} 不是有效的JSON。')\n")
    #         f.write("    # 如果希望非JSON文件也算成功，可以移除 sys.exit(1)\n")
    #         f.write("    # sys.exit(1) \n")
    #         f.write("except Exception as e:\n")
    #         f.write("    print(f'Mock show_results.py: 读取或解析 {sys.argv[1]} 时发生错误: {e}')\n")
    #         f.write("    sys.exit(1)\n")
    #     # 在Unix-like系统上，添加执行权限 (虽然通过 `python script.py` 调用时不是必须的)
    #     if os.name != 'nt':
    #          os.chmod(show_results_script, 0o755)
    #     print(f"创建了虚拟的测试脚本: {show_results_script}")

    # for folder_base_name, file_name_in_folder in test_folders_files_config.items():
    #     full_folder_path = os.path.join(parent_directory, folder_base_name)
    #     if not os.path.exists(full_folder_path):
    #         os.makedirs(full_folder_path)
    #         print(f"创建了测试文件夹: {full_folder_path}")

    #     target_file_path = os.path.join(full_folder_path, file_name_in_folder)
    #     if not os.path.exists(target_file_path) and file_name_in_folder.endswith("_results.json"):
    #         # 创建一个虚拟的 JSON 文件
    #         checkpoint_val_str = folder_base_name.split('-')[-1]
    #         try:
    #             checkpoint_val = int(checkpoint_val_str)
    #         except ValueError:
    #             checkpoint_val = 0 #  对于 "malformed" 之类的名称

    #         dummy_json_content = {
    #             "checkpoint_folder_name": folder_base_name,
    #             "file_processed_by_script": file_name_in_folder,
    #             "example_metric_value": checkpoint_val,
    #             "notes": "这是一个自动生成的测试文件"
    #         }
    #         with open(target_file_path, "w", encoding="utf-8") as f:
    #             json.dump(dummy_json_content, f, indent=2, ensure_ascii=False)
    #         print(f"创建了测试JSON文件: {target_file_path}")
    #     elif not os.path.exists(target_file_path):
    #          with open(target_file_path, "w", encoding="utf-8") as f:
    #             f.write("This is a dummy non-JSON file for testing.")
    #          print(f"创建了测试文件 (非JSON): {target_file_path}")

    #     # 创建一个额外的非匹配json文件以测试glob的鲁棒性
    #     other_json_path = os.path.join(full_folder_path, "some_other_data.json")
    #     if not os.path.exists(other_json_path):
    #         with open(other_json_path, "w", encoding="utf-8") as f:
    #             json.dump({"info": "This is another JSON file, should not be processed by the main logic."}, f)
    #         print(f"创建了额外的JSON文件: {other_json_path}")

    #     # 创建一个非json文件
    #     log_txt_path = os.path.join(full_folder_path, "activity.log")
    #     if not os.path.exists(log_txt_path):
    #         with open(log_txt_path, "w", encoding="utf-8") as f:
    #             f.write("some log data here")
    #         print(f"创建了日志文件: {log_txt_path}")
    # print("--- 测试设置结束 ---\n")
    # --- 测试设置结束 ---

    # 检查父目录是否存在
    if not os.path.isdir(parent_directory):
        print(f"错误: 父目录 '{parent_directory}' 不存在。")
        print("请创建该目录或提供正确的路径。")
    else:
        find_and_process_checkpoints(parent_directory, model_name_to_process, show_results_script)
