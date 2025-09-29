#!/usr/bin/env python3
import json
import argparse

def get_metric(results, task_name, metric_key):
    """
    从 results 中获取指定 task 及指标的数值，
    如果任务或指标不存在，则返回 -1。
    """
    task_data = results.get(task_name, {})
    return task_data.get(metric_key, -1)

def main(json_file=None):
    if json_file is None:
        parser = argparse.ArgumentParser(description="解析实验结果 JSON 文件并打印指定指标")
        parser.add_argument("json_file", help="实验结果的 JSON 文件路径")
        args = parser.parse_args()

        # 读取 JSON 文件
        with open(args.json_file, "r") as f:
            data = json.load(f)
    else:
        with open(json_file, "r") as f:
            data = json.load(f)

    # 所有实验结果在 "results" 字段
    results = data.get("results", {})

    # 按照要求的顺序提取结果
    output = []
    # suffix = "_r1"
    suffix = ""
    
    # 1. vqa_rad_open: (recall_old, recall, precision, f1, bleu)
    task = "vqa_rad_open"+suffix
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 2. vqa_rad_closed: (acc 即 accuracy)
    task = "vqa_rad_closed"+suffix
    output.append(get_metric(results, task, "accuracy,none"))

    # 3. slake_open: (recall_old, recall, precision, f1, bleu)
    task = "slake_open"+suffix
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 4. slake_closed: (acc)
    task = "slake_closed"+suffix
    output.append(get_metric(results, task, "accuracy,none"))

    # 5. path_vqa_open: (recall_old, recall, precision, f1, bleu)
    task = "path_vqa_open"+suffix
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 6. path_vqa_closed: (acc)
    task = "path_vqa_closed"+suffix
    output.append(get_metric(results, task, "accuracy,none"))

    # 7. vqa_med_open: (从 vqa_med 中取 open 指标：recall_old, recall, precision, f1, bleu)
    task = "vqa_med"+suffix
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 8. vqa_med_closed: (acc 对应 accuracy,none, vqa_med 中并不存在此指标则输出 -1)
    # output.append(get_metric(results, task, "accuracy,none"))

    # 9. pmc_vqa: (acc)
    task = "pmc_vqa"+suffix
    output.append(get_metric(results, task, "accuracy,none"))

    # 10. omni_med_vqa_mini: (acc)
    task = "omni_med_vqa_mini"+suffix
    output.append(get_metric(results, task, "accuracy,none"))

    # 输出结果，使用 tab 键分隔，方便复制到 Excel 中
    print("\t".join(str(v) for v in output))

if __name__ == "__main__":
    main()
