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

def main(json_file=None, noise_type='image_gaussian_noise'):
    if json_file is None:
        parser = argparse.ArgumentParser(description="解析实验结果 JSON 文件并打印指定指标")
        parser.add_argument('-n', '--noise_type', type=str, default='image_gaussian_noise', help='噪声类型，默认值为 image_gaussian_noise')
        parser.add_argument("json_file", help="实验结果的 JSON 文件路径")
        args = parser.parse_args()
        noise_type = args.noise_type

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

    # 1. vqa_rad_open: (recall_old, recall, precision, f1, bleu)
    task = "vqa_rad_open"+"_"+noise_type
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 2. vqa_rad_closed: (acc 即 accuracy)
    task = "vqa_rad_closed"+"_"+noise_type
    output.append(get_metric(results, task, "accuracy,none"))

    # 3. slake_open: (recall_old, recall, precision, f1, bleu)
    task = "slake_open"+"_"+noise_type
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 4. slake_closed: (acc)
    task = "slake_closed"+"_"+noise_type
    output.append(get_metric(results, task, "accuracy,none"))

    # 5. path_vqa_open: (recall_old, recall, precision, f1, bleu)
    task = "path_vqa_open"+"_"+noise_type
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 6. path_vqa_closed: (acc)
    task = "path_vqa_closed"+"_"+noise_type
    output.append(get_metric(results, task, "accuracy,none"))

    # 7. vqa_med_open: (从 vqa_med 中取 open 指标：recall_old, recall, precision, f1, bleu)
    task = "vqa_med"+"_"+noise_type
    output.append(get_metric(results, task, "recall_old,none"))
    output.append(get_metric(results, task, "recall,none"))
    output.append(get_metric(results, task, "precision,none"))
    output.append(get_metric(results, task, "f1,none"))
    output.append(get_metric(results, task, "bleu,none"))

    # 8. vqa_med_closed: (acc 对应 accuracy,none, vqa_med 中并不存在此指标则输出 -1)
    # output.append(get_metric(results, task, "accuracy,none"))

    # 9. pmc_vqa: (acc)
    task = "pmc_vqa"+"_"+noise_type
    output.append(get_metric(results, task, "accuracy,none"))

    # 10. omni_med_vqa_mini: (acc)
    task = "omni_med_vqa_mini"+"_"+noise_type
    output.append(get_metric(results, task, "accuracy,none"))

    # 输出结果，使用 tab 键分隔，方便复制到 Excel 中
    print("\t".join(str(v) for v in output))

if __name__ == "__main__":
    # main()

    # # image gaussian noise
    # noise_type = 'image_gaussian_noise'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250510_201421_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-re-5epoch/20250512_225909_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch/20250514_035442_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-mmed-med-5epoch/20250514_181133_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-mmed-med-5epoch/20250515_010639_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-nano-s-tok-share-5epoch/20250518_201337_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch/20250511_185018_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch/20250511_200557_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch/20250511_211921_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch/20250511_223000_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250511_234449_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250514_144356_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250521_200543_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250521_200548_results.json'
    # ]

    # image rotation
    # noise_type = 'image_rotation'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-adamllm-med-full-5epoch/20250510_071221_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-4e2-med-ada-5epoch/20250510_080803_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-3e1-ada-med-5epoch/20250510_093013_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-16e8-med-nano-ds-tok-ada-pre-5epoch/20250510_104858_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch/20250510_115648_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250510_214520_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-re-5epoch/20250512_230005_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch/20250514_051054_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-mmed-med-5epoch/20250514_193708_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch/20250511_185123_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch/20250511_200622_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch/20250511_211742_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch/20250511_222932_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250511_234322_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250514_160804_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250521_200525_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250521_200537_results.json'
    # ]


    # # text char substitution
    # noise_type = 'text_char_substitution'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-adamllm-med-full-5epoch/20250509_190955_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-4e2-med-ada-5epoch/20250509_200603_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-3e1-ada-med-5epoch/20250509_213158_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-16e8-med-nano-ds-tok-ada-pre-5epoch/20250509_225448_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch/20250510_000422_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250510_231127_results.json',
    # ]

    # # text char substitution 0.05
    # noise_type = 'text_char_substitution_005'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-adamllm-med-full-5epoch/20250510_221432_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-4e2-med-ada-5epoch/20250510_231030_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-16e8-med-nano-ds-tok-ada-pre-5epoch/20250511_020906_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-3e1-ada-med-5epoch/20250511_004605_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch/20250511_031800_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250511_043007_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch/20250514_062704_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-mmed-med-5epoch/20250514_210917_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch/20250511_185543_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch/20250511_201303_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch/20250511_212833_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch/20250511_223941_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250511_235246_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250514_172619_results.json'
    # ]

    # # text word deletion
    # noise_type = 'text_word_deletion'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-adamllm-med-full-5epoch/20250510_221256_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-4e2-med-ada-5epoch/20250510_230638_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-16e8-med-nano-ds-tok-ada-pre-5epoch/20250511_015934_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-3e1-ada-med-5epoch/20250511_003103_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch/20250511_030734_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250511_041802_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch/20250514_074128_results.json'
    # ]

    noise_type = 'text_proportional_char_substitution'
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-adamllm-med-full-5epoch/20250514_075509_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-4e2-med-ada-5epoch/20250514_084942_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-16e8-med-nano-ds-tok-ada-pre-5epoch/20250514_113753_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-3e1-ada-med-5epoch/20250514_101552_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-ada-pre-5epoch/20250514_124717_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-ada-med-5epoch/20250514_135831_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-mmed-5epoch/20250514_074128_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-mmed-med-5epoch/20250514_210213_results.json'
    # ]
    # json_file_list = [
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-6e2-med-nano-ds-tok-share-5epoch/20250514_012111_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-12e4-med-nano-ds-tok-share-5epoch/20250514_024256_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-24e8-med-nano-ds-tok-share-5epoch/20250514_035506_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-48e16-med-nano-ds-tok-share-5epoch/20250514_050705_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250514_151822_results.json',
    #     '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250514_163403_results.json'
    # ]
    json_file_list = [
        '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-96e32-med-nano-ds-tok-share-5epoch/20250521_200553_results.json',
        '/mnt/data/haoqiang/workspace/02-lmms-eval/logs/checkpoints__qwen2-vl-2b-instruct-192e64-med-nano-ds-tok-share-5epoch/20250521_200559_results.json'
    ]


    for json_file in json_file_list:
        print('\n'+json_file)
        main(
            json_file=json_file,
            noise_type=noise_type
        )