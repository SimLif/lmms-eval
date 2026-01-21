import datetime
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

dir_name = os.path.dirname(os.path.abspath(__file__))


from lmms_eval.tasks.vqa_rad.metrics import calculate_exactmatch, calculate_f1score


# modified based on https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/utils/eval_utils.py
# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans, random_seed):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    random_flag = False  # whether is random selected answer
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)
    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.Random(random_seed).choice(all_choices)
        random_flag = True
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index, random_flag


def omni_med_vqa_mini_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def omni_med_vqa_mini_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["input"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def omni_med_vqa_mini_process_results(doc, results):
    all_choices = []
    index2ans = {}

    for key in ["option_A", "option_B", "option_C", "option_D"]:
        if doc[key]:
            all_choices.append(key[-1])
            index2ans[key[-1]] = doc[key]

    pred = results[0]
    gt_label = doc["label"]

    pred_label, random_flag = parse_multi_choice_response(pred, all_choices, index2ans, random_seed=0)
    # if random_flag:
    #     eval_logger.warning(f"Randomly selected answer: {pred_label} for question: {doc['input']}")
    #     acc = 0
    # else:
    acc = 100 if str(pred_label) == str(gt_label) else 0

    return {"accuracy": acc, "random_flag": random_flag}
