import datetime
import json
import os
import random
import re
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

    content_match = re.search(r"<answer>(.*?)</answer>", response)
    answer = content_match.group(1).strip() if content_match else response.strip()

    return answer


def pmc_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def pmc_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["input"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def pmc_vqa_process_results(doc, results):
    all_choices = ["A", "B", "C", "D"]

    pred = results[0]
    gt_label = doc["label"]
    index2ans = {"A": doc["A"], "B": doc["B"], "C": doc["C"], "D": doc["D"]}

    pred_label = parse_multi_choice_response(pred, all_choices, index2ans, random_seed=0)
    acc = 100 if str(pred_label) == str(gt_label) else 0

    return {"accuracy": acc}
