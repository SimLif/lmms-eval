import datetime
import json
import os
from collections import defaultdict

from loguru import logger as eval_logger

import sys
dir_name = os.path.dirname(os.path.abspath(__file__))


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.slake.metrics import calculate_exactmatch, calculate_f1score


replace_prompt = " Please answer yes or no."


def slake_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def slake_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def slake_open_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    # exact_match = calculate_exactmatch(pred_ans, gt_ans)
    f1_score, precision, recall = calculate_f1score(pred_ans, gt_ans)

    return {
        # "exact_match": exact_match,
        # "f1": f1_score,
        # "precision": precision,
        "recall": recall * 100,
    }


def slake_closed_process_results(doc, results):
    pred = results[0]
    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    if gt_ans not in pred_ans:
        return {"accuracy": 0}
    else:
        return {"accuracy": 100}
