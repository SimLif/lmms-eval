import datetime
import json
import os
import sys
from collections import defaultdict

from loguru import logger as eval_logger

dir_name = os.path.dirname(os.path.abspath(__file__))


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_metrics import (
    calculate_bleu,
    calculate_exactmatch,
    calculate_f1score,
    calculate_f1score_old,
)
from lmms_eval.tasks._task_utils.answer_utils import parse_reasoning_answer


def vqa_med_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vqa_med_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def vqa_med_process_results(doc, results):
    """
    Unified process function: automatically extracts answer from \\boxed{} or <answer> tags.
    If no tags found, uses raw output (non-think mode behavior).
    """
    pred_raw = results[0]
    # parse_reasoning_answer(strict=False) returns original text if no tags found
    pred = parse_reasoning_answer(pred_raw, strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    f1_score, precision, recall = calculate_f1score(pred_ans, gt_ans)
    _, _, recall_old = calculate_f1score_old(pred_ans, gt_ans)
    bleu_score = calculate_bleu(pred_ans, gt_ans)

    return {
        "f1": f1_score * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "recall_old": recall_old * 100,
        "bleu": bleu_score * 100,
    }
