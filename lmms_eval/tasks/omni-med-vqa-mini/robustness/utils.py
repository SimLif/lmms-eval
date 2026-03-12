"""
Utility functions for OmniMedVQA-mini robustness evaluation.
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def omni_med_vqa_mini_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def omni_med_vqa_mini_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["input"].strip()
    if (
        "pre_prompt" in lmms_eval_specific_kwargs
        and lmms_eval_specific_kwargs["pre_prompt"] != ""
    ):
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if (
        "post_prompt" in lmms_eval_specific_kwargs
        and lmms_eval_specific_kwargs["post_prompt"] != ""
    ):
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def omni_med_vqa_mini_process_results(doc, results):
    """
    Unified process function: automatically extracts answer from \\boxed{} or <answer> tags.
    If no tags found, uses raw output (non-think mode behavior).
    """
    all_choices = []

    for key in ["option_A", "option_B", "option_C", "option_D"]:
        if doc[key]:
            all_choices.append(key[-1])

    pred_raw = results[0]
    # parse_reasoning_answer(strict=False) returns original text if no tags found
    pred = parse_reasoning_answer(pred_raw, strict=False)

    gt_label = doc["label"]

    pred_label = parse_multi_choice_response(pred, all_choices)
    acc = 100 if pred_label and str(pred_label) == str(gt_label) else 0

    return {"accuracy": acc}
