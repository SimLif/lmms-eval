"""
Utility functions for MedQA-USMLE evaluation.
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def medqa_usmle_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for MedQA-USMLE."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options = doc.get("options", {})

    if isinstance(options, dict):
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    elif isinstance(options, list):
        options_text = "\n".join(
            [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
        )
    else:
        options_text = ""

    prompt = f"{question}\n{options_text}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def medqa_usmle_process_results(doc, results):
    """Process results for MedQA-USMLE."""
    pred = parse_reasoning_answer(results[0], strict=False)
    gt_label = doc.get("answer_idx", "")

    all_choices = ["A", "B", "C", "D"]
    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and pred_label == gt_label else 0

    return {"accuracy": acc}
