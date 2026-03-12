"""
Utility functions for SuperGPQA Medicine evaluation.
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def filter_medicine(dataset):
    """Filter for medicine discipline only."""

    def is_medicine(doc):
        discipline = doc.get("discipline", "").lower()
        return discipline == "medicine"

    return dataset.filter(is_medicine)


def supergpqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for SuperGPQA."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options = doc.get("options", [])

    if options:
        options_text = "\n".join(
            [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
        )
        prompt = f"{question}\n{options_text}"
    else:
        prompt = question

    return f"{pre_prompt}{prompt}{post_prompt}"


def supergpqa_process_results(doc, results):
    """Process results for SuperGPQA."""
    pred = parse_reasoning_answer(results[0], strict=False)
    gt_label = doc.get("answer_letter", "")

    options = doc.get("options", [])
    num_options = len(options) if options else 4
    all_choices = [chr(ord("A") + i) for i in range(num_options)]

    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and pred_label == gt_label else 0

    return {"accuracy": acc}
