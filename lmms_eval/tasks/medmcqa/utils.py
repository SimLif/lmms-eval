"""
Utility functions for MedMCQA evaluation.
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def medmcqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for MedMCQA."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options = [
        doc.get("opa", ""),
        doc.get("opb", ""),
        doc.get("opc", ""),
        doc.get("opd", ""),
    ]

    options_text = "\n".join(
        [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
    )
    prompt = f"{question}\n{options_text}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def medmcqa_doc_to_target(doc):
    """Get the target answer."""
    cop = doc.get("cop", 0)  # 0-indexed (0=A, 1=B, 2=C, 3=D)
    return chr(ord("A") + cop)


def medmcqa_process_results(doc, results):
    """Process results for MedMCQA."""
    pred = parse_reasoning_answer(results[0], strict=False)
    cop = doc.get("cop", 0)  # 0-indexed (0=A, 1=B, 2=C, 3=D)
    gt_label = chr(ord("A") + cop)

    all_choices = ["A", "B", "C", "D"]
    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and pred_label == gt_label else 0

    return {"accuracy": acc}
