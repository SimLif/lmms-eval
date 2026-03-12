"""
Utility functions for Medbullets evaluation.
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def _extract_options(doc: dict, num_choices: int) -> list[str]:
    """Extract options from a document, handling both dict and field formats.

    Supports:
    - options as dict: {"A": "text", "B": "text", ...}
    - options as individual fields: opa/opb/opc/opd/ope
    """
    letters = [chr(ord("A") + i) for i in range(num_choices)]

    # Try 'options' dict first (tuenguyen/Medical-Eval-MedBullets_op4 format)
    options_raw = doc.get("options", None)
    if isinstance(options_raw, dict):
        return [options_raw.get(letter, "") for letter in letters]

    # Fall back to individual fields (LangAGI-Lab/medbullets_op5 format)
    field_names = ["opa", "opb", "opc", "opd", "ope"][:num_choices]
    return [doc.get(field, "") for field in field_names]


def medbullets_op4_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for Medbullets op4."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options = _extract_options(doc, 4)

    options_text = "\n".join(
        [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options) if opt]
    )
    prompt = f"{question}\n{options_text}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def medbullets_op4_process_results(doc, results):
    """Process results for Medbullets op4."""
    pred = parse_reasoning_answer(results[0], strict=False)
    gt_label = doc.get("answer_idx", doc.get("answer", ""))

    all_choices = ["A", "B", "C", "D"]
    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and pred_label == gt_label else 0

    return {"accuracy": acc}


def medbullets_op5_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for Medbullets op5."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options = _extract_options(doc, 5)

    options_text = "\n".join(
        [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options) if opt]
    )
    prompt = f"{question}\n{options_text}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def medbullets_op5_process_results(doc, results):
    """Process results for Medbullets op5."""
    pred = parse_reasoning_answer(results[0], strict=False)
    gt_label = doc.get("answer_idx", doc.get("answer", ""))

    all_choices = ["A", "B", "C", "D", "E"]
    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and pred_label == gt_label else 0

    return {"accuracy": acc}
