"""
Utility functions for OmniMedVQA evaluation (full dataset).
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def omni_med_vqa_doc_to_visual(doc):
    """Convert document image to PIL Image."""
    return [doc["image"].convert("RGB")]


def omni_med_vqa_doc_to_target(doc):
    """Get the correct answer label (A/B/C/D) from gt_answer."""
    gt_answer = doc.get("gt_answer", "")
    for opt in ["A", "B", "C", "D"]:
        opt_text = doc.get(f"option_{opt}", "")
        if opt_text == gt_answer:
            return opt
    return ""


def omni_med_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for OmniMedVQA with question and options."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Get question text
    question = doc.get("question", doc.get("input", ""))
    if not question:
        question = doc.get("Question", "")

    # Build options string from option_A/B/C/D fields
    options = []
    for opt in ["A", "B", "C", "D"]:
        opt_text = doc.get(f"option_{opt}", "")
        if opt_text:
            options.append(f"{opt}. {opt_text}")

    # Construct full input: question + options + prompt
    if options:
        options_str = "\n".join(options)
        full_input = f"{question}\n{options_str}"
    else:
        full_input = question

    return f"{pre_prompt}{full_input}{post_prompt}"


def omni_med_vqa_process_results(doc, results):
    """
    Unified process function: automatically extracts answer from \\boxed{} or <answer> tags.
    If no tags found, uses raw output (non-think mode behavior).
    """
    all_choices = []
    gt_label = None

    # Build options mapping and find gt_label from gt_answer
    gt_answer = doc.get("gt_answer", "")
    for opt in ["A", "B", "C", "D"]:
        opt_text = doc.get(f"option_{opt}", "")
        if opt_text:
            all_choices.append(opt)
            # Match gt_answer to find gt_label
            if opt_text == gt_answer:
                gt_label = opt

    # Fallback: try other field naming conventions
    if not all_choices:
        option_keys = [
            ("A", "B", "C", "D"),
            ("opa", "opb", "opc", "opd"),
        ]
        for keys in option_keys:
            found = False
            for i, key in enumerate(keys):
                if key in doc and doc[key]:
                    letter = chr(ord("A") + i)
                    all_choices.append(letter)
                    found = True
            if found:
                break
        gt_label = doc.get("label", doc.get("answer", ""))

    pred_raw = results[0]
    # parse_reasoning_answer(strict=False) returns original text if no tags found
    pred = parse_reasoning_answer(pred_raw, strict=False)

    if not all_choices:
        all_choices = ["A", "B", "C", "D"]

    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100.0 if pred_label and str(pred_label) == str(gt_label) else 0.0

    return {
        "accuracy": {
            "score": acc,
            "dataset": doc.get("dataset", ""),
        }
    }


def omni_med_vqa_aggregate_accuracy(results: list[dict]) -> float:
    """Extract score from dicts and compute mean accuracy."""
    scores = [r["score"] for r in results]
    return sum(scores) / len(scores) if scores else 0.0
