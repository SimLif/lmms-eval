"""
Utility functions for PubMedQA evaluation.

Dataset: openlifescienceai/pubmedqa
Format: Nested data with Context, Question, Options (A/B/C: yes/no/maybe)
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)

CHOICES = ["A", "B", "C"]


def pubmedqa_doc_to_text(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> str:
    """Construct prompt for PubMedQA."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    data = doc["data"]
    context = data["Context"]
    if isinstance(context, list):
        context = "\n".join(context)

    question = data["Question"]
    options = data["Options"]

    # Build options text: A.yes B.no C.maybe
    options_text = "\n".join(
        f"{letter}. {options[letter]}" for letter in CHOICES
    )

    prompt = f"Context: {context}\n\nQuestion: {question}\n{options_text}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def pubmedqa_doc_to_target(doc: dict) -> str:
    """Get the target answer letter."""
    return doc["data"]["Correct Option"]


def pubmedqa_process_results(doc: dict, results: list) -> dict:
    """Process results for PubMedQA."""
    pred = parse_reasoning_answer(results[0], strict=False)
    pred_label = parse_multi_choice_response(pred, CHOICES)
    gt_label = doc["data"]["Correct Option"]

    acc = 100 if pred_label and pred_label == gt_label else 0
    return {"accuracy": acc}
