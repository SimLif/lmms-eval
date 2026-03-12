"""
Utility functions for MedXpertQA evaluation.
"""

import re

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)


def medxpertqa_mm_doc_to_visual(doc):
    """Convert document image to PIL Image for MM version."""
    if "image" in doc and doc["image"] is not None:
        return [doc["image"].convert("RGB")]
    return []


def _count_options_in_question(question: str) -> int:
    """Count the number of answer choices embedded in the question text.

    The simwit datasets embed options as '(A) ... (B) ...' in the question.
    Returns the count of unique option letters found.
    """
    letters = re.findall(r"\([A-Z]\)", question)
    return len(set(letters))


def medxpertqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for MedXpertQA.

    The simwit datasets embed 'Answer Choices: (A)...(B)...' in the question
    field, so we use the question as-is without appending options again.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")

    return f"{pre_prompt}{question}{post_prompt}"


def medxpertqa_process_results(doc, results):
    """Process results for MedXpertQA."""
    pred = parse_reasoning_answer(results[0], strict=False)

    gt_label = doc.get("answer", doc.get("label", ""))

    # Count options from the question text (authoritative source)
    question = doc.get("question", "")
    num_options = _count_options_in_question(question)
    if num_options < 2:
        num_options = 10  # Default for MedXpertQA (A-J)

    all_choices = [chr(ord("A") + i) for i in range(num_options)]

    pred_label = parse_multi_choice_response(pred, all_choices)

    acc = 100 if pred_label and str(pred_label).upper() == str(gt_label).upper() else 0

    return {"accuracy": acc}
