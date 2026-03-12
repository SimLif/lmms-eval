"""
Utility functions for MMLU Medical evaluation.

Medical subjects (aligned with eval-kit):
  anatomy, clinical_knowledge, college_biology, college_medicine,
  medical_genetics, professional_medicine, nutrition, virology,
  high_school_biology
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response,
    parse_reasoning_answer,
)

CHOICES = ["A", "B", "C", "D"]


def mmlu_medical_doc_to_text(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> str:
    """Construct prompt for MMLU medical subjects."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()
    choices = doc["choices"]
    options_text = "\n".join(
        f"{CHOICES[i]}. {choices[i]}" for i in range(len(choices))
    )

    return f"{pre_prompt}{question}\n{options_text}{post_prompt}"


def mmlu_medical_doc_to_target(doc: dict) -> str:
    """Extract ground truth answer letter."""
    return CHOICES[doc["answer"]]


def mmlu_medical_process_results(doc: dict, results: list) -> dict:
    """Process results for MMLU medical subjects."""
    pred = parse_reasoning_answer(results[0], strict=False)
    pred_label = parse_multi_choice_response(pred, CHOICES)
    gt_label = CHOICES[doc["answer"]]

    acc = 100 if pred_label and pred_label == gt_label else 0
    return {"accuracy": acc}
