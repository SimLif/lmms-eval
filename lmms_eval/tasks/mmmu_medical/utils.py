"""
Utility functions for MMMU Medical subset.

Overrides mmmu_process_results to add per-sample ``score`` field
so that the framework can compute Clustered SE (cluster_key=subdomain).
"""

import ast

# Import all functions from the main MMMU utils
from lmms_eval.tasks.mmmu.utils import *  # noqa: F403
from lmms_eval.tasks.mmmu.utils import (
    eval_multi_choice,
    eval_open,
    extract_subset_name,
    get_multi_choice_info,
    parse_multi_choice_response,
    parse_open_response,
)

# Medical subjects in MMMU
MEDICAL_SUBJECTS = [
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
]


def filter_medical(dataset):
    """Filter for medical-related subjects only."""

    def is_medical(doc):
        # Extract subject from the document id
        # Format: validation_Subject_Number or test_Subject_Number
        doc_id = doc.get("id", "")
        for subject in MEDICAL_SUBJECTS:
            if subject in doc_id:
                return True
        return False

    return dataset.filter(is_medical)


def mmmu_process_results(doc, results):
    """Override upstream to add ``score`` for Clustered SE.

    The extra ``score`` field (0.0 / 100.0) is ignored by the existing
    ``mmmu_aggregate_results`` but consumed by ``calculate_clt_aggregate_metric``.
    """
    parsed_preds = []
    for pred in results:
        if doc["question_type"] == "multiple-choice":
            index2ans, all_choices = get_multi_choice_info(
                ast.literal_eval(doc["options"])
            )
            parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        else:
            parsed_pred = parse_open_response(pred)
            parsed_pred = str(parsed_pred[0]) if parsed_pred else ""
        parsed_preds.append(parsed_pred)

    # Compute per-sample correctness (same logic as evaluate_mmmu)
    correct = False
    for pred_i in parsed_preds:
        if doc["question_type"] == "multiple-choice":
            correct = eval_multi_choice(doc["answer"], pred_i)
        else:
            correct = eval_open(doc["answer"], [pred_i])
        if correct:
            break

    subdomain = extract_subset_name(doc["id"])
    mmmu_exact_acc = {
        "id": doc["id"],
        "subdomain": subdomain,
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "parsed_pred": parsed_preds,
        "score": 100.0 if correct else 0.0,
    }
    mmmu_submission = {doc["id"]: parsed_preds[0]}
    return {
        "mmmu_acc": mmmu_exact_acc,
        "mmmu_acc_pass_at_k": mmmu_exact_acc,
        "submission": mmmu_submission,
    }
