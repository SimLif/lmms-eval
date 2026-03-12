"""
Utility functions for MIMIC-CXR report generation evaluation.
"""

from PIL import Image

from lmms_eval.tasks._task_utils.report_metrics import (
    calculate_single_ratescore,
    compute_report_metrics,
)


def mimic_cxr_doc_to_visual(doc):
    """Convert document image to PIL Image."""
    if isinstance(doc["image"], Image.Image):
        return [doc["image"].convert("RGB")]
    return [Image.open(doc["image"]).convert("RGB")]


def mimic_cxr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for report generation."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{post_prompt}"


def mimic_cxr_process_results(doc, results):
    """Process results for MIMIC-CXR report generation."""
    pred = results[0].strip()
    gold = doc["report"].strip()

    metrics = compute_report_metrics(pred, gold)
    metrics["ratescore"] = calculate_single_ratescore(pred, gold)
    return metrics
