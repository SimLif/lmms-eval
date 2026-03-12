"""
Utility functions for IU-XRAY report generation evaluation.
"""

from PIL import Image

from lmms_eval.tasks._task_utils.report_metrics import (
    calculate_single_ratescore,
    compute_report_metrics,
)


def iu_xray_doc_to_visual(doc):
    """Convert document image to PIL Image."""
    if isinstance(doc["image"], Image.Image):
        return [doc["image"].convert("RGB")]
    return [Image.open(doc["image"]).convert("RGB")]


def iu_xray_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct prompt for report generation."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{post_prompt}"


def iu_xray_process_results(doc, results):
    """Process results for IU-XRAY report generation."""
    pred = results[0].strip()
    gold = doc["report"].strip()

    metrics = compute_report_metrics(pred, gold)
    metrics["ratescore"] = calculate_single_ratescore(pred, gold)
    return metrics
