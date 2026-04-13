"""
Utility functions for PathVQA Mini evaluation (sampled dataset).
"""

from lmms_eval.tasks._task_utils.answer_utils import (
    parse_reasoning_answer,
)
from lmms_eval.tasks._task_utils.judge_utils import (
    is_judge_enabled,
    judge_binary,
)
from lmms_eval.tasks._task_utils.vqa_metrics import (
    calculate_bleu,
    calculate_f1score,
    calculate_f1score_old,
)


def path_vqa_mini_doc_to_visual(doc: dict) -> list:
    """Convert document image to PIL Image."""
    return [doc["image"].convert("RGB")]


def path_vqa_mini_doc_to_text(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> str:
    """Construct prompt from question with optional pre/post prompts."""
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs is None:
        return question

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"


def path_vqa_mini_open_process_results(
    doc: dict,
    results: list,
) -> dict:
    """Process open-ended VQA results with F1, BLEU, and optional judge.

    Extracts answer from \\boxed{} or <answer> tags, falling back
    to raw output.
    """
    pred = parse_reasoning_answer(results[0], strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    f1_score, precision, recall = calculate_f1score(
        pred_ans, gt_ans
    )
    _, _, recall_old = calculate_f1score_old(pred_ans, gt_ans)
    bleu_score = calculate_bleu(pred_ans, gt_ans)

    metrics: dict = {
        "f1": f1_score * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "recall_old": recall_old * 100,
        "bleu": bleu_score * 100,
    }

    if is_judge_enabled():
        judge_result = judge_binary(
            question=doc["question"],
            answer=doc["answer"],
            prediction=pred_ans,
        )
        metrics["llm_judge"] = judge_result["score"]
        metrics["judge_model"] = judge_result["judge_model"]
        metrics["judge_raw_response"] = judge_result["judge_raw_response"]
        # Mirror as accuracy so open+closed can be aggregated
        # at group level
        metrics["accuracy"] = judge_result["score"]

    return metrics


def path_vqa_mini_closed_process_results(
    doc: dict,
    results: list,
) -> dict:
    """Process closed-ended VQA results with exact match accuracy.

    Extracts answer from \\boxed{} or <answer> tags, falling back
    to raw output.
    """
    pred = parse_reasoning_answer(results[0], strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    if gt_ans not in pred_ans:
        return {"accuracy": 0}
    else:
        return {"accuracy": 100}
