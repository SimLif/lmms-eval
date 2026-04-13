from lmms_eval.tasks._task_utils.answer_utils import parse_reasoning_answer
from lmms_eval.tasks._task_utils.judge_utils import is_judge_enabled, judge_binary
from lmms_eval.tasks._task_utils.vqa_metrics import (
    calculate_bleu,
    calculate_f1score,
    calculate_f1score_old,
)


def slake_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def slake_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build prompt with bilingual support (en/zh).

    Uses q_lang field to select appropriate prompt:
    - For English: uses post_prompt_en (or post_prompt as fallback)
    - For Chinese: uses post_prompt_zh (or post_prompt as fallback)
    """
    question = doc["question"].strip()
    lang = doc.get("q_lang", "en")

    if lmms_eval_specific_kwargs is None:
        return question

    # Get pre_prompt (language-specific or fallback)
    pre_prompt = ""
    if f"pre_prompt_{lang}" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs[f"pre_prompt_{lang}"]
    elif "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    # Get post_prompt (language-specific or fallback)
    post_prompt = ""
    if f"post_prompt_{lang}" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs[f"post_prompt_{lang}"]
    elif "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def slake_open_process_results(doc, results):
    """Extract answer from \\boxed{} or <answer> tags, falling back to raw output."""
    pred = parse_reasoning_answer(results[0], strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    f1_score, precision, recall = calculate_f1score(pred_ans, gt_ans)
    _, _, recall_old = calculate_f1score_old(pred_ans, gt_ans)
    bleu_score = calculate_bleu(pred_ans, gt_ans)

    metrics = {
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
        # Mirror as accuracy so open+closed can be aggregated at group level
        metrics["accuracy"] = judge_result["score"]

    return metrics


def slake_closed_process_results(doc, results):
    """Extract answer from \\boxed{} or <answer> tags, falling back to raw output."""
    pred = parse_reasoning_answer(results[0], strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    if gt_ans not in pred_ans:
        return {"accuracy": 0}
    else:
        return {"accuracy": 100}
