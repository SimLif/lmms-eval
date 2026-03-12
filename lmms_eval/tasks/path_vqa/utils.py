from lmms_eval.tasks._task_utils.answer_utils import parse_reasoning_answer
from lmms_eval.tasks._task_utils.judge_utils import is_judge_enabled, judge_binary
from lmms_eval.tasks._task_utils.vqa_metrics import (
    calculate_bleu,
    calculate_f1score,
    calculate_f1score_old,
)


def path_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def path_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs is None:
        return question

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"


def path_vqa_open_process_results(doc, results):
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
        judge_score = judge_binary(
            question=doc["question"],
            answer=doc["answer"],
            prediction=pred_ans,
        )
        metrics["llm_judge"] = judge_score
        # Mirror as accuracy so open+closed can be aggregated at group level
        metrics["accuracy"] = judge_score

    return metrics


def path_vqa_closed_process_results(doc, results):
    """Extract answer from \\boxed{} or <answer> tags, falling back to raw output."""
    pred = parse_reasoning_answer(results[0], strict=False)

    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["answer"].lower().strip().replace(".", "")

    if gt_ans not in pred_ans:
        return {"accuracy": 0}
    else:
        return {"accuracy": 100}
