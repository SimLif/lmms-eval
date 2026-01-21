def food101_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def food101_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["input"].strip()
    return question


def food101_process_results(doc, results):
    pred = results[0]
    pred_ans = pred.lower().strip().replace(".", "")
    gt_ans = doc["label"].lower().strip().replace(".", "")

    if gt_ans not in pred_ans:
        return {"accuracy": 0}
    else:
        return {"accuracy": 100}
