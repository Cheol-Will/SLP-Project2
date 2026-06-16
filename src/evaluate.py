"""Dev-set evaluation: answer accuracy + retrieval recall@k."""

import json
from pathlib import Path


def load_gold(path: Path) -> dict[str, dict]:
    """Return {qid: {document_id, answers: [str]}}."""
    gold = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            gold[obj["question_id"]] = {
                "document_id": obj["document_id"],
                "answers": [a.lower().strip() for a in obj["answer"]],
            }
    return gold


def score(predictions: list[dict], gold: dict) -> dict:
    """
    predictions: list of {question_id, answer, document_ids}
    Returns accuracy, retrieval_recall@k, and per-question detail.
    """
    correct = 0
    retrieval_hit = 0
    detail = []

    for pred in predictions:
        qid = pred["question_id"]
        if qid not in gold:
            continue
        g = gold[qid]
        pred_ans = pred["answer"].lower().strip()
        ans_ok = any(pred_ans == ga for ga in g["answers"])
        ret_ok = g["document_id"] in pred["document_ids"]

        correct += int(ans_ok)
        retrieval_hit += int(ret_ok)
        detail.append({
            "question_id": qid,
            "predicted": pred_ans,
            "gold_answers": g["answers"],
            "gold_doc": g["document_id"],
            "retrieved_docs": pred["document_ids"],
            "answer_correct": ans_ok,
            "retrieval_hit": ret_ok,
        })

    total = len(predictions)
    return {
        "accuracy": correct / total if total else 0.0,
        "retrieval_recall": retrieval_hit / total if total else 0.0,
        "correct": correct,
        "retrieval_hits": retrieval_hit,
        "total": total,
        "detail": detail,
    }


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*40}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
    print(f"  Retrieval recall@k: {metrics['retrieval_recall']:.4f}  ({metrics['retrieval_hits']}/{metrics['total']})")
    print(f"{'='*40}\n")

    # Print a sample of failures for analysis
    failures = [d for d in metrics["detail"] if not d["answer_correct"]][:10]
    if failures:
        print("Sample failures:")
        for f in failures:
            ret = "[HIT]" if f["retrieval_hit"] else "[MISS]"
            print(f"  {f['question_id']} {ret}  pred='{f['predicted']}'  gold={f['gold_answers']}")
