"""
Spoken QA pipeline: ASR → Retrieval → LLM generation.

Usage:
  python run_pipeline.py --split dev  --top-k 5 --exp-name dev_k5
  python run_pipeline.py --split test --top-k 5 --exp-name test_k5_final
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Load HF token from .env if present (raw token on first line)
def _load_hf_token():
    for candidate in [Path(".env"), Path("../SLP-Project2/.env")]:
        if candidate.exists():
            token = candidate.read_text().strip().splitlines()[0].strip()
            if token.startswith("hf_"):
                os.environ.setdefault("HF_TOKEN", token)
                return
_load_hf_token()

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "SLP_project02_data"
CACHE_ROOT = ROOT / "cache"
RESULTS_ROOT = ROOT / "results"

SPLIT_CONFIG = {
    "dev": {
        "questions_dir": DATA_ROOT / "release_dev" / "questions",
        "documents_dir": DATA_ROOT / "release_dev" / "documents",
        "q_ids": [f"q{i:03d}" for i in range(300)],
        "d_ids": [f"d{i:03d}" for i in range(500)],
        "gold_path": DATA_ROOT / "release_dev" / "gold.jsonl",
        "pred_filename": "predictions_dev.jsonl",
    },
    "test": {
        "questions_dir": DATA_ROOT / "release" / "questions",
        "documents_dir": DATA_ROOT / "release" / "documents",
        "q_ids": [f"q{i:03d}" for i in range(500, 800)],
        "d_ids": [f"d{i:03d}" for i in range(500, 1000)],
        "gold_path": None,
        "pred_filename": "predictions.jsonl",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--prompt-path", type=Path, default=ROOT / "configs" / "prompt1.json",
                   help="Path to prompt config JSON (default: configs/prompt1.json)")
    p.add_argument("--max_token", type=int, default=80,
                   help="Max new tokens for LLM generation (default: 80)")
    p.add_argument("--force-asr", action="store_true", help="Re-run ASR even if cached")
    p.add_argument("--force-embed", action="store_true", help="Re-embed docs even if cached")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SPLIT_CONFIG[args.split]
    exp_dir = RESULTS_ROOT / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    from src.llm import load_prompt_config
    prompt_cfg = load_prompt_config(args.prompt_path)

    config = {
        "split": args.split,
        "top_k": args.top_k,
        "exp_name": args.exp_name,
        "prompt_path": str(args.prompt_path),
        "max_token": args.max_token,
        "asr_model": "openai/whisper-large-v3",
        "retriever_model": "BAAI/bge-large-en-v1.5",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": prompt_cfg,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(config)
    from src.asr import load_or_transcribe

    q_transcripts = load_or_transcribe(
        audio_dir=cfg["questions_dir"],
        cache_path=CACHE_ROOT / "transcripts" / f"{args.split}_questions.json",
        expected_ids=cfg["q_ids"],
        force=args.force_asr,
    )
    d_transcripts = load_or_transcribe(
        audio_dir=cfg["documents_dir"],
        cache_path=CACHE_ROOT / "transcripts" / f"{args.split}_documents.json",
        expected_ids=cfg["d_ids"],
        force=args.force_asr,
    )

    from src.retriever import load_or_embed_docs, embed_queries, BgeRetriever

    doc_embs, doc_ids = load_or_embed_docs(
        doc_transcripts=d_transcripts,
        emb_cache=CACHE_ROOT / "embeddings" / f"{args.split}_doc_embeddings.npy",
        ids_cache=CACHE_ROOT / "embeddings" / f"{args.split}_doc_ids.json",
        force=args.force_embed,
    )
    q_embs, q_ids = embed_queries(q_transcripts)

    retriever = BgeRetriever(doc_embs, doc_ids, q_embs, q_ids)

    from src.llm import QwenRAG

    llm = QwenRAG(prompt_cfg, max_new_tokens=args.max_token)
    predictions = []

    for qid in tqdm(q_ids):
        top_doc_ids = retriever.retrieve(qid, top_k=args.top_k)
        docs = [(did, d_transcripts[did]) for did in top_doc_ids]
        answer = llm.generate(question=q_transcripts[qid], docs=docs)

        predictions.append({
            "question_id": qid,
            "answer": answer,
            "document_ids": top_doc_ids,
        })

    pred_path = exp_dir / cfg["pred_filename"]
    with open(pred_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\n[Pipeline] Predictions saved to {pred_path}")

    if args.split == "dev" and cfg["gold_path"] and cfg["gold_path"].exists():
        from src.evaluate import load_gold, score, print_metrics

        gold = load_gold(cfg["gold_path"])
        metrics = score(predictions, gold)
        print_metrics(metrics)

        metrics_out = {k: v for k, v in metrics.items() if k != "detail"}
        metrics_out["top_k"] = args.top_k
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"[Pipeline] Metrics saved to {exp_dir / 'metrics.json'}")

        # Save full detail for failure analysis
        with open(exp_dir / "eval_detail.jsonl", "w") as f:
            for row in metrics["detail"]:
                f.write(json.dumps(row) + "\n")

    print(f"[Pipeline] Done. Results in {exp_dir}/")


if __name__ == "__main__":
    main()
