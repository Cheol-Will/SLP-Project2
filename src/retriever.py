import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENCODE_BATCH = 32


def _load_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_ID, device=device)


def load_or_embed_docs(
    doc_transcripts: dict[str, str],
    emb_cache: Path,
    ids_cache: Path,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, doc_ids). Encodes and caches if not present."""
    emb_cache.parent.mkdir(parents=True, exist_ok=True)

    if not force and emb_cache.exists() and ids_cache.exists():
        embeddings = np.load(str(emb_cache))
        with open(ids_cache) as f:
            doc_ids = json.load(f)
        if len(doc_ids) == len(doc_transcripts):
            print(f"[Retriever] Doc embedding cache hit: {emb_cache} ({len(doc_ids)} docs)")
            return embeddings, doc_ids
        print("[Retriever] Doc count mismatch — re-embedding")

    doc_ids = sorted(doc_transcripts.keys())
    texts = [doc_transcripts[did] for did in doc_ids]

    print(f"[Retriever] Encoding {len(texts)} documents with {MODEL_ID}")
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalized for cosine via dot product
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    np.save(str(emb_cache), embeddings.astype(np.float32))
    with open(ids_cache, "w") as f:
        json.dump(doc_ids, f)
    print(f"[Retriever] Saved doc embeddings to {emb_cache}")

    return embeddings, doc_ids


def embed_queries(q_transcripts: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Encode all question transcripts. Returns (embeddings, q_ids)."""
    q_ids = sorted(q_transcripts.keys())
    texts = [QUERY_PREFIX + q_transcripts[qid] for qid in q_ids]

    print(f"[Retriever] Encoding {len(texts)} queries with {MODEL_ID}")
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings.astype(np.float32), q_ids


class BgeRetriever:
    """Stateless retriever: all embeddings pre-computed, retrieval is pure numpy."""

    def __init__(
        self,
        doc_embeddings: np.ndarray,  # shape (N_docs, D), L2-normalized
        doc_ids: list[str],
        q_embeddings: np.ndarray,    # shape (N_queries, D), L2-normalized
        q_ids: list[str],
    ):
        self.doc_embeddings = doc_embeddings
        self.doc_ids = doc_ids
        self.q_embeddings = q_embeddings
        self._q_index = {qid: i for i, qid in enumerate(q_ids)}

    def retrieve(self, qid: str, top_k: int) -> list[str]:
        q_vec = self.q_embeddings[self._q_index[qid]]   # (D,)
        scores = self.doc_embeddings @ q_vec              # (N_docs,)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.doc_ids[i] for i in top_indices]
