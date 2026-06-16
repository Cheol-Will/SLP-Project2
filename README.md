# SLP Project 2: Spoken Question Answering — Complete Requirements

---

## How to Run the Code End-to-End

This section describes how to reproduce the results from scratch.

### Prerequisites

- Python **≥ 3.13**
- CUDA-capable GPU with **≥ 16 GB VRAM** (for Qwen2.5-7B-Instruct in bfloat16)
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Install dependencies

```bash
uv sync
```

This installs all packages listed in `pyproject.toml`, including PyTorch 2.5.1 (CUDA 12.1), HuggingFace Transformers, sentence-transformers, librosa, etc.

### 2. Set up Hugging Face token

Whisper-large-v3 and Qwen2.5-7B-Instruct require a HuggingFace account token. Create a file named `.env` in the project root with your token on the first line:

```
HF_TOKEN="YOUR_API_KEY"
```

The pipeline reads this file automatically at startup.

### 3. Place the data

Unzip the dataset so the directory structure looks like:

```
data/
└── SLP_project02_data/
    ├── release_dev/
    │   ├── questions/   # q000.wav … q299.wav
    │   ├── documents/   # d000.wav … d499.wav
    │   └── gold.jsonl
    └── release/
        ├── questions/   # q500.wav … q799.wav
        └── documents/   # d500.wav … d999.wav
```

### 4. Run on the dev set (validation)

```bash
python main.py \
  --split dev \
  --top-k 4 \
  --prompt-path configs/prompt_final.json \
  --max_token 100 \
  --exp-name dev-final
```

This will:
1. **ASR** — transcribe all 300 questions and 500 documents with Whisper-large-v3. Transcripts are cached to `cache/transcripts/dev_questions.json` and `cache/transcripts/dev_documents.json`. Subsequent runs reuse the cache automatically.
2. **Retriever** — embed all document transcripts with bge-large-en-v1.5. Embeddings are cached to `cache/embeddings/dev_doc_embeddings.npy`. Query embeddings are computed fresh each run (fast).
3. **LLM** — generate answers with Qwen2.5-7B-Instruct using the prompt in `configs/prompt_final.json`.
4. **Evaluate** — accuracy and retrieval recall@k are printed and saved to `results/dev-final/metrics.json`.
5. **Output** — predictions saved to `results/dev-final/predictions_dev.jsonl`.

### 5. Run on the test set (final submission)

```bash
python main.py \
  --split test \
  --top-k 4 \
  --prompt-path configs/prompt_final.json \
  --max_token 100 \
  --exp-name test-final
```

The final `predictions.jsonl` (300 lines, q500–q799) is saved to:

```
results/test-final/predictions.jsonl
```


### 6. Final model configuration

| Component | Model | Parameters |
|---|---|---|
| ASR | `openai/whisper-large-v3` | 1.55B |
| Retriever | `BAAI/bge-large-en-v1.5` | 0.34B |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | 7.62B |
| **Total** | | **9.51B** |

Best dev-set result: **accuracy 0.500 (150/300)**, retrieval recall@4 0.837, using `configs/prompt_final.json` with `--top-k 4 --max_token 100`.
