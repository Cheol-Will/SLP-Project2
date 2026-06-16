# SLP Project 2: Spoken Question Answering — Complete Requirements
---

## 1. Task Definition

**Goal:** Build an **ASR -> Retriever -> LLM** three-stage cascade that answers spoken questions based on a corpus of spoken documents.

- **Data:**
  - Validation set: `data/release_dev` with labels (gold.jsonl).
  - Test set: `data/release/*/*.wav` without labels and this will be evaluated. 
  - We choose our final model and hyperparameters from validation set.

- **Input:**
  - spoken questions (`data/release_dev/questions/q*.wav`)
  - spoken documents (`data/release_dev/documents/d*.wav`)
  - Both question audio and document audio are **16 kHz mono waveforms**
  - The corpus contains the gold (correct) document for every question, **plus distractors**

- **Output (per question):**
  - A **short text answer**
  - The **list of document IDs** that were actually included in the LLM prompt

> **NOT** a transcription task. Transcripts are an intermediate artifact. The system is **evaluated on answer correctness**, not transcription quality. Selecting a small, highly relevant set of documents is explicitly part of the design goal.

---

## 2. Pipeline Architecture

The required architecture is a strict three-stage cascade. Each stage feeds into the next:

```
Audio (question) ──► ASR ──► question transcript ──► Retriever ──► top-k doc IDs ──► LLM ──► answer (text)
Audio (documents) ─► ASR ──► document transcripts ──────────────────────────────────────►
```

**Use below models**
Whisper-large-v3 (1.55B) bge-large-en-v1.5 (0.34B) Qwen2.5-7B-Instruct (7.62B) 9.51B
- Convert speech (questions and documents) to text
- Rank document transcripts by relevance to the query transcript 
- Generate a short answer using the selected document(s) as context

---

## 3. Data / Input Specification

### Directory structure

```
release_dev/               ← Dev set (use for local development and validation)
  questions/q000.wav
  questions/q001.wav
  ...
  questions/q299.wav       ← 300 question audio files
  documents/d000.wav
  documents/d001.wav
  ...
  documents/d499.wav       ← 500 document audio files
  gold.jsonl               ← Reference labels (DEV ONLY)

release/                   ← Test set (submit predictions for this)
  questions/q500.wav
  ...
  questions/q799.wav       ← 300 question audio files
  documents/d500.wav
  ...
  documents/d999.wav       ← 500 document audio files
                           ← NO gold labels provided
```

### Audio format
- All files: **16 kHz, mono**

---

## 4. Stage 1 — ASR (Automatic Speech Recognition)

### Functional requirement
- Convert every audio file (questions **and** documents) into a plain text transcript for caching.
- Input: 16 kHz mono waveform
- Output: text transcript of the spoken content

### Design constraints
- ASR quality **directly bounds** the rest of the pipeline:
  - Substitution and deletion errors corrupt the retriever's input.
  - Capitalization, punctuation, and spelling vary across models and must be handled consistently.
  - Pretrained text models downstream need a transcript (or a speech-aware interface).

### Implementation recommendations (from slides)
- **Transcribe all documents once** and **cache all transcripts to disk** — do not re-run ASR every time.
- For the same ASR model, use the cached transcripts during retrieval and LLM generation.
- Cache documents and transcripts once

---

## 5. Stage 2 — Retriever

### Functional requirement
- Given the question transcript, **rank all document transcripts by relevance** and return the top-k most relevant document IDs.

### Option A: Lexical retrieval
- Methods: **BM25**, TF-IDF
- Scores documents by term-frequency statistics over shared tokens.
- **Parameter-free → counts as 0B toward the parameter budget.**
- Strong baseline for factoid questions with named entities.

### Option B: Dense retrieval
- Methods: **bge**, **e5**, **gte** embedding models
- Encode questions and documents into a shared embedding space.
- Score by cosine similarity or dot product.
- Better than lexical retrieval at handling paraphrasing and lexical mismatch.
- **Document embeddings can (and should) be indexed once** and reused. Use cache so that we can try varying top-k.

### Key design decisions
- Choose top-k carefully (see Section 7: Minimal Document Inclusion).
- Hybrid approaches (combining lexical + dense) are not prohibited.

---

## 6. Stage 3 — LLM (Retrieval-Augmented Generation)

### Functional requirement
- Feed the top retrieved document transcripts as context to an LLM.
- The LLM generates a **short text answer** grounded on the retrieved documents.

### Constraints (strictly from slides)
- **No fine-tuning is permitted.**
- Only **prompting** and **in-context examples** are allowed.
- Instruction-tuned models tend to give verbose answers by default → explicitly constrain the output in the prompt (e.g., *"Answer with the shortest correct span."*)
- Suggested models: `Qwen2.5`, `Llama-3.1`, `Mistral`

### k selection (number of documents passed to LLM)
- Small k: keeps the LLM focused, but risks missing the gold document.
- Large k: increases recall, but lengthens the prompt, can distract the model, and increases latency.
- **Recommended setting: 3–5 documents** (see Section 7).

---

## 7. Minimal Document Inclusion

- **Principle: include only documents the system believes are necessary.**
- Passing all 500 documents to the LLM defeats the purpose of the retriever.
- A larger top-k may improve recall but increases latency and can distract the LLM.
- **Recommended setting: 3–5 documents.**
- `document_ids` in the output must be the IDs **actually used in the LLM prompt**, not the raw retriever top-k list.

---

## 8. Parameter Budget

- **Hard limit: ≤ 10B total parameters** across all three components combined.
- What to count: ASR encoder + decoder, all retriever checkpoint(s), LLM including its embedding weights.
- Exceeding 10B → **0 points** for the model size component (5 pts at stake).
- **Open-weight models only.** Closed-API models (GPT, Claude, Gemini, etc.) are **not allowed**.

### Example feasible configurations (from slides)

| ASR | Retriever | LLM | Total |
|---|---|---|---|
| Whisper-large-v3 (1.55B) | bge-large-en-v1.5 (0.34B) | Qwen2.5-7B-Instruct (7.62B) | **9.51B** |
| Whisper-large-v3 (1.55B) | BM25 (0B) | Llama-3.1-8B-Instruct (8.03B) | **9.58B** |
| Whisper-medium.en (0.77B) | bge-base-en (0.11B) | Qwen2.5-7B-Instruct (7.62B) | **8.50B** |
| Distil-Whisper-large-v3 (0.76B) | bge-small-en (0.03B) | Phi-3.5-mini (3.82B) | **4.61B** |

---

## 9. Output Format

### File: `predictions.jsonl`
- One JSON object per line
- Exactly **300 lines** total (one per question in `release/`)
- Keys: `question_id`, `answer`, `document_ids`

### Example
```jsonl
{"question_id":"q500","answer":"pakistan","document_ids":["d686"]}
{"question_id":"q501","answer":"sigmund freud","document_ids":["d838","d502"]}
```

### Rules
- `question_id`: matches the filename stem (e.g., `q500` for `q500.wav`), covers q500–q799.
- `answer`: **short-form text**, not a full sentence (e.g., a name, date, place, entity).
- `document_ids`: the document IDs **used in the LLM prompt** — not the raw retriever ranking list.
- Follow the exact format used in `release_dev/` predictions.

---

## 10. Evaluation

### Scoring breakdown (total: 30 points)

### Answer correctness rules
- Each prediction is judged **correct or incorrect** (binary).
- **Semantically equivalent answers are considered correct** (e.g., "Pakistan" == "pakistan").
- Answers must be in **short form**, not full sentences.

### Local evaluation (dev set)
- Use `gold.jsonl` from `release_dev/` to measure accuracy during development (validation set).
- The gold file is only available for the dev split (q000–q299 / d000–d499).

---

## 11. Submission Requirements

### Deadline
**2026-06-18 (Thu) 23:59**

### Files to submit on iCampus (3 separate uploads)

| File | Description |
|---|---|
| `2026xxxxxx-project2.zip` | Entire project code, **without model weights** |
| `2026xxxxxx-project2-report.pdf` | Brief report (PDF format) |
| `2026xxxxxx-project2-predictions.jsonl` | Final predictions for `release/` |

### `predictions.jsonl` requirements
- Must cover **every question ID** in `release/` (q500–q799).
- Must follow given format.
- Exactly **300 lines**, one JSON object per line.

### Project zip requirements
- Must include a **`README.md`** describing how to run the code end-to-end.
- Do **not** include model weights.

### Report requirements
- Format: **PDF**
- Length: **≤ 6 pages**
- Font: **11pt** for main text
- Must include all four of the following sections:
  1. **Chosen models** with total parameter count
  2. **Retrieval strategy** description
  3. **Prompt design** (the exact or representative prompt used)
  4. **Failure case analysis** (examples where the system failed and why)

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
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
python run_pipeline.py \
  --split dev \
  --top-k 4 \
  --prompt-path configs/prompt30.json \
  --max_token 100 \
  --exp-name dev-final
```

This will:
1. **ASR** — transcribe all 300 questions and 500 documents with Whisper-large-v3. Transcripts are cached to `cache/transcripts/dev_questions.json` and `cache/transcripts/dev_documents.json`. Subsequent runs reuse the cache automatically.
2. **Retriever** — embed all document transcripts with bge-large-en-v1.5. Embeddings are cached to `cache/embeddings/dev_doc_embeddings.npy`. Query embeddings are computed fresh each run (fast).
3. **LLM** — generate answers with Qwen2.5-7B-Instruct using the prompt in `configs/prompt30.json`.
4. **Evaluate** — accuracy and retrieval recall@k are printed and saved to `results/dev-final/metrics.json`.
5. **Output** — predictions saved to `results/dev-final/predictions_dev.jsonl`.

### 5. Run on the test set (final submission)

```bash
python run_pipeline.py \
  --split test \
  --top-k 4 \
  --prompt-path configs/prompt30.json \
  --max_token 100 \
  --exp-name test-final
```

The final `predictions.jsonl` (300 lines, q500–q799) is saved to:

```
results/test-final/predictions.jsonl
```

This is the file to submit to iCampus.

### 6. Final model configuration

| Component | Model | Parameters |
|---|---|---|
| ASR | `openai/whisper-large-v3` | 1.55B |
| Retriever | `BAAI/bge-large-en-v1.5` | 0.34B |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | 7.62B |
| **Total** | | **9.51B** |

Best dev-set result: **accuracy 0.500 (150/300)**, retrieval recall@4 0.837, using `configs/prompt30.json` with `--top-k 4 --max_token 100`.

### 7. Useful flags

| Flag | Description |
|---|---|
| `--force-asr` | Re-run ASR even if transcripts are already cached |
| `--force-embed` | Re-embed documents even if embeddings are already cached |
| `--top-k N` | Number of documents passed to the LLM (explored: 3–7) |
| `--prompt-path PATH` | Path to prompt config JSON (see `configs/`) |
| `--max_token N` | Maximum new tokens for LLM generation |

### 8. Utility scripts

```bash
# Summarise all experiments sorted by accuracy
python summary.py

# Generate failure case study JSON for a specific experiment
python analyze.py --exp dev-final

# Generate failure case studies for all experiments at once
python analyze.py
```

Failure case studies are saved to `results/analysis/<exp_name>/case_study.json`. Each entry includes the question text, predicted answer, gold answer, gold document text, and all retrieved document texts.

---

## 12. Summary Checklist

### Code
- [ ] ASR: transcribes all 300 questions + 500 documents, caches to disk
- [ ] Retriever: indexes document embeddings once; queries at inference time
- [ ] LLM: prompt-only (no fine-tuning), outputs short-form answers
- [ ] Total loaded params verified ≤ 10B
- [ ] Only open-weight models used (no closed APIs)
- [ ] `run_pipeline.py` (or equivalent) produces `predictions.jsonl` for `release/`
- [ ] `README.md` explains how to reproduce results end-to-end

### Output
- [ ] `predictions.jsonl`: exactly 300 lines, q500–q799, correct JSON format
- [ ] `document_ids` = IDs actually passed to LLM prompt (not raw top-k list)
- [ ] Answers are short-form (not full sentences)

### Submission
- [ ] `2026xxxxxx-project2.zip` (no model weights)
- [ ] `2026xxxxxx-project2-report.pdf` (≤6 pages, 11pt, PDF)
- [ ] `2026xxxxxx-project2-predictions.jsonl`
- [ ] Submitted on iCampus before **2026-06-18 23:59**

### Report
- [ ] Chosen models listed with exact parameter counts (sum ≤ 10B)
- [ ] Retrieval strategy described
- [ ] Prompt design shown (exact prompt template)
- [ ] Failure case analysis included