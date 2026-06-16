"""Stage 1: ASR using Whisper-large-v3 with disk caching."""

import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

MODEL_ID = "openai/whisper-large-v3"
BATCH_SIZE = 8
SAMPLE_RATE = 16_000


def load_or_transcribe(
    audio_dir: Path,
    cache_path: Path,
    expected_ids: list[str],
    force: bool = False,
) -> dict[str, str]:
    """Return {audio_id: transcript}. Runs ASR only for IDs missing from cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, str] = {}
    if not force and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        missing = [aid for aid in expected_ids if aid not in cache]
        if not missing:
            print(f"[ASR] Cache hit: {cache_path} ({len(cache)} transcripts)")
            return cache
        print(f"[ASR] Cache partial: {len(missing)} IDs missing from {cache_path}")
    else:
        missing = expected_ids
        print(f"[ASR] Transcribing {len(missing)} files from {audio_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        chunk_length_s=30,   # split audio >30s into overlapping chunks automatically
        model_kwargs=model_kwargs,
    )

    # Load all audio upfront to avoid pipeline overhead per sample
    audio_inputs = []
    for aid in tqdm(missing, desc="Loading audio"):
        wav, _ = librosa.load(str(audio_dir / f"{aid}.wav"), sr=SAMPLE_RATE, mono=True)
        audio_inputs.append({"array": wav, "sampling_rate": SAMPLE_RATE})

    # Explicit batch loop so tqdm updates after every batch (not at the end)
    gen_kwargs = {"language": "english", "task": "transcribe"}
    results = []
    n_batches = (len(audio_inputs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(audio_inputs), BATCH_SIZE), total=n_batches, desc="Transcribing"):
        batch = audio_inputs[i : i + BATCH_SIZE]
        batch_out = asr(batch, generate_kwargs=gen_kwargs)
        if isinstance(batch_out, dict):
            results.append(batch_out)
        else:
            results.extend(batch_out)

    for aid, res in zip(missing, results):
        cache[aid] = res["text"].strip()

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[ASR] Saved {len(cache)} transcripts to {cache_path}")

    del asr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache
