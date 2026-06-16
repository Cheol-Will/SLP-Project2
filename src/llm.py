"""Stage 3: RAG answer generation using Qwen2.5-7B-Instruct."""

import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 30


def load_prompt_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class QwenRAG:
    def __init__(self, prompt_cfg: dict, max_new_tokens: int = MAX_NEW_TOKENS):
        self._system = prompt_cfg["system_prompt"]
        self._few_shot = prompt_cfg.get("few_shot", [])
        self._user_template = prompt_cfg["user_template"]
        self._doc_format = prompt_cfg["doc_format"]
        self._max_new_tokens = max_new_tokens

        print(f"[LLM] Loading {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("[LLM] Model ready")

    def _build_messages(self, question: str, docs: list[tuple[str, str]]) -> list[dict]:
        doc_block = "\n\n".join(
            self._doc_format.format(doc_id=did, text=text) for did, text in docs
        )
        user_content = self._user_template.format(doc_block=doc_block, question=question)

        messages = [{"role": "system", "content": self._system}]
        messages.extend(self._few_shot)  # already in {"role": ..., "content": ...} format
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, question: str, docs: list[tuple[str, str]]) -> str:
        messages = self._build_messages(question, docs)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return _clean(raw)


def _clean(text: str) -> str:
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line.lower().startswith("answer:"):
            answer = line[len("answer:"):].strip()
            answer = re.sub(r"[.!?,;:]+$", "", answer)
            return answer.lower()
    text = text.strip().split("\n")[0].strip()
    text = re.sub(r"^(the answer is[:\s]+|answer[:\s]+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    return text.lower()