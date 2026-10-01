import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from core.chunking import best_passages


def build_prompt(query: str, documents: list[dict[str, Any]], max_context_chars: int) -> str:
    """Give each document an equal share of the budget, filled with its most relevant passage."""
    per_doc = max(200, max_context_chars // max(1, len(documents)))
    blocks = []
    for i, doc in enumerate(documents, 1):
        passage = doc.get("passage") or next(
            iter(best_passages(query, doc.get("content", ""), limit=1, size=per_doc)), ""
        )
        blocks.append(f"Doc {i}: {passage[:per_doc]}")
    context = "\n\n".join(blocks)
    return (
        f"Context:\n{context}\n\nQuestion: {query}\n"
        "Answer concisely based only on the context. "
        "If the context does not contain the answer, say you don't know."
    )


class TextGenerator:
    """Answer generation through Ollama (CPU-friendly) or Transformers (CUDA)."""

    def __init__(
        self,
        model_name: str,
        *,
        backend: str = "ollama",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 600,
        max_context_chars: int = 8000,
        max_new_tokens: int = 256,
        load_on_startup: bool = False,
    ):
        if backend not in {"ollama", "transformers"}:
            raise ValueError("GENERATOR_BACKEND must be 'ollama' or 'transformers'")
        self.model_name = model_name
        self.backend = backend
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = None
        self.model: Any = None
        if load_on_startup:
            self.load()

    @property
    def loaded(self) -> bool:
        if self.backend == "ollama":
            return self._ollama_has_model()
        return self.model is not None

    def _ollama_has_model(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as response:
                models = json.load(response).get("models", [])
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return any(model.get("name") == self.model_name for model in models)

    def load(self) -> None:
        if self.backend == "ollama":
            if not self._ollama_has_model():
                raise RuntimeError(
                    f"Ollama at {self.host} is not serving '{self.model_name}'; "
                    f"run `ollama pull {self.model_name}`"
                )
            return
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "the transformers backend needs a CUDA GPU; a 7B model in fp32 does not fit "
                "in CPU memory. Use GENERATOR_BACKEND=ollama instead."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        )

    def generate(self, query: str, documents: list[dict[str, Any]]) -> str:
        prompt = build_prompt(query, documents, self.max_context_chars)
        if self.backend == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_transformers(prompt)

    def stream(self, query: str, documents: list[dict[str, Any]]) -> Iterator[str]:
        """Yield the answer in pieces as the model produces them (whole answer for transformers)."""
        prompt = build_prompt(query, documents, self.max_context_chars)
        if self.backend != "ollama":
            yield self._generate_transformers(prompt)
            return
        body = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens, "num_ctx": 4096},
        }).encode()
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for line in response:  # Ollama streams one JSON object per line
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    def _generate_ollama(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens, "num_ctx": 4096},
        }).encode()
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["message"]["content"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        self.load()
        if hasattr(self.tokenizer, "apply_chat_template") and "phi-2" not in self.model_name.casefold():
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            output = self.model.generate(encoded, max_new_tokens=self.max_new_tokens, do_sample=False)
            return self.tokenizer.decode(output[0][encoded.shape[-1]:], skip_special_tokens=True).strip()
        encoded = self.tokenizer(f"Instruct: {prompt}\nOutput:", return_tensors="pt").to(self.model.device)
        output = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
