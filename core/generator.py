from typing import Any


class TextGenerator:
    """Loads a Transformers causal language model on its first generation call."""

    def __init__(self, model_name: str, *, load_on_startup: bool = False):
        self.model_name = model_name
        self.tokenizer: Any = None
        self.model: Any = None
        if load_on_startup:
            self.load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        options: dict[str, Any] = {"device_map": "auto"}
        if torch.cuda.is_available():
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **options)

    def generate(self, query: str, documents: list[dict[str, Any]]) -> str:
        self.load()
        context = "\n\n".join(
            f"Doc {i}: {doc['content']}" for i, doc in enumerate(documents, 1)
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely based only on the context."
        if hasattr(self.tokenizer, "apply_chat_template") and "phi-2" not in self.model_name.casefold():
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            output = self.model.generate(encoded, max_new_tokens=400, do_sample=False)
            return self.tokenizer.decode(output[0][encoded.shape[-1]:], skip_special_tokens=True).strip()
        encoded = self.tokenizer(f"Instruct: {prompt}\nOutput:", return_tensors="pt").to(self.model.device)
        output = self.model.generate(**encoded, max_new_tokens=400, do_sample=False)
        return self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
