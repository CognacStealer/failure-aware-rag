import io
import json
import unittest
from unittest.mock import patch

from core.generator import TextGenerator, build_prompt


class FakeTokens:
    shape = (1, 2)

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, key):
        return self


class FakeGeneratedSequence:
    def __getitem__(self, key):
        return self


class FakeGeneratedOutput:
    def __getitem__(self, index):
        return FakeGeneratedSequence()


class FakeBatchEncoding(dict):
    def to(self, device):
        for value in self.values():
            value.to(device)
        return self


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeTokens()

    def decode(self, tokens, skip_special_tokens):
        return "mocked response"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeGeneratedOutput()


class GeneratorTests(unittest.TestCase):
    def test_load_is_deferred_until_generation_and_chat_prompt_is_used(self):
        generator = TextGenerator("org/chat-model", backend="transformers")
        self.assertFalse(generator.loaded)
        tokenizer = FakeTokenizer()
        model = FakeModel()

        def fake_load():
            generator.tokenizer = tokenizer
            generator.model = model

        generator.load = fake_load
        docs = [{"content": "the evidence", "title": "Evidence"}]
        answer = generator.generate("question?", docs)

        self.assertEqual(answer, "mocked response")
        self.assertTrue(generator.loaded)
        self.assertIn("Question: question?", tokenizer.messages[0]["content"])
        self.assertIn("the evidence", tokenizer.messages[0]["content"])
        self.assertEqual(len(model.calls), 1)

    def test_phi2_uses_instruct_output_prompt(self):
        generator = TextGenerator("microsoft/phi-2", backend="transformers")
        captured = {}

        class LegacyTokenizer(FakeTokenizer):
            apply_chat_template = None

            def __call__(self, prompt, return_tensors):
                captured["prompt"] = prompt
                captured["return_tensors"] = return_tensors
                return FakeBatchEncoding(input_ids=FakeTokens())

        tokenizer = LegacyTokenizer()
        generator.tokenizer = tokenizer
        generator.model = FakeModel()
        answer = generator.generate("question?", [{"content": "context"}])

        self.assertEqual(answer, "mocked response")
        self.assertTrue(captured["prompt"].startswith("Instruct: Context:"))
        self.assertTrue(captured["prompt"].endswith("Output:"))


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OllamaGeneratorTests(unittest.TestCase):
    def test_ollama_backend_posts_chat_request(self):
        generator = TextGenerator("qwen2.5:7b-instruct-q4_K_M", host="http://ollama:1")
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            return FakeHTTPResponse(json.dumps({"message": {"content": " the answer "}}).encode())

        with patch("core.generator.urllib.request.urlopen", fake_urlopen):
            answer = generator.generate("question?", [{"content": "the evidence"}])

        self.assertEqual(answer, "the answer")
        self.assertEqual(captured["url"], "http://ollama:1/api/chat")
        self.assertEqual(captured["body"]["model"], "qwen2.5:7b-instruct-q4_K_M")
        self.assertIn("the evidence", captured["body"]["messages"][0]["content"])

    def test_loaded_reflects_whether_ollama_serves_the_model(self):
        generator = TextGenerator("qwen2.5:7b-instruct-q4_K_M")
        tags = json.dumps({"models": [{"name": "qwen2.5:7b-instruct-q4_K_M"}]}).encode()
        with patch("core.generator.urllib.request.urlopen", return_value=FakeHTTPResponse(tags)):
            self.assertTrue(generator.loaded)
        with patch("core.generator.urllib.request.urlopen", side_effect=OSError("down")):
            self.assertFalse(generator.loaded)
            with self.assertRaisesRegex(RuntimeError, "not serving"):
                generator.load()

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            TextGenerator("model", backend="bad")


class PromptTests(unittest.TestCase):
    def test_prompt_stays_within_budget_and_prefers_relevant_passages(self):
        long_doc = "filler words here " * 2000 + "the rollback command is deploy undo"
        prompt = build_prompt("what is the rollback command", [{"content": long_doc}] * 4, 2000)
        self.assertLess(len(prompt), 2600)
        self.assertIn("rollback command is deploy undo", prompt)

    def test_prompt_uses_precomputed_passage(self):
        prompt = build_prompt("q", [{"content": "full text", "passage": "graded passage"}], 1000)
        self.assertIn("graded passage", prompt)
        self.assertNotIn("full text", prompt)


if __name__ == "__main__":
    unittest.main()
