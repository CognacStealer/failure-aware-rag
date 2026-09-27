import unittest

from core.generator import TextGenerator


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
        generator = TextGenerator("org/chat-model")
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
        generator = TextGenerator("microsoft/phi-2")
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


if __name__ == "__main__":
    unittest.main()
