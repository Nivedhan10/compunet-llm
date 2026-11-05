"""Unit and integration tests for TinyTalk Transformer, tokenizer, and API."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tinytalk.tokenizer import CharTokenizer
from tinytalk.model import TinyTransformer, CausalSelfAttention, TransformerBlock


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.sample_text = "Attention is all you need."
        self.tokenizer = CharTokenizer.from_text(self.sample_text)

    def test_encode_decode_roundtrip(self):
        encoded = self.tokenizer.encode(self.sample_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, self.sample_text)

    def test_unknown_character_fallback(self):
        # Character not in vocabulary should map to <unk> id (0) and be skipped on decode
        encoded = self.tokenizer.encode("Z")
        self.assertEqual(encoded, [self.tokenizer.stoi["<unk>"]])
        self.assertEqual(self.tokenizer.decode(encoded), "")

    def test_vocab_serialization(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.tokenizer.save(tmp_path)
            loaded = CharTokenizer.load(tmp_path)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.itos, self.tokenizer.itos)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 32
        self.d_model = 32
        self.n_heads = 4
        self.n_layers = 2
        self.block_size = 48
        self.model = TinyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            block_size=self.block_size,
            dropout=0.0
        )
        self.model.eval()

    def test_forward_output_shapes(self):
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        logits, loss = self.model(tokens, targets)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_weight_tying(self):
        # Weight tying: language head weights must be the exact same object as token embedding weights
        self.assertTrue(torch.equal(self.model.language_head.weight, self.model.token_embedding.weight))
        self.assertIs(self.model.language_head.weight, self.model.token_embedding.weight)

    def test_causality_property(self):
        # Future tokens must not influence predictions at past positions.
        tokens_a = torch.tensor([[1, 2, 3, 4]])
        tokens_b = torch.tensor([[1, 2, 3, 9]])  # only position 3 differs

        logits_a, _ = self.model(tokens_a)
        logits_b, _ = self.model(tokens_b)

        # Logits at positions 0, 1, 2 should be identical within numerical precision
        self.assertTrue(torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-5))

    def test_generation_shape(self):
        prompt = torch.tensor([[1, 2, 3]])
        generated = self.model.generate(prompt, max_new_tokens=10, temperature=0.7, top_k=4)
        self.assertEqual(generated.shape, (1, 13))

    def test_generation_with_eos(self):
        # Multi-row batch prompt exercises batch-safe EOS handling without .item() scalar errors
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        generated = self.model.generate(prompt, max_new_tokens=15, eos_id=0)
        self.assertIsInstance(generated, torch.Tensor)
        self.assertEqual(generated.shape[0], 2)
        self.assertGreaterEqual(generated.shape[1], 3)


class TestAPIEndpoints(unittest.TestCase):
    def test_health_check(self):
        from fastapi.testclient import TestClient
        from tinytalk.api import app

        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ready")

    def test_chat_schema_validation(self):
        from fastapi.testclient import TestClient
        from tinytalk.api import app

        client = TestClient(app)
        # Invalid role should be rejected with 422 Unprocessable Entity
        response = client.post("/chat", json={
            "message": "Hello",
            "history": [{"role": "hacker", "content": "test"}]
        })
        self.assertEqual(response.status_code, 422)

    def test_chat_end_to_end_generation(self):
        from fastapi.testclient import TestClient
        from tinytalk.api import app

        client = TestClient(app)
        response = client.post("/chat", json={
            "message": "What is attention?",
            "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("reply", data)
        self.assertIsInstance(data["reply"], str)
        self.assertGreater(len(data["reply"]), 0)

    def test_chat_long_message_handling(self):
        from fastapi.testclient import TestClient
        from tinytalk.api import app

        client = TestClient(app)
        # Message longer than block_size (128) must not crash the endpoint
        long_message = "What is attention in transformer models? " * 10
        response = client.post("/chat", json={
            "message": long_message,
            "history": []
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("reply", response.json())


if __name__ == "__main__":
    unittest.main()
