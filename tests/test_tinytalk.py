"""Unit and integration tests for TinyTalk Transformer, tokenizer, and API."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

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
        # Subclass to force predicting a known token and verify EOS early stopping on multi-row batches
        class EarlyStopModel(TinyTransformer):
            def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
                batch, seq_len = token_ids.shape
                logits = torch.zeros((batch, seq_len, self.config["vocab_size"]), device=token_ids.device)
                logits[:, :, 5] = 100.0  # Force model to predict token ID 5
                return logits, None

        model = EarlyStopModel(vocab_size=32, d_model=32, n_heads=4, n_layers=2, block_size=48)
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # With eos_id=5, generation must stop immediately on step 1
        stopped = model.generate(prompt, max_new_tokens=20, eos_id=5)
        self.assertEqual(stopped.shape, (2, 4))
        self.assertTrue((stopped[:, -1] == 5).all())

        # Without eos_id, generation completes all requested tokens
        full = model.generate(prompt, max_new_tokens=5)
        self.assertEqual(full.shape, (2, 8))


class TestAPIEndpoints(unittest.TestCase):
    def test_health_check(self):
        from tinytalk.api import health
        result = health()
        self.assertEqual(result, {"status": "ready"})

    def test_chat_schema_validation(self):
        from pydantic import ValidationError
        from tinytalk.api import ChatRequest, ChatMessage

        # The browser can omit this field and still reach the trained model on a novel query.
        self.assertEqual(ChatRequest(message="Hello").max_new_tokens, 64)

        # Invalid role should fail Pydantic validation
        with self.assertRaises(ValidationError):
            ChatRequest(message="Hello", history=[ChatMessage(role="hacker", content="test")])

    def test_chat_end_to_end_generation(self):
        from tinytalk.api import chat, ChatRequest, ChatMessage

        response = chat(ChatRequest(
            message="What is attention?",
            history=[ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!")],
            max_new_tokens=10
        ))
        self.assertIsInstance(response.reply, str)
        self.assertGreater(len(response.reply), 0)

    def test_chat_uses_transformer_by_default_after_retrieval_miss(self):
        from tinytalk.api import chat, ChatRequest

        class EmptyRAG:
            def retrieve(self, *args, **kwargs):
                return []

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = {"block_size": 128}
                self.requested_limit = None

            def generate(self, token_ids, **kwargs):
                self.requested_limit = kwargs["max_new_tokens"]
                addition = torch.tensor([[2]], device=token_ids.device)
                return torch.cat((token_ids, addition), dim=1)

        class FakeTokenizer:
            def encode(self, text):
                return [1]

            def decode(self, ids):
                return "Generated fallback answer."

        model = FakeModel()
        with patch("tinytalk.api.rag_instance", return_value=EmptyRAG()), \
             patch("tinytalk.api.find_developer_response", return_value=None), \
             patch("tinytalk.api.fetch_online_knowledge", return_value=(None, None, None)), \
             patch("tinytalk.api.model_bundle", return_value=(model, FakeTokenizer())):
            response = chat(ChatRequest(message="What is quuxification?"))

        self.assertEqual(response.source, "generator")
        self.assertEqual(model.requested_limit, 64)

    def test_chat_long_message_handling(self):
        from tinytalk.api import chat, ChatRequest, format_prompt

        long_message = "What is attention in transformer models? " * 10

        # 1. Directly assert prompt formatting preserves role markers and enforces bounds
        formatted = format_prompt(long_message, [], block_size=128)
        self.assertTrue(formatted.startswith("User: "))
        self.assertTrue(formatted.endswith("\nAssistant:"))
        self.assertLessEqual(len(formatted), 128)

        # 2. Assert the chat route processes the long message without crashing
        response = chat(ChatRequest(message=long_message, history=[], max_new_tokens=10))
        self.assertIsInstance(response.reply, str)
        self.assertGreater(len(response.reply), 0)


class TestTinyRAG(unittest.TestCase):
    def setUp(self):
        from tinytalk.rag import TinyRAG
        self.rag = TinyRAG()

    def test_rag_exact_match(self):
        matches = self.rag.retrieve("How do you work?", top_k=1)
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].matched_prompt, "How do you work?")
        self.assertIn("embeddings", matches[0].answer)

    def test_rag_low_score_for_unrelated_query(self):
        matches = self.rag.retrieve("What is the recipe for chocolate cake?", top_k=1, threshold=2.2)
        self.assertEqual(len(matches), 0)

    def test_rag_rejects_subject_mismatch(self):
        # Specific query for Django commands must never match unrelated "start the frontend"
        matches = self.rag.retrieve("can you tell me the start command for django generally in terminal?", top_k=1)
        for m in matches:
            self.assertNotIn("frontend", m.matched_prompt.lower())
            self.assertNotIn("vite", m.answer.lower())


class TestDevAssistantAndSearch(unittest.TestCase):
    def test_dev_assistant_django(self):
        from tinytalk.dev_assistant import find_developer_response
        res = find_developer_response("can you tell me the start command for django generally in terminal?")
        self.assertIsNotNone(res)
        self.assertIn("Django", res["title"])
        self.assertIn("python manage.py runserver", res["content"])
        self.assertIn("docs.djangoproject.com", res["doc_url"])

    def test_online_search_clean_query(self):
        from tinytalk.online_search import clean_topic_query
        cleaned = clean_topic_query("can you tell me about the James Webb Space Telescope please?")
        self.assertEqual(cleaned, "James Webb Space Telescope")

    def test_online_search_multi_clause_cleaning(self):
        from tinytalk.online_search import clean_topic_query
        cleaned = clean_topic_query("how to make biryani? give me steps proeprly?")
        self.assertEqual(cleaned, "biryani")

    def test_tech_entity_resolution_for_java(self):
        from tinytalk.online_search import resolve_tech_entity
        self.assertEqual(resolve_tech_entity("java"), "Java (programming language)")
        self.assertEqual(resolve_tech_entity("java prommaing language"), "Java (programming language)")

    def test_reference_url_allowlist(self):
        from tinytalk.online_search import is_allowed_reference_url
        self.assertTrue(is_allowed_reference_url("https://en.wikipedia.org/wiki/Transformer"))
        self.assertFalse(is_allowed_reference_url("http://en.wikipedia.org/wiki/Transformer"))
        self.assertFalse(is_allowed_reference_url("https://wikipedia.org.example.com/fake"))

    def test_followup_entity_uses_user_history_not_assistant_text(self):
        from tinytalk.online_search import detect_intent_and_entity
        history = [
            {"role": "user", "content": "Who is Virat Kohli?"},
            {"role": "assistant", "content": "I can also discuss Cristiano Ronaldo."},
        ]
        entity, intent, _ = detect_intent_and_entity("Did he retire from T20?", history)
        self.assertEqual(entity, "Virat Kohli")
        self.assertEqual(intent, "retirement_status")

    def test_dynamic_internet_retrieval_biryani(self):
        from tinytalk.online_search import fetch_online_knowledge
        with patch("tinytalk.online_search.search_web_ddg", return_value=(None, None, None)), \
             patch("tinytalk.online_search.search_wikipedia", return_value=("Biryani", "A rice dish with spices.", "https://en.wikipedia.org/wiki/Biryani")):
            title, extract, url = fetch_online_knowledge("how to make biryani? give me steps proeprly?")
        self.assertIsNotNone(extract)
        self.assertTrue(len(extract) > 20)
        self.assertIsNotNone(url)

    def test_dynamic_internet_retrieval_animal(self):
        from tinytalk.online_search import fetch_online_knowledge
        with patch("tinytalk.online_search.search_wikipedia", return_value=("Red kangaroo", "The red kangaroo is a large kangaroo native to Australia.", "https://en.wikipedia.org/wiki/Red_kangaroo")):
            title, extract, url = fetch_online_knowledge("what is a red kangaroo?")
        self.assertIsNotNone(extract)
        self.assertIn("kangaroo", extract.lower())
        self.assertIn("wikipedia.org", url)

    def test_dynamic_internet_retrieval_stackoverflow(self):
        from tinytalk.online_search import fetch_online_knowledge
        with patch("tinytalk.online_search.search_stack_overflow", return_value=("ZeroDivisionError", "Catch it with try/except.", "https://stackoverflow.com/questions/1/example")):
            title, extract, url = fetch_online_knowledge("how to catch ZeroDivisionError in python?")
        self.assertIsNotNone(extract)
        self.assertIn("stackoverflow.com", url)


if __name__ == "__main__":
    unittest.main()
