import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from ettin_guardrails.inference import predict


class FakeClassifier(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        values = (input_ids * attention_mask).sum(dim=1).float()
        return torch.stack((values, -values), dim=1), values


class InferenceTests(unittest.TestCase):
    config = {"tokenizer": {"max_length": 16}, "training": {"precision": "fp32"}}

    def test_batches_match_single_pass_and_preserve_order(self):
        def tokenize(prompts, **kwargs):
            rows = [list(map(int, prompt.split())) for prompt in prompts]
            width = max(map(len, rows))
            return {
                "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
            }

        tokenizer = Mock(side_effect=tokenize)
        prompts = ["1 2", "-2", "0 1 2", "3", "-1 0"]
        model = FakeClassifier()
        expected = predict(model, tokenizer, prompts, self.config, "cpu", batch_size=5)
        tokenizer.reset_mock()
        actual = predict(model, tokenizer, iter(prompts), self.config, "cpu", batch_size=2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([len(call.args[0]) for call in tokenizer.call_args_list], [2, 2, 1])
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(model.training)
        self.assertEqual(predict(model, tokenizer, "1", self.config, "cpu").shape, (1, 2))

    def test_cuda_oom_retries_without_losing_or_duplicating_rows(self):
        attempts = []

        def run_batch(classifier, tokenizer, prompts, config, device, dtype):
            attempts.append(list(prompts))
            if len(prompts) > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return torch.tensor([[float(prompt)] for prompt in prompts])

        with patch("ettin_guardrails.inference._predict_batch", side_effect=run_batch), \
                patch("torch.cuda.device", return_value=nullcontext()), \
                patch("torch.cuda.empty_cache") as empty_cache:
            result = predict(Mock(), Mock(), list("0123456"), self.config, "cuda:1", batch_size=6)
        self.assertEqual(result.flatten().tolist(), list(range(7)))
        self.assertEqual([len(batch) for batch in attempts], [6, 3, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(empty_cache.call_count, 2)

    def test_single_item_oom_and_other_errors_propagate(self):
        for error in [torch.cuda.OutOfMemoryError("simulated OOM"), RuntimeError("model error")]:
            with self.subTest(error=type(error)), \
                    patch("ettin_guardrails.inference._predict_batch", side_effect=error), \
                    patch("torch.cuda.empty_cache") as empty_cache:
                with self.assertRaises(type(error)):
                    predict(Mock(), Mock(), "text", self.config, "cuda", batch_size=1)
                empty_cache.assert_not_called()

    def test_invalid_inputs(self):
        for batch_size in [0, -1, True, 1.5]:
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                predict(Mock(), Mock(), "text", self.config, "cpu", batch_size=batch_size)
        with self.assertRaises(ValueError):
            predict(Mock(), Mock(), [], self.config, "cpu")
