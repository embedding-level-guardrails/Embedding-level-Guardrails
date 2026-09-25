import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from transformers import BertConfig, BertModel

from ettin_guardrails.checkpoint import load_classifier
from ettin_guardrails.model import BaseClassifier, Classifier, LinearProbe
from ettin_guardrails.train import TrainingContext, _initialize_model, train_classifier_epoch


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.backbone_config = BertConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2,
        )
        for target, kwargs in [
            ("AutoModel.from_pretrained", {"side_effect": lambda *a, **kw: BertModel(self.backbone_config)}),
            ("AutoConfig.from_pretrained", {"return_value": self.backbone_config}),
        ]:
            patcher = patch(f"ettin_guardrails.model.{target}", **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
        self.cfg = OmegaConf.load(Path(__file__).parents[1] / "conf/default.yaml")
        self.cfg.models.classifier.head_hidden_dim = 4

    def test_common_interface_and_linear_probe_buffers(self):
        with self.assertRaises(TypeError):
            BaseClassifier()
        model = LinearProbe()
        self.assertIsInstance(model, BaseClassifier)
        self.assertNotIsInstance(model, Classifier)
        self.assertEqual(set(dict(model.named_parameters())), {"mlp.weight", "mlp.bias"})
        self.assertFalse(hasattr(model, "embedder"))
        self.assertEqual(list(model.backbone.parameters()), [])
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.mlp.training)
        model.to(dtype=torch.float64)
        logits, embedding = model(self.batch["input_ids"], self.batch["attention_mask"])
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(embedding.dtype, torch.float64)
        self.assertFalse(embedding.requires_grad)
        self.assertTrue(logits.requires_grad)

    def test_probe_mean_pooling_ignores_padding_and_handles_empty_mask(self):
        model = LinearProbe()
        hidden = torch.tensor([[[2.0] * 8, [4.0] * 8, [100.0] * 8], [[5.0] * 8] * 3])
        mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
        with patch.object(model.backbone, "forward", return_value=SimpleNamespace(last_hidden_state=hidden)):
            logits, embedding = model(self.batch["input_ids"], mask)
        torch.testing.assert_close(embedding, torch.tensor([[3.0] * 8, [0.0] * 8]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_shared_training_loop_updates_expected_weights(self):
        for model_type in ("linear-probe", "classifier"):
            with self.subTest(model_type=model_type):
                context = TrainingContext(
                    self.cfg, torch.device("cpu"), torch.float32, False,
                    [self.batch], [self.batch], None,
                )
                state = _initialize_model(model_type, context)
                backbone = state.model.backbone if model_type == "linear-probe" else state.model.embedder.encoder
                backbone_before = {
                    k: v.clone() for k, v in backbone.state_dict().items()
                }
                head_before = [p.detach().clone() for p in state.model.mlp.parameters()]
                metrics = train_classifier_epoch(context, state)
                self.assertTrue(torch.isfinite(torch.tensor([metrics.train_loss, metrics.val_loss])).all())
                self.assertEqual(state.global_step, 1)
                self.assertTrue(any(
                    not torch.equal(old, new)
                    for old, new in zip(head_before, state.model.mlp.parameters())
                ))
                backbone_changed = any(
                    not torch.equal(value, backbone.state_dict()[key])
                    for key, value in backbone_before.items()
                )
                self.assertEqual(backbone_changed, model_type == "classifier")

    def test_checkpoint_loader_restores_both_models(self):
        for model_type in ("linear-probe", "classifier", "joint-classifier", "ettin-ce"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                model = LinearProbe() if model_type == "linear-probe" else Classifier(head_hidden_dim=4)
                model.eval()
                expected = model(self.batch["input_ids"], self.batch["attention_mask"])
                checkpoint = {
                    "model_name": model_type,
                    "model": model.state_dict(),
                    "config": OmegaConf.to_container(self.cfg, resolve=False),
                }
                path = Path(directory) / "model.pt"
                torch.save(checkpoint, path)
                tokenizer = Mock()
                with patch("ettin_guardrails.checkpoint.AutoTokenizer.from_pretrained", return_value=tokenizer):
                    restored, loaded_tokenizer, _ = load_classifier(path)
                self.assertIs(type(restored), type(model))
                self.assertIs(loaded_tokenizer, tokenizer)
                restored.eval()
                actual = restored(self.batch["input_ids"], self.batch["attention_mask"])
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(before, after)
                if model_type == "linear-probe":
                    self.assertEqual(list(restored.backbone.parameters()), [])
