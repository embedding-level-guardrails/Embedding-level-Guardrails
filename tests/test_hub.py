"""Тесты выгрузки на Hub (src/eguard/hub.py): без сети и без настоящей модели."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from eguard.hub import (  # noqa: E402
    BASE_FILES,
    CHECKPOINT_FILES,
    HEAD_FILE,
    assemble_release,
    harm_probability,
    head_to_safetensors,
    load_release_head,
    render_model_card,
)


def _fake_checkpoint(path: Path, with_head: bool = True) -> Path:
    path.mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        (path / name).write_text("stub")
    if with_head:
        head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(384, 2))
        torch.save(head.state_dict(), path / "head.pt")
    return path


def _fake_fetch(tmp: Path):
    def fetch(repo_id: str, filename: str) -> str:
        target = tmp / "base" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{repo_id}:{filename}")
        return str(target)
    return fetch


def test_head_conversion_keeps_weights(tmp_path):
    """Голова в safetensors — обычный Linear с теми же весами и тем же выходом."""
    seq = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(384, 2)).eval()
    torch.save(seq.state_dict(), tmp_path / "head.pt")
    head_to_safetensors(tmp_path / "head.pt", tmp_path / HEAD_FILE)

    linear = load_release_head(tmp_path)
    x = torch.randn(5, 384)
    assert torch.allclose(linear(x), seq(x))
    assert torch.allclose(harm_probability(linear, x), torch.softmax(seq(x), -1)[:, 1])


def test_assemble_release_layout(tmp_path):
    ckpt = _fake_checkpoint(tmp_path / "ckpt")
    files = assemble_release(ckpt, tmp_path / "out", fetch=_fake_fetch(tmp_path))
    assert set(files) == set(CHECKPOINT_FILES) | set(BASE_FILES) | {HEAD_FILE}
    assert "head.pt" not in files                         # pickle не выгружается


def test_assemble_release_without_head_and_rebuild(tmp_path):
    """Без головы (contrastive) head.safetensors нет; повторная сборка не оставляет старых файлов."""
    out = tmp_path / "out"
    assemble_release(_fake_checkpoint(tmp_path / "with"), out, fetch=_fake_fetch(tmp_path))
    files = assemble_release(_fake_checkpoint(tmp_path / "without", with_head=False), out,
                             fetch=_fake_fetch(tmp_path))
    assert HEAD_FILE not in files
    assert load_release_head(out) is None


def test_model_card_sections():
    common = dict(repo_id="me/model", base_model="intfloat/e5-small-v2", n_pairs=32538,
                  epochs=2, batch_size=32, lr=2e-5, temperature=0.05)
    rows = [{"name": "WildGuardTest", "n": 1699, "auroc": 0.925, "tpr": 0.444, "fpr": 0.026, "fnr": 0.403}]

    joint = render_model_card(objective="joint", threshold=0.7765, eval_rows=rows, **common)
    assert "me/model" in joint and "0.7765" in joint and "head.safetensors" in joint
    assert "| WildGuardTest | 1,699 | 0.925 |" in joint
    assert "$" not in joint                               # все плейсхолдеры подставлены

    contrastive = render_model_card(objective="contrastive", **common)
    assert "head.safetensors" not in contrastive and "## Evaluation" not in contrastive
