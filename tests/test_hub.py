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
    rows = [{"name": "WildGuardTest", "n": 1699, "auroc": 0.925, "tpr": 0.444, "fpr": 0.026, "fnr": 0.403}]
    card = render_model_card(repo_id="me/model", base_model="intfloat/e5-small-v2", training="TRAINING",
                             threshold=0.7765, eval_rows=rows)
    assert "me/model" in card and "0.7765" in card and "head.safetensors" in card and "TRAINING" in card
    assert "| WildGuardTest | 1,699 | 0.925 |" in card
    assert "$" not in card                                # все плейсхолдеры подставлены
    assert "HarmBench" not in card

    no_head = render_model_card(repo_id="me/model", base_model="b", training="t", harmbench=True)
    assert "head.safetensors" not in no_head and "## Evaluation" not in no_head
    assert "HarmBench" in no_head


def test_training_description_matches_run():
    """Карточка не должна приписывать модели чужие данные и лосс."""
    from eguard.hub import describe_training

    common = dict(base_model="e5", epochs=2, batch_size=32, lr=2e-5, temperature=0.05,
                  supcon_temperature=0.07)
    supcon = describe_training(objective="supcon_joint", n_texts=43025, n_pairs=None,
                               harmbench=False, synthetic=False, **common)
    assert "Supervised Contrastive (SupCon) + cross-entropy" in supcon and "43,025 labeled prompts" in supcon
    assert "triplets" not in supcon and "HarmBench" not in supcon and "no synthetic" in supcon
    assert "SupCon temperature 0.07" in supcon

    infonce = describe_training(objective="joint", n_texts=48985, n_pairs=32538,
                                harmbench=True, synthetic=True, **common)
    assert "32,538 triplets" in infonce and "HarmBench" in infonce and "InfoNCE temperature 0.05" in infonce
