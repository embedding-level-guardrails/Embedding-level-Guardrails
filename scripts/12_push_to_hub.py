"""Шаг 12: загрузить собранную папку на Hugging Face Hub и проверить результат.

    python scripts/12_push_to_hub.py --repo-id USER/e5-small-v2-wildguardmix-joint
    python scripts/12_push_to_hub.py --repo-id USER/... --public

Папку сначала собирает scripts/11_export_hf.py.

Что делает:
1. Проверяет вход в Hugging Face (`hf auth login` или переменная HF_TOKEN). Если токена
   нет совсем — спросит его прямо в терминале; если токен есть, но не работает, —
   объяснит почему (истёк, лимит запросов, нет сети).
2. Создаёт репозиторий — по умолчанию ПРИВАТНЫЙ (--public, чтобы сразу открыть).
   Если репозиторий уже есть, его видимость не меняется.
3. Загружает папку одним коммитом. Повторный запуск загрузит изменения новым коммитом.
4. Скачивает модель обратно с Hub и сверяет с исходным чекпоинтом — так же, как
   11_export_hf.py сверял локальную папку.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.hub import compare_with_checkpoint  # noqa: E402
from eguard.utils import get_logger  # noqa: E402

logger = get_logger("push_to_hub")


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def ensure_login(api) -> str:
    """Имя пользователя Hub.

    Вход запрашивается, только если токена нет совсем. Любая другая ошибка —
    истёкший токен, лимит запросов, нет сети — показывается как есть: раньше скрипт
    выдавал её за «вы не вошли» и просил токен, хотя `hf auth login` уже был сделан.
    """
    from huggingface_hub import get_token, login

    if get_token() is None:
        print("Токен Hugging Face не найден (ни `hf auth login`, ни переменной HF_TOKEN).\n"
              "Вставьте токен с правом Write (https://huggingface.co/settings/tokens):")
        login()

    try:
        return api.whoami()["name"]
    except Exception as exc:
        status = _http_status(exc)
        if status == 401:
            raise SystemExit("Токен найден, но Hugging Face его не принимает (401): скорее всего, истёк "
                             "или отозван. Войдите заново: hf auth login") from exc
        if status == 429:
            raise SystemExit("Hugging Face временно ограничил число проверок входа (429). "
                             "Подождите несколько минут и запустите снова.") from exc
        raise SystemExit(f"Не удалось проверить вход в Hugging Face: {type(exc).__name__}: {exc}") from exc


def _explain_upload_error(exc: Exception, repo_id: str) -> SystemExit:
    status = _http_status(exc)
    if status in (401, 403):
        return SystemExit(f"Hugging Face отказал в записи в {repo_id} ({status}). Токену не хватает "
                          "прав записи или у вас нет доступа к этому пространству имён. Создайте токен "
                          "с правом Write и войдите с ним: hf auth login")
    return SystemExit(f"Загрузка не удалась: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="например USER/e5-small-v2-wildguardmix-joint")
    ap.add_argument("--folder", default="artifacts/hf_release/wgm-joint")
    ap.add_argument("--public", action="store_true", help="создать публичный репозиторий (по умолчанию приватный)")
    ap.add_argument("--message", default="Upload fine-tuned guardrail encoder")
    ap.add_argument("--config", default="configs/rq3_wildguardmix.yaml", help="для сверки с чекпоинтом")
    ap.add_argument("--run", default=None, help="прогон для сверки (по умолчанию имя папки)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    folder = Path(args.folder)
    if not (folder / "README.md").exists():
        raise SystemExit(f"{folder} не собрана — сначала scripts/11_export_hf.py")
    if args.repo_id.startswith("USER/"):
        raise SystemExit("Замените USER в --repo-id на свой логин на Hugging Face")

    api = HfApi()
    user = ensure_login(api)
    logger.info("Вход выполнен: %s", user)

    try:
        url = api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
        logger.info("Репозиторий: %s", url)
        api.upload_folder(repo_id=args.repo_id, folder_path=str(folder), commit_message=args.message)
    except Exception as exc:
        raise _explain_upload_error(exc, args.repo_id) from exc
    logger.info("Загружено: %s", folder)

    if not args.skip_verify:
        cfg = load_config(args.config)
        run = args.run or folder.name
        checkpoint = Path(cfg.training.get("output_dir", "artifacts/runs")) / run / "checkpoint"
        check = compare_with_checkpoint(args.repo_id, checkpoint, cfg.encoders[0])
        ok = check["max_abs_diff"] < 1e-4
        print(f"\nСверка модели с Hub и чекпоинта {checkpoint}: "
              f"макс. расхождение {check['max_abs_diff']:.2e} — {'OK' if ok else 'РАСХОЖДЕНИЕ'}")
        for text, p in zip(check["texts"], check.get("p_harm_release", [])):
            print(f"  P(harm) {p:.3f}  {text}")
        if not ok:
            raise SystemExit(1)

    visibility = "публичный" if args.public else "приватный"
    print(f"\nГотово ({visibility}): https://huggingface.co/{args.repo_id}")
    if not args.public:
        print("Открыть для всех: Settings → Change visibility на странице репозитория.")


if __name__ == "__main__":
    main()
