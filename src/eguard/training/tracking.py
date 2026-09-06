"""Тонкая обёртка над MLflow.

Смысл обёртки в двух вещах. Во-первых, mlflow не должен быть жёсткой зависимостью
пайплайна: без него шаги всё равно должны отрабатывать, поэтому при `enabled: false`
или отсутствии пакета возвращается no-op с тем же интерфейсом. Во-вторых, для RQ3
важно, чтобы в run попадали не только метрики, но и полный конфиг с составом пар:
сравнение objective против classification head имеет смысл только тогда, когда
видно, что данные и гиперпараметры совпадали.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger(__name__)


def flatten_params(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Вложенный конфиг -> плоские ключи вида `training.loss`. MLflow не хранит дерево."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_params(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ", ".join(map(str, obj)) if obj else "[]"
    else:
        out[prefix] = obj
    return out


class NullRun:
    """Заглушка с интерфейсом Tracker: пайплайн работает без mlflow."""

    run_id = None
    enabled = False

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...
    def finish(self, status: str = "FINISHED") -> None: ...


class MLflowRun:
    def __init__(self, run, client_module):
        self._run = run
        self._mlflow = client_module
        self.run_id = run.info.run_id
        self.enabled = True

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: ("" if v is None else v) for k, v in flatten_params(params).items()}
        # MLflow режет значения параметров; длинные значения обрезаем сами, явно.
        flat = {k: (str(v)[:490] + "…" if len(str(v)) > 500 else v) for k, v in flat.items()}
        for i in range(0, len(flat), 100):                      # батчами: лимит на запрос
            chunk = dict(list(flat.items())[i : i + 100])
            self._mlflow.log_params(chunk)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and _is_number(v)}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        path = Path(path)
        if not path.exists():
            return
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def finish(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def resume_run(cfg_mlflow: dict[str, Any], run_id: str | None):
    """Дозапись в существующий прогон.

    Нужна, чтобы метрики LODO-оценки лежали в том же run, где училась модель:
    иначе сравнение objective приходится собирать вручную из двух экспериментов.
    """
    if not run_id or not cfg_mlflow.get("enabled", True):
        return NullRun()
    try:
        import mlflow
    except ImportError:
        return NullRun()

    mlflow.set_tracking_uri(cfg_mlflow.get("tracking_uri") or "sqlite:///mlflow.db")
    try:
        run = mlflow.start_run(run_id=run_id)
    except Exception as exc:                      # прогон удалён или база другая
        logger.warning("Не удалось продолжить run %s: %s", run_id, exc)
        return NullRun()
    logger.info("MLflow: дозапись в run %s", run_id)
    return MLflowRun(run, mlflow)


def start_run(cfg_mlflow: dict[str, Any], run_name: str | None = None,
              tags: dict[str, Any] | None = None):
    """Возвращает MLflowRun или NullRun — вызывающий код различать их не обязан."""
    if not cfg_mlflow.get("enabled", True):
        logger.info("MLflow выключен в конфиге")
        return NullRun()

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow не установлен — обучение пойдёт без трекинга")
        return NullRun()

    # sqlite, а не file:./mlruns: с MLflow 3.x файловый бэкенд отдаёт исключение,
    # если не выставить MLFLOW_ALLOW_FILE_STORE=true.
    uri = cfg_mlflow.get("tracking_uri") or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg_mlflow.get("experiment", "eguard"))
    run = mlflow.start_run(run_name=run_name)

    wrapper = MLflowRun(run, mlflow)
    if tags:
        wrapper.set_tags(tags)
    logger.info("MLflow run %s (%s, experiment=%s)", run.info.run_id, uri,
                cfg_mlflow.get("experiment", "eguard"))
    return wrapper
