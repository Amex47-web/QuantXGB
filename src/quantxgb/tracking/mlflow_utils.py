"""Local MLflow tracking (sqlite backend under ``artifacts/``)."""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


class Tracker:
    """Thin wrapper so the pipeline runs identically with tracking disabled."""

    def __init__(self, artifacts_dir: Path, enabled: bool = True, experiment: str = "quantxgb"):
        self.enabled = enabled
        self.artifacts_dir = artifacts_dir
        self.experiment = experiment
        self._run: Any = None

    @contextlib.contextmanager
    def run(self, run_name: str, tags: dict[str, str]) -> Iterator[Tracker]:
        if not self.enabled:
            yield self
            return
        import mlflow

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(self.artifacts_dir / 'mlflow.db').resolve()}")
        exp = mlflow.get_experiment_by_name(self.experiment)
        if exp is None:
            loc = (self.artifacts_dir / "mlflow-artifacts").resolve().as_uri()
            exp_id = mlflow.create_experiment(self.experiment, artifact_location=loc)
        else:
            exp_id = exp.experiment_id
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as r:
            self._run = r
            try:
                yield self
            finally:
                self._run = None

    def params(self, params: dict[str, Any]) -> None:
        if self.enabled and self._run is not None:
            import mlflow

            flat = _flatten(params)
            items = list(flat.items())
            for i in range(0, len(items), 100):
                mlflow.log_params(dict(items[i : i + 100]))

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            import mlflow

            clean = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, int | float) and not math.isnan(float(v))
            }
            if clean:
                mlflow.log_metrics(clean, step=step)

    def artifacts(self, path: Path) -> None:
        if self.enabled and self._run is not None:
            import mlflow

            if path.is_dir():
                mlflow.log_artifacts(str(path))
            elif path.exists():
                mlflow.log_artifact(str(path))
