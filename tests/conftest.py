from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from quantxgb.config import AppConfig, load_config
from quantxgb.synthetic import (
    SyntheticData,
    SyntheticSpec,
    make_synthetic,
    synthetic_config,
)

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("QUANTXGB_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network tests disabled (set QUANTXGB_NETWORK_TESTS=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_config() -> AppConfig:
    return load_config(root=ROOT)


@pytest.fixture(scope="session")
def small_spec() -> SyntheticSpec:
    return SyntheticSpec(n_assets=8, start=dt.date(2004, 1, 1), end=dt.date(2008, 12, 31), seed=3)


@pytest.fixture(scope="session")
def small_cfg(
    base_config: AppConfig, small_spec: SyntheticSpec, tmp_path_factory: pytest.TempPathFactory
) -> AppConfig:
    cfg = synthetic_config(base_config, small_spec)
    tmp = tmp_path_factory.mktemp("qx")
    paths = cfg.data.paths.model_copy(
        update={
            "raw": tmp / "raw",
            "curated": tmp / "curated",
            "features": tmp / "features",
            "artifacts": tmp / "artifacts",
        }
    )
    return cfg.model_copy(update={"data": cfg.data.model_copy(update={"paths": paths})})


@pytest.fixture(scope="session")
def small_data(small_cfg: AppConfig, small_spec: SyntheticSpec) -> SyntheticData:
    return make_synthetic(small_cfg, small_spec)
