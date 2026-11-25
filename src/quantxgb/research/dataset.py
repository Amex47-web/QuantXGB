"""Assemble the modelling dataset (features + labels) and persist feature builds."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quantxgb.config import AppConfig
from quantxgb.features.build import FeatureBuild, build_features
from quantxgb.features.registry import render_dictionary
from quantxgb.labels.forward_returns import attach_labels, daily_labels
from quantxgb.utils.io import frame_hash, read_json, write_json, write_parquet_atomic


@dataclass
class Dataset:
    frame: (
        pl.DataFrame
    )  # date, ticker, asset_class, features..., fwd_ret, fwd_xs_ret, fwd_grade, label_*
    market_daily: pl.DataFrame
    feature_names: list[str]
    sessions: list[dt.date]

    @property
    def dates(self) -> list[dt.date]:
        return sorted(self.frame["date"].unique().to_list())


def assemble(
    prices: pl.DataFrame,
    macro: pl.DataFrame | None,
    cfg: AppConfig,
    sessions: list[dt.date],
) -> tuple[Dataset, FeatureBuild]:
    fb = build_features(prices, macro, cfg, sessions)
    labels = daily_labels(prices, cfg.labels.horizon, cfg.labels.execution_delay_sessions)
    frame = attach_labels(fb.matrix, labels, cfg.labels.n_grades)
    return Dataset(frame, fb.market_daily, fb.feature_names, sessions), fb


def feature_dir(cfg: AppConfig) -> Path:
    key = cfg.section_hash("data", "features", "labels")
    return cfg.features_dir / key


def save_dataset(ds: Dataset, fb: FeatureBuild, cfg: AppConfig, data_hash: str) -> Path:
    out = feature_dir(cfg)
    write_parquet_atomic(ds.frame, out / "dataset.parquet")
    write_parquet_atomic(ds.market_daily, out / "market_daily.parquet")
    write_json(
        {
            "feature_names": ds.feature_names,
            "data_hash": data_hash,
            "dataset_hash": frame_hash(ds.frame),
            "config_hash": cfg.section_hash("data", "features", "labels"),
            "sessions": [d.isoformat() for d in ds.sessions],
        },
        out / "meta.json",
    )
    (out / "feature_dictionary.md").write_text(render_dictionary(fb.specs), encoding="utf-8")
    return out


def load_dataset(cfg: AppConfig) -> tuple[Dataset, dict[str, object]]:
    d = feature_dir(cfg)
    if not (d / "dataset.parquet").exists():
        raise FileNotFoundError(
            f"no feature build for the current config at {d} — run `quantxgb features build`"
        )
    meta = read_json(d / "meta.json")
    ds = Dataset(
        frame=pl.read_parquet(d / "dataset.parquet"),
        market_daily=pl.read_parquet(d / "market_daily.parquet"),
        feature_names=list(meta["feature_names"]),
        sessions=[dt.date.fromisoformat(s) for s in meta["sessions"]],
    )
    return ds, meta
