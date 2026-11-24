"""Typed configuration loaded from ``configs/*.yaml`` (PRD §13).

Every YAML file carries ``status: proposed | frozen``. Research decisions are owned by
the project owner; code never silently changes them. ``AppConfig.all_frozen`` is checked
before the one-shot holdout evaluation is allowed to run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Status = Literal["proposed", "frozen"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- data.yaml
class AssetSpec(_Strict):
    ticker: str
    asset_class: str
    description: str = ""
    approx_inception: dt.date | None = None


class FredConfig(_Strict):
    base_url: str = "https://api.stlouisfed.org/fred"
    observation_start: dt.date = dt.date(1990, 1, 1)
    max_requests_per_minute: int = Field(100, gt=0, le=120)
    page_limit: int = Field(100_000, gt=0, le=100_000)


class MacroSeriesSpec(_Strict):
    id: str
    revised: bool
    description: str = ""


class AvailabilityConfig(_Strict):
    price_lag_sessions: int = Field(0, ge=0)
    daily_macro_lag_sessions: int = Field(1, ge=0)
    revised_macro_lag_sessions: int = Field(1, ge=0)


class PathsConfig(_Strict):
    raw: Path = Path("data/raw")
    curated: Path = Path("data/curated")
    features: Path = Path("data/features")
    artifacts: Path = Path("artifacts")


class DataConfig(_Strict):
    status: Status
    calendar: str = "XNYS"
    price_start: dt.date
    price_end: dt.date | None = None
    market_proxy: str = "SPY"
    rates_proxy: str = "TLT"
    universe: list[AssetSpec]
    fred: FredConfig = FredConfig()
    macro_series: list[MacroSeriesSpec] = []
    availability: AvailabilityConfig = AvailabilityConfig()
    paths: PathsConfig = PathsConfig()

    @field_validator("universe")
    @classmethod
    def _unique_tickers(cls, v: list[AssetSpec]) -> list[AssetSpec]:
        tickers = [a.ticker for a in v]
        if len(tickers) != len(set(tickers)):
            raise ValueError("duplicate tickers in universe")
        if len(tickers) < 2:
            raise ValueError("universe needs at least two assets for cross-sectional ranking")
        return v

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.universe]

    @property
    def asset_classes(self) -> dict[str, str]:
        return {a.ticker: a.asset_class for a in self.universe}


# ----------------------------------------------------------------------- features.yaml
class MomentumConfig(_Strict):
    return_lookbacks: list[int] = [5, 10, 20, 60, 120, 252]
    voladj_lookbacks: list[int] = [20, 60, 120, 252]
    skip_month: dict[str, int] = {"long": 252, "short": 21}
    xsrank_lookbacks: list[int] = [20, 60, 120, 252]


class VolatilityConfig(_Strict):
    lookbacks: list[int] = [20, 60, 120]
    downside_lookback: int = 60
    ratio: dict[str, int] = {"short": 20, "long": 120}
    parkinson_lookback: int = 20


class TrendConfig(_Strict):
    ema_spans: list[int] = [20, 50, 200]
    crossover: dict[str, int] = {"fast": 20, "slow": 50}
    slope_lookback: int = 60
    up_fraction_lookback: int = 20


class LiquidityConfig(_Strict):
    dollar_volume_lookback: int = 20
    volume_z_lookback: int = 60
    amihud_lookback: int = 20


class RiskConfig(_Strict):
    beta_lookbacks: list[int] = [60, 252]
    drawdown_lookback: int = 252
    max_drawdown_lookback: int = 120
    moments_lookback: int = 60


class CrossAssetConfig(_Strict):
    rel_market_lookbacks: list[int] = [20, 60, 120]
    rel_class_lookbacks: list[int] = [20, 60]
    corr_lookback: int = 60


class MarketConfig(_Strict):
    avg_corr_lookback: int = 60
    trend_sma: int = 200
    market_vol_lookback: int = 20


MacroTransform = Literal["level", "diff", "pct_change", "pct_rank", "zscore"]


class MacroFeatureSpec(_Strict):
    name: str
    series: str
    transform: MacroTransform
    periods: int | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _periods_required(self) -> MacroFeatureSpec:
        if self.transform != "level" and self.periods is None:
            raise ValueError(f"macro feature {self.name}: transform {self.transform} needs periods")
        return self


class FeaturesConfig(_Strict):
    status: Status
    min_history_sessions: int = Field(252, gt=0)
    momentum: MomentumConfig = MomentumConfig()
    volatility: VolatilityConfig = VolatilityConfig()
    trend: TrendConfig = TrendConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    risk: RiskConfig = RiskConfig()
    cross_asset: CrossAssetConfig = CrossAssetConfig()
    market: MarketConfig = MarketConfig()
    macro: list[MacroFeatureSpec] = []


# ------------------------------------------------------------------------- labels.yaml
class LabelsConfig(_Strict):
    status: Status
    signal_frequency: Literal["weekly", "daily"] = "weekly"
    horizon: int = Field(20, gt=0)
    alt_horizons: list[int] = []
    execution_delay_sessions: int = Field(1, ge=1)
    n_grades: int = Field(5, ge=2, le=31)


# -------------------------------------------------------------------------- model.yaml
class RankerConfig(_Strict):
    objective: Literal["rank:ndcg", "rank:pairwise"] = "rank:ndcg"
    eval_metric: str = "ndcg@5"
    early_stopping_rounds: int = 50
    inner_validation_fraction: float = Field(0.2, gt=0.0, lt=0.5)
    params: dict[str, float | int] = {}


class QuantileConfig(_Strict):
    alphas: list[float] = [0.05, 0.5, 0.95]
    early_stopping_rounds: int = 50
    params: dict[str, float | int] = {}

    @field_validator("alphas")
    @classmethod
    def _sorted_alphas(cls, v: list[float]) -> list[float]:
        if sorted(v) != v or any(not 0 < a < 1 for a in v):
            raise ValueError("quantile alphas must be sorted and in (0, 1)")
        return v


class RandomForestConfig(_Strict):
    n_estimators: int = 300
    max_depth: int | None = 6
    min_samples_leaf: int = 50
    max_features: float = 0.5


class BaselinesConfig(_Strict):
    momentum_feature: str = "mom_12_1"
    ridge_alpha: float = 10.0
    random_forest: RandomForestConfig = RandomForestConfig()


class TuningConfig(_Strict):
    enabled: bool = False
    n_trials: int = Field(40, gt=0)
    search_space: dict[str, list[float]] = {}


class RegimeRulesConfig(_Strict):
    vol_feature: str = "vix_pct_252"
    vol_fallback_feature: str = "market_vol_20"
    high_vol_quantile: float = Field(0.8, gt=0.0, lt=1.0)
    trend_feature: str = "market_above_sma200"


class HMMConfig(_Strict):
    enabled: bool = True
    n_states: int = Field(2, ge=2, le=5)
    covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"
    n_iter: int = 200
    inputs: list[str] = ["market_ret_1d", "market_vol_20"]


class RegimeConfig(_Strict):
    rules: RegimeRulesConfig = RegimeRulesConfig()
    hmm: HMMConfig = HMMConfig()
    add_regime_features: bool = True


class ModelConfig(_Strict):
    status: Status
    seed: int = 42
    ranker: RankerConfig = RankerConfig()
    quantile: QuantileConfig = QuantileConfig()
    baselines: BaselinesConfig = BaselinesConfig()
    tuning: TuningConfig = TuningConfig()
    regime: RegimeConfig = RegimeConfig()


# --------------------------------------------------------------------- validation.yaml
class BootstrapConfig(_Strict):
    n_resamples: int = 2000
    mean_block_length: int = 20
    confidence: float = 0.95


class ValidationConfig(_Strict):
    status: Status
    min_train_years: int = Field(5, gt=0)
    test_months: int = Field(6, gt=0)
    embargo_sessions: int = Field(5, ge=0)
    bootstrap: BootstrapConfig = BootstrapConfig()
    ic_newey_west_lags: int | None = None


# ----------------------------------------------------------------------- backtest.yaml
class DateRange(_Strict):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("date range end before start")
        return self


class SplitsConfig(_Strict):
    warmup_end: dt.date
    development: DateRange
    holdout: DateRange

    @model_validator(mode="after")
    def _non_overlapping(self) -> SplitsConfig:
        if not self.warmup_end < self.development.start:
            raise ValueError("warmup must end before the development period")
        if not self.development.end < self.holdout.start:
            raise ValueError("development period must end before the holdout")
        return self


class TimingConfig(_Strict):
    execution_delay_sessions: int = Field(1, ge=1)
    rebalance_every_n_signals: int = Field(4, ge=1)


class ConstraintsConfig(_Strict):
    long_only: bool = True
    w_max: float = Field(0.25, gt=0.0, le=1.0)
    asset_class_max: float = Field(0.5, gt=0.0, le=1.0)
    turnover_max: float = Field(0.5, gt=0.0, le=2.0)


class ExpectedReturnsConfig(_Strict):
    method: Literal["grinold"] = "grinold"
    ic_min_history: int = Field(26, ge=1)
    vol_feature: str = "vol_60"


class OptimizerConfig(_Strict):
    type: Literal["mean_variance", "cvar"] = "mean_variance"
    covariance: Literal["sample", "ledoit_wolf", "ewma"] = "ledoit_wolf"
    covariance_lookback: int = 252
    ewma_halflife: int = 60
    risk_aversion: float = Field(5.0, ge=0.0)
    turnover_penalty: float = Field(0.001, ge=0.0)
    cvar_beta: float = Field(0.95, gt=0.5, lt=1.0)
    cvar_kappa: float = Field(1.0, ge=0.0)
    cvar_scenarios: Literal["historical", "quantile_copula"] = "historical"
    n_copula_scenarios: int = 2000


class BenchmarksConfig(_Strict):
    momentum_top_k: int = 5
    sixty_forty: dict[str, float] = {"SPY": 0.6, "IEF": 0.4}
    market: str = "SPY"
    inverse_vol_feature: str = "vol_60"


class BacktestConfig(_Strict):
    status: Status
    splits: SplitsConfig
    timing: TimingConfig = TimingConfig()
    costs_bps: list[float] = [5.0, 10.0, 20.0]
    base_cost_bps: float = 10.0
    constraints: ConstraintsConfig = ConstraintsConfig()
    expected_returns: ExpectedReturnsConfig = ExpectedReturnsConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    benchmarks: BenchmarksConfig = BenchmarksConfig()
    risk_free_series: str | None = "DGS3MO"
    seed: int = 42


# ---------------------------------------------------------------------------- loader
class Secrets(BaseSettings):
    """API credentials from environment / ``.env`` (never committed)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    fred_api_key: str | None = None
    alphavantage_api_key: str | None = None
    sec_user_agent: str | None = None


class AppConfig(_Strict):
    root: Path
    data: DataConfig
    features: FeaturesConfig
    labels: LabelsConfig
    model: ModelConfig
    validation: ValidationConfig
    backtest: BacktestConfig

    @property
    def all_frozen(self) -> bool:
        return all(
            c.status == "frozen"
            for c in (
                self.data,
                self.features,
                self.labels,
                self.model,
                self.validation,
                self.backtest,
            )
        )

    def path(self, p: Path) -> Path:
        """Resolve a config path relative to the project root."""
        return p if p.is_absolute() else self.root / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.data.paths.raw)

    @property
    def curated_dir(self) -> Path:
        return self.path(self.data.paths.curated)

    @property
    def features_dir(self) -> Path:
        return self.path(self.data.paths.features)

    @property
    def artifacts_dir(self) -> Path:
        return self.path(self.data.paths.artifacts)

    def section_hash(self, *sections: str) -> str:
        """Stable short hash of selected config sections (for artefact keys)."""
        payload = {s: getattr(self, s).model_dump(mode="json") for s in sections}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def find_root(start: Path | None = None) -> Path:
    """Project root = ``$QUANTXGB_ROOT`` or the nearest parent containing ``configs/``."""
    env = os.environ.get("QUANTXGB_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open() as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def load_config(config_dir: Path | str | None = None, root: Path | None = None) -> AppConfig:
    root = (root or find_root()).resolve()
    cdir = Path(config_dir) if config_dir is not None else root / "configs"
    if not cdir.is_absolute():
        cdir = root / cdir
    return AppConfig(
        root=root,
        data=DataConfig.model_validate(_read_yaml(cdir / "data.yaml")),
        features=FeaturesConfig.model_validate(_read_yaml(cdir / "features.yaml")),
        labels=LabelsConfig.model_validate(_read_yaml(cdir / "labels.yaml")),
        model=ModelConfig.model_validate(_read_yaml(cdir / "model.yaml")),
        validation=ValidationConfig.model_validate(_read_yaml(cdir / "validation.yaml")),
        backtest=BacktestConfig.model_validate(_read_yaml(cdir / "backtest.yaml")),
    )
