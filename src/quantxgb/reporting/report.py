"""Markdown research report for a run (numbers only — interpretation is the owner's)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl

from quantxgb.research.runs import read_manifest
from quantxgb.utils.io import read_json


def _f(x: Any, nd: int = 3, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, int | float):
        return f"{x * 100:.{max(nd - 2, 1)}f}%" if pct else f"{x:.{nd}f}"
    return str(x)


def render_run_report(run_dir: Path) -> str:
    m = read_manifest(run_dir)
    met = read_json(run_dir / "metrics.json")
    folds = read_json(run_dir / "fold_metrics.json")
    lines = [
        f"# Run report — `{m['run_id']}`",
        "",
        f"- Created: {m['created']}  ·  git: `{m['git_sha']}`  ·  config hash: `{m['config_hash']}`",
        f"- Data hash: `{m.get('data_hash', 'n/a')}`",
        f"- Hypothesis (pre-registered by owner): {m.get('hypothesis') or '_none given_'}",
    ]
    unfrozen = [k for k, v in m["config_status"].items() if v != "frozen"]
    if unfrozen:
        lines.append(f"- ⚠️ Configs not frozen: {', '.join(unfrozen)} — exploratory run.")
    if m.get("shuffle_labels"):
        lines.append("- ⚠️ SANITY RUN with shuffled labels: IC must be ≈ 0.")
    lines += ["", "## Out-of-sample signal quality (walk-forward, development period)", ""]
    lines.append(
        "| Model | IC mean | 95% CI | ICIR | t (NW) | % IC>0 | NDCG@5 | Top-k hit | Q-spread | Dates |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in met["models"].items():
        lines.append(
            f"| {name} | {_f(s.get('ic_mean'), 4)} | [{_f(s.get('ic_mean_ci_low'), 4)}, {_f(s.get('ic_mean_ci_high'), 4)}] "
            f"| {_f(s.get('icir'))} | {_f(s.get('ic_tstat_nw'), 2)} | {_f(s.get('ic_pct_positive'), pct=True)} "
            f"| {_f(s.get('ndcg_at_5'))} | {_f(s.get('top_k_hit_rate'), pct=True)} | {_f(s.get('quantile_spread'), 4)} "
            f"| {s.get('n_dates')} |"
        )
    lines += [
        "",
        f"Newey–West lags: {met.get('newey_west_lags')} · folds: {met.get('n_folds')}",
        "",
    ]
    lines += ["## IC by regime (rule-based regimes, thresholds fitted per training window)", ""]
    regimes = sorted({r for d in met.get("by_regime", {}).values() for r in d})
    if regimes:
        lines.append("| Model | " + " | ".join(regimes) + " |")
        lines.append("|---|" + "---|" * len(regimes))
        for name, d in met["by_regime"].items():
            cells = [
                f"{_f(d[r]['ic_mean'], 4)} (n={d[r]['n_dates']})" if r in d else "–"
                for r in regimes
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    q = met.get("quantile")
    if q:
        lines += [
            "",
            "## Quantile forecasts vs. unconditional-quantile baseline",
            "",
            "| Metric | XGB quantile | Baseline |",
            "|---|---|---|",
        ]
        for k in sorted(
            k for k in q if not k.startswith("baseline_") and k != "coverage_by_regime"
        ):
            lines.append(f"| {k} | {_f(q[k], 5)} | {_f(q.get('baseline_' + k), 5)} |")
        if q.get("coverage_by_regime"):
            lines.append("")
            lines.append(
                "Coverage by regime: "
                + ", ".join(f"{k}: {_f(v, pct=True)}" for k, v in q["coverage_by_regime"].items())
            )
    lines += [
        "",
        "## Folds",
        "",
        "| Fold | Test | n_train | Trees | XGB IC | Momentum IC | Crossing |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {f['test_start']} → {f['test_end']} | {f['n_train']} | {f['ranker_trees']} "
            f"| {_f(f.get('ic_mean_xgb'), 4)} | {_f(f.get('ic_mean_momentum'), 4)} | {_f(f.get('quantile_crossing_rate'), pct=True)} |"
        )
    imp_path = run_dir / "feature_importance.parquet"
    if imp_path.exists():
        imp = (
            pl.read_parquet(imp_path)
            .group_by("feature")
            .agg(pl.col("mean_abs_shap").mean())
            .sort("mean_abs_shap", descending=True)
            .head(15)
        )
        lines += [
            "",
            "## Top features (mean |SHAP| of the ranker, averaged over folds)",
            "",
            "> Attribution is not causation; correlated features share credit arbitrarily.",
            "",
            "| Feature | mean |SHAP| |",
            "|---|---|",
        ]
        lines += [
            f"| `{r['feature']}` | {r['mean_abs_shap']:.5f} |" for r in imp.iter_rows(named=True)
        ]
    bt_path = run_dir / "backtest" / "metrics.json"
    if bt_path.exists():
        lines += render_backtest(read_json(bt_path))
    lines += [
        "",
        "## Assumptions",
        "",
        "- Execution at the close of t+1; costs are fixed one-way bps on traded notional (stress "
        "assumptions, not measured spreads or market impact).",
        "- Universe chosen ex ante with hindsight (surviving ETFs): residual survivorship bias.",
        "- Development-period walk-forward only; the final holdout is not used here.",
        "",
        "## Interpretation (owner)",
        "",
        "_To be written by the owner. Do not edit numbers above._",
        "",
    ]
    return "\n".join(lines)


def render_backtest(bt: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## Backtest ({bt['label']}, {bt['schedule'][0]} → {bt['end']}, {bt['n_rebalances']} rebalances)",
        "",
    ]
    strategies = sorted(bt["strategies"].items(), key=lambda kv: (kv[1]["is_baseline"], kv[0]))
    costs = sorted(next(iter(bt["strategies"].values()))["by_cost"], key=float)
    for cost in costs:
        lines += [
            f"### Cost {cost} bps",
            "",
            "| Strategy | CAGR | Vol | Sharpe | Sharpe 95% CI | Sortino | MaxDD | Turnover/yr | Cost drag/yr | IR vs EW |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in strategies:
            p = s["by_cost"][cost]
            ci = (
                f"[{_f(p.get('sharpe_ci_low'), 2)}, {_f(p.get('sharpe_ci_high'), 2)}]"
                if "sharpe_ci_low" in p
                else "–"
            )
            lines.append(
                f"| {name}{' (baseline)' if s['is_baseline'] else ''} | {_f(p.get('cagr'), pct=True)} | {_f(p.get('ann_vol'), pct=True)} "
                f"| {_f(p.get('sharpe'), 2)} | {ci} | {_f(p.get('sortino'), 2)} | {_f(p.get('max_drawdown'), pct=True)} "
                f"| {_f(p.get('turnover_annual'), 2)} | {_f(p.get('cost_drag_annual'), pct=True)} | {_f(p.get('information_ratio'), 2)} |"
            )
        lines.append("")
    lines += [
        "### Overfitting diagnostics",
        "",
        "| Strategy | DSR (abs) | DSR (active vs EW) | Trials | Optimizer relaxations |",
        "|---|---|---|---|---|",
    ]
    for name, s in strategies:
        if s["is_baseline"]:
            continue
        d, da = s.get("deflated_sharpe", {}), s.get("deflated_sharpe_active", {})
        lines.append(
            f"| {name} | {_f(d.get('dsr'))} | {_f(da.get('dsr'))} | {d.get('n_trials')} | {s.get('n_relaxations', 0)} |"
        )
    pbo = bt.get("pbo", {})
    lines += [
        "",
        f"PBO (CSCV over {len(pbo.get('strategies', []))} strategies, {pbo.get('n_splits')} splits): **{_f(pbo.get('pbo'))}**",
        "",
    ]
    return lines


def write_run_report(run_dir: Path) -> Path:
    path = run_dir / "report.md"
    path.write_text(render_run_report(run_dir))
    return path
