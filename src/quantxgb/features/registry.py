"""Feature registry (PRD FR-FEAT-02).

Every feature declares its formula, lookback, economic rationale and availability rule.
``research/feature_dictionary.md`` is generated from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scope = Literal["asset", "cross_sectional", "market", "macro", "regime"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    formula: str
    lookback: int
    rationale: str
    scope: Scope = "asset"
    availability: str = "close of t (uses sessions ≤ t only)"
    model_input: bool = True  # False: kept for regime models / diagnostics, not fed to rankers


def render_dictionary(specs: list[FeatureSpec]) -> str:
    lines = [
        "# Feature dictionary",
        "",
        "_Generated from `quantxgb.features` — do not edit by hand. "
        "Regenerate with `quantxgb features dictionary`._",
        "",
        "Notation: `TR` total-return index, `r` daily log return of TR, `σ_k` annualised "
        "std of `r` over k sessions, `M` market proxy, all windows trailing and inclusive of t.",
        "",
    ]
    families: dict[str, list[FeatureSpec]] = {}
    for s in specs:
        families.setdefault(s.family, []).append(s)
    for fam, items in families.items():
        lines += [f"## {fam}", "", "| Feature | Formula | Lookback | Scope | Rationale |"]
        lines.append("|---|---|---|---|---|")
        for s in items:
            lines.append(f"| `{s.name}` | {s.formula} | {s.lookback} | {s.scope} | {s.rationale} |")
        lines.append("")
    return "\n".join(lines)
