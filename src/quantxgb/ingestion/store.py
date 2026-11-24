"""Curated storage: Parquet files + a DuckDB catalogue of views (PRD FR-DATA-07)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from quantxgb.utils.io import frame_hash, read_json, write_json, write_parquet_atomic

PRICES = "prices.parquet"
MACRO = "macro_pit.parquet"
MANIFEST = "manifest.json"


class CuratedStore:
    def __init__(self, curated_dir: Path) -> None:
        self.dir = curated_dir

    @property
    def prices_path(self) -> Path:
        return self.dir / PRICES

    @property
    def macro_path(self) -> Path:
        return self.dir / MACRO

    def write(self, prices: pl.DataFrame, macro: pl.DataFrame | None) -> dict[str, str]:
        write_parquet_atomic(prices, self.prices_path)
        hashes = {"prices": frame_hash(prices)}
        if macro is not None:
            write_parquet_atomic(macro, self.macro_path)
            hashes["macro"] = frame_hash(macro)
        elif self.macro_path.exists():
            self.macro_path.unlink()
        write_json(hashes, self.dir / MANIFEST)
        return hashes

    def prices(self) -> pl.DataFrame:
        if not self.prices_path.exists():
            raise FileNotFoundError(f"{self.prices_path} missing — run `quantxgb data build`")
        return pl.read_parquet(self.prices_path)

    def macro(self) -> pl.DataFrame | None:
        return pl.read_parquet(self.macro_path) if self.macro_path.exists() else None

    def data_hash(self) -> str:
        m = self.dir / MANIFEST
        if not m.exists():
            return "none"
        h = read_json(m)
        return "-".join(f"{k}:{v}" for k, v in sorted(h.items()))

    def build_duckdb(self, db_path: Path) -> Path:
        """(Re)create a DuckDB file exposing curated Parquet files as views."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(db_path)) as con:
            for view, path in (("prices", self.prices_path), ("macro_pit", self.macro_path)):
                if path.exists():
                    escaped = str(path.resolve()).replace("'", "''")
                    con.execute(
                        f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{escaped}')"
                    )
        return db_path
