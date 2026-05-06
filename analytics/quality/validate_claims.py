#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "synthetic_claims.csv"


def main() -> int:
    df = pd.read_csv(CSV)
    need = {"text", "lang", "label"}
    if not need <= set(df.columns):
        print("bad columns", file=sys.stderr)
        return 1
    if df["text"].isna().any() or (df["text"].str.len() < 8).any():
        print("bad text", file=sys.stderr)
        return 1
    if not df["lang"].isin(["en", "fa", "tr"]).all():
        print("unexpected lang", file=sys.stderr)
        return 1
    if not df["label"].between(0, 3).all():
        print("label range", file=sys.stderr)
        return 1
    print("OK: synthetic claims pass quality checks.", len(df), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
