"""
Feature engineering for MLS matches.

Adds lagged win form (last 5), H2H wins, and season index.
Writes data/mls_featured.csv.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from model_utils import (
    LAG_WINDOW,
    FEATURE_COLS,
    h2h_lag_column_names,
    lag_vector,
    team_lag_column_names,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

MATCHES_CSV = DATA_DIR / "mls_matches.csv"
OUT_CSV = DATA_DIR / "mls_featured.csv"


def add_lagged_team_features(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    team_wins: dict[int, list[int]] = defaultdict(list)

    cnames = team_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    def append_side(team_id: int, prefix: str) -> None:
        tw = lag_vector(team_wins[team_id], window, 0.0)
        for k in range(window):
            cols[f"{prefix}_win_lag_{k + 1}"].append(tw[k])

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        home_won = int(row["home_win"])
        away_won = int(row["away_win"])

        append_side(home_id, "home")
        append_side(away_id, "away")

        team_wins[home_id].append(home_won)
        team_wins[away_id].append(away_won)

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_lagged_h2h(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    h2h: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)
    cnames = h2h_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_won = int(row["home_win"])

        key = (min(home_id, away_id), max(home_id, away_id))
        past = [p for p in h2h[key] if p[0] < game_date][-window:]
        wins_chrono: list[int] = []
        for _past_date, past_home_id, past_home_won in past:
            if past_home_id == home_id:
                wins_chrono.append(past_home_won)
            else:
                wins_chrono.append(1 - past_home_won)

        lags = lag_vector(wins_chrono, window, 0.0)
        for k in range(window):
            cols[f"home_h2h_win_lag_{k + 1}"].append(lags[k])

        h2h[key].append((game_date, home_id, home_won))

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    mapping = {s: i for i, s in enumerate(seasons)}
    out = df.copy()
    out["season_index"] = out["season"].map(mapping).astype(int)
    return out


def main() -> None:
    print(f"Loading {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    print(f"  Building lags (window={LAG_WINDOW})...")
    df = add_lagged_team_features(df)
    df = add_lagged_h2h(df)
    df = add_season_index(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Rows: {len(df)}  features: {len(FEATURE_COLS)}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
