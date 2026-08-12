"""
Baseline: always bet home underdogs (AvgCH > AvgCA) at closing AvgCH.

Also reports fair-odds (de-vigged) P/L for research comparison.

Usage:
  python scripts/eval_home_underdog.py
  python scripts/eval_home_underdog.py --fair
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "USA.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "tables"


def load_home_dogs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Season"] = df["Season"].astype(str)
    for c in ("AvgCH", "AvgCD", "AvgCA"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Season", "Res", "AvgCH", "AvgCD", "AvgCA"])
    df = df[df["Res"].isin(["H", "D", "A"])]
    df = df[(df["AvgCH"] > 1) & (df["AvgCD"] > 1) & (df["AvgCA"] > 1)]
    out = df.loc[df["AvgCH"] > df["AvgCA"]].copy()

    ph = 1.0 / out["AvgCH"]
    pd_ = 1.0 / out["AvgCD"]
    pa = 1.0 / out["AvgCA"]
    total = ph + pd_ + pa
    out["p_home_fair"] = ph / total
    out["fair_home_odds"] = 1.0 / out["p_home_fair"]
    out["home_win"] = (out["Res"] == "H").astype(int)
    return out.reset_index(drop=True)


def season_summary(df: pd.DataFrame, *, fair: bool) -> pd.DataFrame:
    odds = df["fair_home_odds"] if fair else df["AvgCH"]
    profit = np.where(df["home_win"].astype(bool), odds - 1.0, -1.0)
    work = df.copy()
    work["bet_odds"] = odds
    work["profit"] = profit

    g = work.groupby("Season", sort=True)
    summary = g.agg(
        n_bets=("profit", "size"),
        n_wins=("home_win", "sum"),
        win_rate=("home_win", "mean"),
        avg_bet_odds=("bet_odds", "mean"),
        avg_p_home_fair=("p_home_fair", "mean"),
        profit=("profit", "sum"),
    )
    summary["roi"] = summary["profit"] / summary["n_bets"]
    summary["gap"] = summary["win_rate"] - summary["avg_p_home_fair"]
    summary["n_wins"] = summary["n_wins"].astype(int)

    overall = pd.DataFrame(
        {
            "n_bets": [len(work)],
            "n_wins": [int(work["home_win"].sum())],
            "win_rate": [work["home_win"].mean()],
            "avg_bet_odds": [work["bet_odds"].mean()],
            "avg_p_home_fair": [work["p_home_fair"].mean()],
            "profit": [work["profit"].sum()],
            "roi": [work["profit"].sum() / len(work)],
            "gap": [work["home_win"].mean() - work["p_home_fair"].mean()],
        },
        index=pd.Index(["ALL"], name="Season"),
    )
    return pd.concat([summary, overall])


def print_report(summary: pd.DataFrame, *, fair: bool) -> None:
    title = (
        "Always bet HOME UNDERDOG at FAIR odds (no vig), 1u flat"
        if fair
        else "Always bet HOME UNDERDOG at AvgCH (with vig), 1u flat"
    )
    print(f"\n{title}\n")
    view = summary.copy()
    view["win_rate"] = view["win_rate"].map(lambda x: f"{100 * x:.1f}%")
    view["avg_p_home_fair"] = view["avg_p_home_fair"].map(lambda x: f"{100 * x:.1f}%")
    view["gap"] = view["gap"].map(lambda x: f"{100 * x:+.1f}pp")
    view["roi"] = view["roi"].map(lambda x: f"{100 * x:+.1f}%")
    view["profit"] = view["profit"].map(lambda x: f"{x:+.1f}")
    view["avg_bet_odds"] = view["avg_bet_odds"].map(lambda x: f"{x:.3f}")
    view = view.rename(
        columns={
            "n_bets": "bets",
            "n_wins": "wins",
            "win_rate": "hit%",
            "avg_bet_odds": "avg_odds",
            "avg_p_home_fair": "fair%",
            "profit": "P/L (u)",
            "roi": "ROI",
        }
    )
    print(view[["bets", "wins", "hit%", "fair%", "gap", "avg_odds", "P/L (u)", "ROI"]].to_string())
    all_row = summary.loc["ALL"]
    print(
        f"\nVerdict: {'PROFITABLE' if all_row['profit'] > 0 else 'NOT profitable'} "
        f"overall — {all_row['profit']:+.1f}u on {int(all_row['n_bets'])} bets "
        f"({100 * all_row['roi']:+.1f}% ROI)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline home-underdog backtest.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--fair",
        action="store_true",
        help="Price bets at de-vigged fair home odds instead of AvgCH.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    dogs = load_home_dogs(args.csv)
    summary = season_summary(dogs, fair=args.fair)
    out = args.output or (
        DEFAULT_OUT_DIR
        / ("home_underdog_fair_by_season.csv" if args.fair else "home_underdog_by_season.csv")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, float_format="%.6f")
    print_report(summary, fair=args.fair)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
