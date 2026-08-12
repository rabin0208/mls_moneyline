"""
Compare multinomial P(H/D/A) to closing fair market and simulate edge bets.

For each game, compute edge on each side:
  edge_side = P_model(side) - P_fair(side)

Betting rule (default): if max edge ≥ threshold, bet that side once at
AvgCH / AvgCD / AvgCA (vig included).

Usage:
  python scripts/eval_multinomial_edge.py
  python scripts/eval_multinomial_edge.py --sides H,A
  python scripts/eval_multinomial_edge.py --edge 0,0.03,0.05,0.08
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from model_utils import FEATURE_COLS, TEST_SEASONS
from odds_utils import flat_bet_profit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "mls_featured.csv"
CLASS_ORDER = ["H", "D", "A"]
ODDS_COL = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}
FAIR_COL = {"H": "p_home_fair", "D": "p_draw_fair", "A": "p_away_fair"}
PROB_COL = {"H": "p_home_model", "D": "p_draw_model", "A": "p_away_model"}


def load_games(seasons: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(str)
    if seasons is not None:
        df = df.loc[df["season"].isin(seasons)].copy()
    df["season_index"] = pd.to_numeric(df["season_index"], errors="coerce").fillna(0)
    needed = FEATURE_COLS + [
        "Res",
        "AvgCH",
        "AvgCD",
        "AvgCA",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
    ]
    df = df.dropna(subset=needed)
    df = df.loc[df["Res"].isin(CLASS_ORDER)].copy()
    return df.reset_index(drop=True)


def score_model(df: pd.DataFrame) -> pd.DataFrame:
    model_path = MODELS_DIR / "multinomial_logistic.pkl"
    scaler_path = MODELS_DIR / "multinomial_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise SystemExit(
            "Missing multinomial model. Run: python scripts/fit_multinomial_model.py"
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    out = df.copy()
    X = scaler.transform(out[FEATURE_COLS].astype(float))
    proba = model.predict_proba(X)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    for side in CLASS_ORDER:
        out[PROB_COL[side]] = proba[:, class_to_idx[side]]
    out["mkt_pick"] = out[["p_home_fair", "p_draw_fair", "p_away_fair"]].idxmax(axis=1).map(
        {
            "p_home_fair": "H",
            "p_draw_fair": "D",
            "p_away_fair": "A",
        }
    )
    out["model_pick"] = out[[PROB_COL[s] for s in CLASS_ORDER]].idxmax(axis=1).map(
        {
            PROB_COL["H"]: "H",
            PROB_COL["D"]: "D",
            PROB_COL["A"]: "A",
        }
    )
    for side in CLASS_ORDER:
        out[f"edge_{side}"] = out[PROB_COL[side]] - out[FAIR_COL[side]]
    return out


def summarize_quality(df: pd.DataFrame) -> None:
    y = df["Res"]
    y_idx = y.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p_model = df[[PROB_COL[s] for s in CLASS_ORDER]].to_numpy()
    p_mkt = df[[FAIR_COL[s] for s in CLASS_ORDER]].to_numpy()

    print("\nProbability quality (1X2):")
    print(
        f"  {'model':12s}  acc={accuracy_score(y, df['model_pick']):.4f}  "
        f"logloss={log_loss(y_idx, p_model, labels=[0, 1, 2]):.4f}"
    )
    print(
        f"  {'market':12s}  acc={accuracy_score(y, df['mkt_pick']):.4f}  "
        f"logloss={log_loss(y_idx, p_mkt, labels=[0, 1, 2]):.4f}"
    )


def pick_bets(df: pd.DataFrame, edge: float, sides: list[str]) -> pd.DataFrame:
    """Bet the single best-edge side among `sides` if edge ≥ threshold."""
    out = df.copy()
    bet_side: list[str | None] = []
    bet_odds: list[float | None] = []
    bet_won: list[bool | None] = []
    bet_profit: list[float] = []
    bet_edge: list[float | None] = []

    for _, row in out.iterrows():
        best_side = None
        best_edge = -1e9
        for side in sides:
            e = float(row[f"edge_{side}"])
            if e > best_edge:
                best_edge = e
                best_side = side
        if best_side is None or best_edge < edge:
            bet_side.append(None)
            bet_odds.append(None)
            bet_won.append(None)
            bet_profit.append(0.0)
            bet_edge.append(None)
            continue
        odds = float(row[ODDS_COL[best_side]])
        won = str(row["Res"]) == best_side
        bet_side.append(best_side)
        bet_odds.append(odds)
        bet_won.append(won)
        bet_profit.append(flat_bet_profit(odds, won))
        bet_edge.append(best_edge)

    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_won"] = bet_won
    out["bet_profit"] = bet_profit
    out["bet_edge"] = bet_edge
    return out


def roi_row(bets: pd.DataFrame, edge: float, sides_label: str) -> dict:
    placed = bets.dropna(subset=["bet_side"])
    if placed.empty:
        return {
            "sides": sides_label,
            "edge": edge,
            "n_bets": 0,
            "n_wins": 0,
            "hit_rate": float("nan"),
            "profit": 0.0,
            "roi": float("nan"),
            "n_H": 0,
            "n_D": 0,
            "n_A": 0,
            "avg_odds": float("nan"),
        }
    return {
        "sides": sides_label,
        "edge": edge,
        "n_bets": int(len(placed)),
        "n_wins": int(placed["bet_won"].sum()),
        "hit_rate": float(placed["bet_won"].mean()),
        "profit": float(placed["bet_profit"].sum()),
        "roi": float(placed["bet_profit"].sum() / len(placed)),
        "n_H": int((placed["bet_side"] == "H").sum()),
        "n_D": int((placed["bet_side"] == "D").sum()),
        "n_A": int((placed["bet_side"] == "A").sum()),
        "avg_odds": float(placed["bet_odds"].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Multinomial model vs market edge bets.")
    p.add_argument(
        "--seasons",
        type=str,
        default=",".join(TEST_SEASONS),
        help="Comma-separated seasons (default: holdout).",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated edge thresholds.",
    )
    p.add_argument(
        "--sides",
        type=str,
        default="H,D,A",
        help="Comma-separated sides eligible to bet (subset of H,D,A).",
    )
    args = p.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    edges = [float(x) for x in args.edge.split(",") if x.strip()]
    sides = [s.strip().upper() for s in args.sides.split(",") if s.strip()]
    for s in sides:
        if s not in CLASS_ORDER:
            raise SystemExit(f"Invalid side {s}; use H, D, A")
    sides_label = ",".join(sides)

    print(f"Scoring seasons {seasons}...")
    games = score_model(load_games(seasons))
    print(f"  Games: {len(games)}")
    summarize_quality(games)

    rows = []
    print(f"\nFlat 1u edge bets (best side among [{sides_label}], payout closing Avg*):")
    print(
        f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  "
        f"{'P/L':>8}  {'ROI':>7}  H/D/A  avg_odds"
    )
    for e in edges:
        scored = pick_bets(games, e, sides)
        row = roi_row(scored, e, sides_label)
        rows.append(row)
        if row["n_bets"] == 0:
            print(f"  {e:6.2f}      0      0     n/a     +0.0      n/a  0/0/0      n/a")
            continue
        print(
            f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  "
            f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
            f"{100 * row['roi']:+6.1f}%  "
            f"{row['n_H']}/{row['n_D']}/{row['n_A']}  {row['avg_odds']:.3f}"
        )

    # Also evaluate H,A only and A only as common directional slices
    extra_specs = []
    if sides == CLASS_ORDER:
        extra_specs = [("H,A", ["H", "A"]), ("A", ["A"]), ("H", ["H"]), ("D", ["D"])]
    for label, side_list in extra_specs:
        print(f"\nFlat 1u edge bets (best side among [{label}]):")
        print(
            f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  "
            f"{'P/L':>8}  {'ROI':>7}  H/D/A  avg_odds"
        )
        for e in edges:
            scored = pick_bets(games, e, side_list)
            row = roi_row(scored, e, label)
            rows.append(row)
            if row["n_bets"] == 0:
                print(f"  {e:6.2f}      0      0     n/a     +0.0      n/a  0/0/0      n/a")
                continue
            print(
                f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  "
                f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
                f"{100 * row['roi']:+6.1f}%  "
                f"{row['n_H']}/{row['n_D']}/{row['n_A']}  {row['avg_odds']:.3f}"
            )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    roi_path = TABLES_DIR / "multinomial_edge_roi.csv"
    pd.DataFrame(rows).to_csv(roi_path, index=False, float_format="%.6f")
    print(f"\nWrote {roi_path}")

    # Predictions at edge=0 for inspection
    scored0 = pick_bets(games, 0.0, sides)
    keep = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "Res",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
        "p_home_model",
        "p_draw_model",
        "p_away_model",
        "edge_H",
        "edge_D",
        "edge_A",
        "model_pick",
        "mkt_pick",
        "bet_side",
        "bet_odds",
        "bet_edge",
        "bet_won",
        "bet_profit",
    ]
    pred_path = TABLES_DIR / "multinomial_predictions_holdout.csv"
    scored0[keep].to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")

    pos = pd.DataFrame(rows)
    pos = pos.loc[pos["profit"] > 0].sort_values("roi", ascending=False)
    print("\nPositive P/L cells (if any):")
    if pos.empty:
        print("  none")
    else:
        view = pos.copy()
        view["hit_rate"] = view["hit_rate"].map(
            lambda x: f"{100 * x:.1f}%" if pd.notna(x) else "n/a"
        )
        view["roi"] = view["roi"].map(lambda x: f"{100 * x:+.1f}%" if pd.notna(x) else "n/a")
        view["profit"] = view["profit"].map(lambda x: f"{x:+.1f}")
        print(
            view[
                ["sides", "edge", "n_bets", "n_wins", "hit_rate", "profit", "roi", "n_H", "n_D", "n_A"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
