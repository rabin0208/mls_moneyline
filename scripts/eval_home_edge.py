"""
Compare model P(home) to closing fair market and simulate home-edge bets.

Pays out at AvgCH (vig included). Optional filters:
  - all: any game where model edge ≥ threshold
  - home_dog: only when home is closing underdog (AvgCH > AvgCA)
  - home_fav: only when home is closing favorite (AvgCH < AvgCA)

Usage:
  python scripts/eval_home_edge.py
  python scripts/eval_home_edge.py --model logistic --filter home_dog
  python scripts/eval_home_edge.py --model elo --filter all,home_dog
  python scripts/eval_home_edge.py --model both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from model_utils import FEATURE_COLS, TEST_SEASONS
from odds_utils import flat_bet_profit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "mls_featured.csv"
HOME_ELO_CSV = DATA_DIR / "mls_home_elo.csv"


def summarize_probs(name: str, y: pd.Series, p: pd.Series) -> None:
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
        auc_s = f"{auc:.4f}"
    except ValueError:
        auc_s = "n/a"
    print(
        f"  {name:12s}  acc={acc:.4f}  auc={auc_s}  "
        f"logloss={log_loss(y, p):.4f}  brier={brier_score_loss(y, p):.4f}"
    )


def load_featured(seasons: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(str)
    if seasons is not None:
        df = df.loc[df["season"].isin(seasons)].copy()
    df["season_index"] = pd.to_numeric(df["season_index"], errors="coerce").fillna(0)
    needed = FEATURE_COLS + ["home_win", "p_home_fair", "AvgCH", "AvgCA"]
    return df.dropna(subset=needed).reset_index(drop=True)


def score_logistic(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "home_logistic.pkl")
    scaler = joblib.load(MODELS_DIR / "home_scaler.pkl")
    out = df.copy()
    X = scaler.transform(out[FEATURE_COLS].astype(float))
    out["p_home_model"] = model.predict_proba(X)[:, 1]
    return out


def score_elo(df: pd.DataFrame) -> pd.DataFrame:
    elo_path = HOME_ELO_CSV
    cal_path = MODELS_DIR / "home_elo_calibrator.pkl"
    if not elo_path.exists() or not cal_path.exists():
        raise SystemExit(
            "Missing home Elo artifacts. Run: python scripts/fit_home_elo.py"
        )
    elo = pd.read_csv(elo_path)
    calibrator = joblib.load(cal_path)
    out = df.merge(elo[["game_id", "elo_diff"]], on="game_id", how="left")
    if out["elo_diff"].isna().any():
        raise SystemExit("Some games missing elo_diff; re-run fit_home_elo.py")
    out["p_home_model"] = calibrator.predict_proba(out[["elo_diff"]].astype(float))[:, 1]
    return out


def apply_filter(df: pd.DataFrame, filt: str) -> pd.DataFrame:
    if filt == "all":
        return df
    if filt == "home_dog":
        return df.loc[df["AvgCH"] > df["AvgCA"]].copy()
    if filt == "home_fav":
        return df.loc[df["AvgCH"] < df["AvgCA"]].copy()
    raise ValueError(f"Unknown filter: {filt}")


def roi_row(df: pd.DataFrame, edge: float, filt: str, model_name: str) -> dict:
    pool = apply_filter(df, filt)
    pool = pool.copy()
    pool["edge_home"] = pool["p_home_model"] - pool["p_home_fair"]
    bets = pool.loc[pool["edge_home"] >= edge]
    if bets.empty:
        return {
            "model": model_name,
            "filter": filt,
            "edge": edge,
            "n_bets": 0,
            "n_wins": 0,
            "hit_rate": float("nan"),
            "avg_odds": float("nan"),
            "profit": 0.0,
            "roi": float("nan"),
        }
    profits = [
        flat_bet_profit(float(o), bool(w))
        for o, w in zip(bets["AvgCH"], bets["home_win"].astype(bool))
    ]
    pl = float(np.sum(profits))
    n = len(bets)
    return {
        "model": model_name,
        "filter": filt,
        "edge": edge,
        "n_bets": n,
        "n_wins": int(bets["home_win"].sum()),
        "hit_rate": float(bets["home_win"].mean()),
        "avg_odds": float(bets["AvgCH"].mean()),
        "profit": pl,
        "roi": pl / n,
    }


def evaluate_model(
    df: pd.DataFrame,
    model_name: str,
    filters: list[str],
    edges: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = df.copy()
    scored["edge_home"] = scored["p_home_model"] - scored["p_home_fair"]

    y = scored["home_win"].astype(int)
    print(f"\n===== {model_name} =====")
    print("Probability quality (home_win):")
    summarize_probs(model_name, y, scored["p_home_model"])
    summarize_probs("market", y, scored["p_home_fair"])

    rows = []
    for filt in filters:
        print(f"\nFlat 1u home bets [{filt}] (payout AvgCH, vig included):")
        print(
            f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  "
            f"{'P/L':>8}  {'ROI':>7}  avg_odds"
        )
        for e in edges:
            row = roi_row(scored, e, filt, model_name)
            rows.append(row)
            if row["n_bets"] == 0:
                print(f"  {e:6.2f}      0      0     n/a     +0.0      n/a      n/a")
                continue
            print(
                f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  "
                f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
                f"{100 * row['roi']:+6.1f}%  {row['avg_odds']:.3f}"
            )
    return scored, pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Home model vs market edge evaluation.")
    p.add_argument(
        "--model",
        choices=["logistic", "elo", "both"],
        default="both",
        help="Which home model(s) to evaluate.",
    )
    p.add_argument(
        "--filter",
        type=str,
        default="all,home_dog,home_fav",
        help="Comma-separated filters: all, home_dog, home_fav.",
    )
    p.add_argument(
        "--seasons",
        type=str,
        default=",".join(TEST_SEASONS),
        help="Comma-separated seasons to evaluate (default: holdout).",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated edge thresholds.",
    )
    args = p.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    filters = [f.strip() for f in args.filter.split(",") if f.strip()]
    edges = [float(x) for x in args.edge.split(",") if x.strip()]

    base = load_featured(seasons)
    print(f"Scoring seasons {seasons}...  games={len(base)}")
    print(
        f"  Home rate={base['home_win'].mean():.3f}  "
        f"fair P(home)={base['p_home_fair'].mean():.3f}  "
        f"home dogs={(base['AvgCH'] > base['AvgCA']).sum()}"
    )

    # Always-home baselines on the same pool
    for filt in filters:
        pool = apply_filter(base, filt)
        if pool.empty:
            continue
        pl = float(
            sum(
                flat_bet_profit(float(o), bool(w))
                for o, w in zip(pool["AvgCH"], pool["home_win"].astype(bool))
            )
        )
        print(
            f"  baseline always-home [{filt}]: n={len(pool)}  "
            f"P/L={pl:+.1f}  ROI={100 * pl / len(pool):+.1f}%"
        )

    models: list[tuple[str, callable]] = []
    if args.model in ("logistic", "both"):
        models.append(("logistic", score_logistic))
    if args.model in ("elo", "both"):
        models.append(("elo", score_elo))

    all_roi = []
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, scorer in models:
        if name == "logistic" and not (MODELS_DIR / "home_logistic.pkl").exists():
            raise SystemExit("Missing home logistic. Run: python scripts/fit_home_model.py")
        scored = scorer(base)
        scored_full, roi = evaluate_model(scored, name, filters, edges)
        all_roi.append(roi)

        keep = [
            "game_id",
            "game_date",
            "season",
            "home_name",
            "away_name",
            "home_win",
            "AvgCH",
            "AvgCA",
            "p_home_fair",
            "p_home_model",
            "edge_home",
        ]
        out_pred = TABLES_DIR / f"home_{name}_predictions_holdout.csv"
        scored_full[keep].to_csv(out_pred, index=False)
        print(f"Wrote {out_pred}")

    roi_df = pd.concat(all_roi, ignore_index=True)
    roi_path = TABLES_DIR / "home_edge_roi.csv"
    roi_df.to_csv(roi_path, index=False, float_format="%.6f")
    print(f"\nWrote {roi_path}")

    pos = roi_df.loc[roi_df["profit"] > 0].sort_values("roi", ascending=False)
    print("\nPositive P/L cells (if any):")
    if pos.empty:
        print("  none")
    else:
        view = pos.copy()
        view["hit_rate"] = view["hit_rate"].map(lambda x: f"{100 * x:.1f}%")
        view["roi"] = view["roi"].map(lambda x: f"{100 * x:+.1f}%")
        view["profit"] = view["profit"].map(lambda x: f"{x:+.1f}")
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
