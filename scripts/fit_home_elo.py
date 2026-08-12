"""
Fit Elo calibrated to P(home win), then save for home-edge evaluation.

Grid-searches K and home advantage on train seasons using home_win log-loss.

Usage:
  python scripts/fit_home_elo.py
  python scripts/fit_home_elo.py --k 40 --home-adv 140
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from elo_utils import DEFAULT_HOME_ADV, DEFAULT_K, EloState
from model_utils import TEST_SEASONS, train_test_season_sets

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MATCHES_CSV = DATA_DIR / "mls_matches.csv"

K_GRID = [15, 20, 25, 30, 40]
HOME_ADV_GRID = [60, 80, 100, 120, 140]


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(str)
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    needed = [
        "game_id",
        "game_date",
        "season",
        "home_id",
        "away_id",
        "home_name",
        "away_name",
        "Res",
        "home_win",
        "p_home_fair",
        "AvgCH",
        "AvgCA",
    ]
    return df.dropna(subset=needed).reset_index(drop=True)


def walk_elo(df: pd.DataFrame, k: float, home_adv: float) -> tuple[pd.DataFrame, EloState]:
    state = EloState(k=k, home_adv=home_adv)
    rows: list[dict] = []
    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        r_home, r_away, elo_diff, e_home = state.pre_match(home_id, away_id)
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "elo_home": r_home,
                "elo_away": r_away,
                "elo_diff": elo_diff,
                "elo_e_home": e_home,
            }
        )
        state.update(home_id, away_id, str(row["Res"]))
    out = df.copy()
    out = out.merge(pd.DataFrame(rows), on="game_id", how="left")
    return out, state


def fit_calibrator(train: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train[["elo_diff"]].astype(float), train["home_win"].astype(int))
    return model


def train_logloss(df: pd.DataFrame, k: float, home_adv: float, train_seasons: set[str]) -> float:
    walked, _ = walk_elo(df, k=k, home_adv=home_adv)
    train = walked.loc[walked["season"].isin(train_seasons)]
    if train.empty:
        return float("inf")
    model = fit_calibrator(train)
    p = model.predict_proba(train[["elo_diff"]].astype(float))[:, 1]
    return float(log_loss(train["home_win"].astype(int), p))


def grid_search(df: pd.DataFrame, train_seasons: set[str]) -> tuple[float, float, float]:
    best = (float("inf"), DEFAULT_K, DEFAULT_HOME_ADV)
    print("Grid-searching K × home_adv on train home_win log-loss...")
    for k in K_GRID:
        for h in HOME_ADV_GRID:
            ll = train_logloss(df, k, h, train_seasons)
            print(f"  K={k:5.1f}  H={h:5.1f}  train_logloss={ll:.4f}")
            if ll < best[0]:
                best = (ll, float(k), float(h))
    print(f"Best: K={best[1]}, home_adv={best[2]} (train logloss={best[0]:.4f})")
    return best[1], best[2], best[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Fit Elo calibrated to MLS home_win.")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--home-adv", type=float, default=None)
    args = p.parse_args()

    if not MATCHES_CSV.exists():
        raise SystemExit(f"Missing {MATCHES_CSV}. Run: python scripts/prepare_data.py")

    df = load_matches()
    all_seasons = sorted(df["season"].astype(str).unique())
    train_seasons, test_seasons = train_test_season_sets(all_seasons)
    print(f"Matches: {len(df)}")
    print(
        f"Train: {sorted(train_seasons)[0]} … {sorted(train_seasons)[-1]} "
        f"({len(train_seasons)} seasons)"
    )
    print(f"Test: {sorted(test_seasons)}")

    if args.k is not None and args.home_adv is not None:
        best_k, best_h = float(args.k), float(args.home_adv)
        print(f"Using fixed params: K={best_k}, home_adv={best_h}")
    else:
        best_k, best_h, _ = grid_search(df, train_seasons)

    walked, final_state = walk_elo(df, k=best_k, home_adv=best_h)
    train = walked.loc[walked["season"].isin(train_seasons)].copy()
    test = walked.loc[walked["season"].isin(test_seasons)].copy()
    if test.empty:
        raise SystemExit(f"No test rows for {TEST_SEASONS}")

    calibrator = fit_calibrator(train)
    for split_name, split in (("TRAIN", train), ("TEST", test)):
        p_home = calibrator.predict_proba(split[["elo_diff"]].astype(float))[:, 1]
        y = split["home_win"].astype(int).to_numpy()
        p_mkt = split["p_home_fair"].to_numpy()
        print(f"\nProbability quality (home_win) — {split_name}:")
        print(
            f"  elo     acc={((p_home >= 0.5).astype(int) == y).mean():.4f}  "
            f"auc={roc_auc_score(y, p_home):.4f}  "
            f"logloss={log_loss(y, p_home):.4f}  "
            f"brier={brier_score_loss(y, p_home):.4f}"
        )
        print(
            f"  market  acc={((p_mkt >= 0.5).astype(int) == y).mean():.4f}  "
            f"auc={roc_auc_score(y, p_mkt):.4f}  "
            f"logloss={log_loss(y, p_mkt):.4f}  "
            f"brier={brier_score_loss(y, p_mkt):.4f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "k": best_k,
        "home_adv": best_h,
        "test_seasons": sorted(test_seasons),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "coef_elo_diff": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    joblib.dump(calibrator, MODELS_DIR / "home_elo_calibrator.pkl")
    joblib.dump(
        {"k": best_k, "home_adv": best_h, "ratings": dict(final_state.ratings)},
        MODELS_DIR / "home_elo_state.pkl",
    )
    with open(MODELS_DIR / "home_elo_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    walked[
        [
            "game_id",
            "game_date",
            "season",
            "home_id",
            "away_id",
            "home_name",
            "away_name",
            "elo_home",
            "elo_away",
            "elo_diff",
            "elo_e_home",
        ]
    ].to_csv(DATA_DIR / "mls_home_elo.csv", index=False)

    id_to_name: dict[int, str] = {}
    for _, row in df.iterrows():
        id_to_name[int(row["home_id"])] = str(row["home_name"])
        id_to_name[int(row["away_id"])] = str(row["away_name"])
    ratings_df = pd.DataFrame(
        [
            {"team_id": tid, "team": id_to_name.get(tid, str(tid)), "elo": rating}
            for tid, rating in final_state.ratings.items()
        ]
    ).sort_values("elo", ascending=False)
    ratings_df.to_csv(TABLES_DIR / "home_elo_ratings_final.csv", index=False)
    print(f"\nSaved home Elo artifacts to {MODELS_DIR}/ and {DATA_DIR / 'mls_home_elo.csv'}")
    print("\nTop ratings:")
    print(ratings_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
