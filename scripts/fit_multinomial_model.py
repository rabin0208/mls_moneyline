"""
Fit multinomial logistic regression to predict 1X2 outcomes (H / D / A).

Train on all seasons before TEST_SEASONS; evaluate on the holdout seasons.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from model_utils import (
    FEATURE_COLS,
    TEST_SEASONS,
    print_feature_importance,
    season_train_test_masks,
    verify_test_set,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "mls_featured.csv"
TARGET_COL = "Res"
CLASS_ORDER = ["H", "D", "A"]


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    df["season_index"] = pd.to_numeric(df["season_index"], errors="coerce").fillna(0)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "p_home_fair", "p_draw_fair", "p_away_fair"])
    df = df.loc[df[TARGET_COL].isin(CLASS_ORDER)].copy()
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(str)
    season = df["season"].astype(str)
    return X, y, season


def season_split(
    X: pd.DataFrame, y: pd.Series, season: pd.Series, test_seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask, test_mask = season_train_test_masks(season, test_seasons)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def main() -> None:
    print(f"Loading {FEATURED_CSV}...")
    X, y, season = load_and_prepare()
    print(f"  Samples: {len(X)}, features: {len(FEATURE_COLS)}")

    X_train, X_test, y_train, y_test = season_split(X, y, season, TEST_SEASONS)
    verify_test_set(y_test, TEST_SEASONS)
    print(
        f"  Train: {len(y_train)} (seasons before {min(TEST_SEASONS)}), "
        f"test: {len(y_test)} ({TEST_SEASONS})"
    )
    for label in CLASS_ORDER:
        print(
            f"  Rate {label}: train={ (y_train == label).mean():.3f}  "
            f"test={(y_test == label).mean():.3f}"
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nFitting multinomial logistic regression (predict H/D/A)...")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Align class order for metrics / later eval
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    for c in CLASS_ORDER:
        if c not in class_to_idx:
            raise RuntimeError(f"Model missing class {c}; got {model.classes_}")

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    # reorder columns to H, D, A
    col_idx = [class_to_idx[c] for c in CLASS_ORDER]
    y_prob_ordered = y_prob[:, col_idx]
    y_true_idx = y_test.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_true_idx, y_prob_ordered, labels=[0, 1, 2])
    print(f"  Test 1X2 accuracy: {acc:.4f}")
    print(f"  Test log-loss:     {ll:.4f}")

    # Per-class coef magnitudes (mean abs across features for each class)
    # coef_ shape (n_classes, n_features) for multinomial
    print("\n  Mean |coef| by class:")
    for c in model.classes_:
        i = class_to_idx[c]
        print(f"    {c}: {np.abs(model.coef_[i]).mean():.4f}")

    # Feature importance for home class (most actionable)
    if "H" in class_to_idx:
        print_feature_importance(
            FEATURE_COLS,
            model.coef_[class_to_idx["H"]].tolist(),
            title="Home-class coefficients (abs = importance)",
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "multinomial_logistic.pkl")
    joblib.dump(scaler, MODELS_DIR / "multinomial_scaler.pkl")
    joblib.dump(CLASS_ORDER, MODELS_DIR / "multinomial_class_order.pkl")
    print(f"\nSaved model/scaler to {MODELS_DIR}/")

    # Save H-class coefficients for inspection
    coef = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coef_H": model.coef_[class_to_idx["H"]],
            "coef_D": model.coef_[class_to_idx["D"]],
            "coef_A": model.coef_[class_to_idx["A"]],
        }
    )
    coef["abs_coef_H"] = coef["coef_H"].abs()
    coef = coef.sort_values("abs_coef_H", ascending=False)
    coef_path = TABLES_DIR / "multinomial_coefficients.csv"
    coef.to_csv(coef_path, index=False)
    print(f"Saved coefficients to {coef_path}")


if __name__ == "__main__":
    main()
