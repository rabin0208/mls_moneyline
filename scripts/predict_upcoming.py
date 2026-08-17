"""
Score upcoming MLS games with Bayesian Dixon-Coles (fair 1X2).

Fits on all completed matches in data/mls_matches.csv (warm-starts from a
saved model when possible), pulls the next ESPN slate + DraftKings 1X2, and
writes model probabilities / fair odds vs the current book.

Usage:
  python scripts/predict_upcoming.py
  python scripts/predict_upcoming.py --dates 20260815,20260816,20260817
  python scripts/predict_upcoming.py --refit
  python scripts/predict_upcoming.py --map-only
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from bayesian_dc_utils import (
    DEFAULT_N_POSTERIOR,
    DEFAULT_SIGMA_ATT,
    DEFAULT_SIGMA_DEF,
    DEFAULT_XI,
    BayesianDCModel,
)
from dixon_coles_utils import is_legacy_dc_model
from fit_bayesian_dc import fit_on_history, load_matches
from odds_utils import implied_prob, remove_vig_3way

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LIVE_MODEL = MODELS_DIR / "bayesian_dc_model_live.pkl"

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard"
ESPN_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.espn.com/soccer/",
}

# ESPN / common names → football-data USA.csv names
NAME_ALIASES = {
    "Atlanta United": "Atlanta Utd",
    "Atlanta United FC": "Atlanta Utd",
    "CF Montréal": "CF Montreal",
    "Charlotte FC": "Charlotte",
    "Chicago Fire FC": "Chicago Fire",
    "D.C. United": "DC United",
    "Houston Dynamo FC": "Houston Dynamo",
    "Inter Miami CF": "Inter Miami",
    "LA Galaxy": "Los Angeles Galaxy",
    "LAFC": "Los Angeles FC",
    "Minnesota United FC": "Minnesota United",
    "New York City FC": "New York City",
    "NYCFC": "New York City",
    "Orlando City SC": "Orlando City",
    "Red Bull New York": "New York Red Bulls",
    "Seattle Sounders FC": "Seattle Sounders",
    "St. Louis CITY SC": "St. Louis City",
    "St. Louis City SC": "St. Louis City",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps",
}


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_name(s: str) -> str:
    s = _strip_accents(s).lower()
    s = s.replace("&", "and").replace(".", " ")
    s = re.sub(r"\b(fc|sc|cf)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def map_team(name: str, known: set[str]) -> str:
    if name in known:
        return name
    if name in NAME_ALIASES and NAME_ALIASES[name] in known:
        return NAME_ALIASES[name]
    target = _norm_name(name)
    hits = [t for t in known if _norm_name(t) == target]
    if len(hits) == 1:
        return hits[0]
    # looser: token containment
    hits = [t for t in known if target and (target in _norm_name(t) or _norm_name(t) in target)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"Cannot map team {name!r} to football-data names")


def american_to_decimal(odds) -> float:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return float("nan")
    s = str(odds).strip().upper().replace("−", "-")
    if s in {"EVEN", "EV", "PK"}:
        return 2.0
    n = float(s)
    if n > 0:
        return 1.0 + n / 100.0
    if n < 0:
        return 1.0 + 100.0 / abs(n)
    return float("nan")


def _ml_american(block: dict | None, side: str) -> str | None:
    if not block:
        return None
    side_block = block.get(side) or {}
    for key in ("close", "open"):
        odds = (side_block.get(key) or {}).get("odds")
        if odds:
            return str(odds)
    return None


def _espn_events(dates: list[str]) -> list[dict]:
    """Fetch ESPN events, trying per-day then a compact date range."""
    payloads = []
    for d in dates:
        url = f"{ESPN_SCOREBOARD}?dates={d}&limit=50"
        r = requests.get(url, headers=ESPN_UA, timeout=30)
        if r.status_code == 403:
            break
        r.raise_for_status()
        payloads.append(r.json())
    if not payloads and len(dates) >= 1:
        lo, hi = min(dates), max(dates)
        url = f"{ESPN_SCOREBOARD}?dates={lo}-{hi}&limit=50"
        r = requests.get(url, headers=ESPN_UA, timeout=30)
        r.raise_for_status()
        payloads.append(r.json())
    events = []
    seen = set()
    for payload in payloads:
        for event in payload.get("events") or []:
            eid = event.get("id")
            if eid in seen:
                continue
            seen.add(eid)
            events.append(event)
    return events


def load_fixtures_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df.sort_values("kickoff_utc").reset_index(drop=True)


def fetch_espn_slate(dates: list[str]) -> pd.DataFrame:
    """Pull ESPN scoreboard events (and DraftKings 1X2 when present)."""
    rows = []
    for event in _espn_events(dates):
        comps = (event.get("competitions") or [{}])[0]
        home = away = None
        for c in comps.get("competitors") or []:
            display = (c.get("team") or {}).get("displayName")
            if c.get("homeAway") == "home":
                home = display
            else:
                away = display
        if not home or not away:
            continue
        kick = pd.to_datetime(event.get("date"), utc=True)
        dk_h = dk_d = dk_a = np.nan
        book = None
        odds_list = comps.get("odds") or []
        if odds_list:
            o0 = odds_list[0]
            book = (o0.get("provider") or {}).get("name")
            ml = o0.get("moneyline") or {}
            ah = _ml_american(ml, "home")
            ad = _ml_american(ml, "draw")
            aa = _ml_american(ml, "away")
            if ah is None and o0.get("homeTeamOdds"):
                ah = (o0.get("homeTeamOdds") or {}).get("moneyLine")
            if aa is None and o0.get("awayTeamOdds"):
                aa = (o0.get("awayTeamOdds") or {}).get("moneyLine")
            if ad is None and o0.get("drawOdds"):
                ad = (o0.get("drawOdds") or {}).get("moneyLine")
            dk_h = american_to_decimal(ah)
            dk_d = american_to_decimal(ad)
            dk_a = american_to_decimal(aa)
        rows.append(
            {
                "espn_id": event.get("id"),
                "kickoff_utc": kick,
                "espn_home": home,
                "espn_away": away,
                "book": book,
                "odds_H": dk_h,
                "odds_D": dk_d,
                "odds_A": dk_a,
            }
        )
    if not rows:
        raise SystemExit(f"No ESPN MLS events for dates {dates}")
    out = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)
    return out


def load_or_fit(
    hist: pd.DataFrame,
    *,
    refit: bool,
    map_only: bool,
    n_posterior: int,
    xi: float,
    sigma_att: float,
    sigma_def: float,
) -> BayesianDCModel:
    if (not refit) and LIVE_MODEL.exists():
        try:
            saved = joblib.load(LIVE_MODEL)
            if not is_legacy_dc_model(saved):
                print(f"Using saved live model ({LIVE_MODEL.name}). Pass --refit to rebuild.")
                return saved
            print("Saved live model has no league intercept; refitting.")
        except Exception as exc:
            print(f"Could not load {LIVE_MODEL.name}: {exc}")

    prev = None
    for path in (LIVE_MODEL, MODELS_DIR / "bayesian_dc_model_2026.pkl", MODELS_DIR / "bayesian_dc_model.pkl"):
        if path.exists():
            try:
                prev = joblib.load(path)
                print(f"Warm-start from {path.name}")
                break
            except Exception as exc:
                print(f"Could not load {path.name}: {exc}")

    print(f"Fitting Bayesian DC on {len(hist)} completed matches (through {hist['game_date'].max().date()})...")
    t0 = time.time()
    model = fit_on_history(
        hist,
        xi=xi,
        sigma_att=sigma_att,
        sigma_def=sigma_def,
        n_posterior=n_posterior,
        compute_laplace=not map_only,
        prev=prev,
    )
    print(
        f"  Done in {time.time() - t0:.1f}s  "
        f"intercept={model.intercept:.3f}  home_adv={model.home_adv:.3f}  "
        f"rho={model.rho:.3f}  teams={len(model.teams)}"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LIVE_MODEL)
    print(f"  Saved {LIVE_MODEL}")
    return model


def score_slate(slate: pd.DataFrame, model: BayesianDCModel, known: set[str], *, use_posterior: bool) -> pd.DataFrame:
    rows = []
    for _, row in slate.iterrows():
        home = map_team(str(row["espn_home"]), known)
        away = map_team(str(row["espn_away"]), known)
        pred = model.predict_match(home, away, use_posterior=use_posterior)
        p_h, p_d, p_a = pred["p_home"], pred["p_draw"], pred["p_away"]
        fair_h, fair_d, fair_a = 1.0 / p_h, 1.0 / p_d, 1.0 / p_a

        oh, od, oa = float(row["odds_H"]), float(row["odds_D"]), float(row["odds_A"])
        has_book = np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)
        if has_book:
            mkt = remove_vig_3way(implied_prob(oh), implied_prob(od), implied_prob(oa))
        else:
            mkt = (float("nan"), float("nan"), float("nan"))

        edges = {
            "H": p_h - mkt[0],
            "D": p_d - mkt[1],
            "A": p_a - mkt[2],
        }
        if has_book:
            best = max(edges, key=edges.get)
            best_edge = edges[best]
        else:
            best, best_edge = None, float("nan")

        model_pick = max(
            [("H", p_h), ("D", p_d), ("A", p_a)],
            key=lambda t: t[1],
        )[0]
        kick_et = row["kickoff_utc"].tz_convert("America/New_York")
        rows.append(
            {
                "kickoff_et": kick_et.strftime("%Y-%m-%d %H:%M"),
                "home_name": home,
                "away_name": away,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "p_home_lo": pred.get("p_home_lo"),
                "p_home_hi": pred.get("p_home_hi"),
                "lam": pred.get("lam"),
                "mu": pred.get("mu"),
                "fair_odds_H": fair_h,
                "fair_odds_D": fair_d,
                "fair_odds_A": fair_a,
                "book": row.get("book"),
                "odds_H": oh,
                "odds_D": od,
                "odds_A": oa,
                "p_home_mkt": mkt[0],
                "p_draw_mkt": mkt[1],
                "p_away_mkt": mkt[2],
                "edge_H": edges["H"],
                "edge_D": edges["D"],
                "edge_A": edges["A"],
                "best_side": best,
                "best_edge": best_edge,
                "model_pick": model_pick,
            }
        )
    return pd.DataFrame(rows)


def _fmt_odds(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "  n/a"
    return f"{x:5.2f}"


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "  n/a"
    return f"{100 * x:4.1f}%"


def print_table(df: pd.DataFrame) -> None:
    print("\nUpcoming MLS — Bayesian Dixon–Coles fair 1X2")
    print("P = model probability. Fair = 1/P (no vig). Book = DraftKings current (when ESPN has it).")
    print("Edge = P_model − P_fair_book. Large edges were overconfident on 2025; shop, don't auto-bet.\n")
    header = (
        f"{'Kickoff ET':<16} {'Match':<42} "
        f"{'P(H)':>6} {'P(D)':>6} {'P(A)':>6}  "
        f"{'fair H/D/A':>16}  {'book H/D/A':>16}  "
        f"{'xG':>7} {'pick':>4} {'edge':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        match = f"{r['home_name']} vs {r['away_name']}"
        fair = f"{_fmt_odds(r['fair_odds_H'])}/{_fmt_odds(r['fair_odds_D'])}/{_fmt_odds(r['fair_odds_A'])}"
        book = f"{_fmt_odds(r['odds_H'])}/{_fmt_odds(r['odds_D'])}/{_fmt_odds(r['odds_A'])}"
        xg = f"{r['lam']:.1f}-{r['mu']:.1f}" if np.isfinite(r["lam"]) else "  n/a"
        if r["best_side"] and np.isfinite(r["best_edge"]):
            edge = f"{r['best_side']} {_fmt_pct(r['best_edge'])}"
        else:
            edge = "n/a"
        print(
            f"{r['kickoff_et']:<16} {match:<42} "
            f"{_fmt_pct(r['p_home']):>6} {_fmt_pct(r['p_draw']):>6} {_fmt_pct(r['p_away']):>6}  "
            f"{fair:>16}  {book:>16}  "
            f"{xg:>7} {r['model_pick']:>4} {edge:>10}"
        )


def default_dates(n_days: int = 4) -> list[str]:
    start = datetime.now(timezone.utc).date()
    return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]


def main() -> None:
    p = argparse.ArgumentParser(description="Predict upcoming MLS games with Bayesian DC.")
    p.add_argument(
        "--dates",
        type=str,
        default="",
        help="Comma-separated ESPN dates YYYYMMDD (default: today UTC through +3 days).",
    )
    p.add_argument(
        "--fixtures",
        type=str,
        default="",
        help="CSV with espn_home/espn_away/kickoff_utc and optional odds_H/D/A (skips ESPN).",
    )
    p.add_argument("--refit", action="store_true", help="Refit on all completed matches.")
    p.add_argument("--map-only", action="store_true", help="MAP only (skip Laplace posterior).")
    p.add_argument("--n-posterior", type=int, default=min(120, DEFAULT_N_POSTERIOR))
    p.add_argument("--xi", type=float, default=DEFAULT_XI)
    p.add_argument("--sigma-att", type=float, default=DEFAULT_SIGMA_ATT)
    p.add_argument("--sigma-def", type=float, default=DEFAULT_SIGMA_DEF)
    args = p.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()] or default_dates()
    hist = load_matches()
    known = set(hist["home_name"]) | set(hist["away_name"])
    print(f"History: {len(hist)} matches through {hist['game_date'].max().date()}")

    fixtures_path = Path(args.fixtures) if args.fixtures else PROJECT_ROOT / "data" / "upcoming_fixtures.csv"
    slate = None
    if args.fixtures:
        print(f"Loading fixtures from {fixtures_path}...")
        slate = load_fixtures_csv(fixtures_path)
    else:
        print(f"Fetching ESPN slate for {', '.join(dates)}...")
        try:
            slate = fetch_espn_slate(dates)
        except Exception as exc:
            if fixtures_path.exists():
                print(f"  ESPN unavailable ({exc}). Falling back to {fixtures_path}.")
                slate = load_fixtures_csv(fixtures_path)
            else:
                raise
    print(f"  {len(slate)} upcoming games")

    model = load_or_fit(
        hist,
        refit=args.refit,
        map_only=args.map_only,
        n_posterior=args.n_posterior,
        xi=args.xi,
        sigma_att=args.sigma_att,
        sigma_def=args.sigma_def,
    )

    scored = score_slate(slate, model, known, use_posterior=not args.map_only)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "bayesian_dc_upcoming.csv"
    scored.to_csv(out_path, index=False, float_format="%.6f")
    print_table(scored)
    print(f"\nWrote {out_path}")
    print(
        "Use: compare book odds to fair odds. Bet only when a book is worse than the "
        "model *and* you would still like the price vs the eventual close (CLV)."
    )


if __name__ == "__main__":
    main()
