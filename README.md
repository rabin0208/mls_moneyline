# MLS Moneyline

ML project focused on **P(home win)** vs closing market odds, with selective betting
(especially home underdogs).

Uses [football-data.co.uk](https://www.football-data.co.uk/usa.php) USA CSV (`data/USA.csv`).

**Results & how to use the models in practice:** see [FINDINGS.md](FINDINGS.md)
(Bayesian Dixon–Coles with league intercept; calibration vs edge; soft/early lines + CLV).

## Setup

```bash
conda env create -f environment.yml
conda activate mls_moneyline
```

Or reuse the Liga MX env:

```bash
conda activate liga_mx_moneyline
```

Run all commands from the project root.

Holdout season: `2025` (train on all seasons through `2024`).

## Primary pipeline (home win)

### 1. Prepare matches + market columns
```bash
python scripts/prepare_data.py
```
Writes `data/mls_matches.csv`.

### 2. Feature engineering
```bash
python scripts/build_features.py
```
Lagged wins (last 5 home/away), H2H, season → `data/mls_featured.csv`.

### 3. Fit home models
```bash
python scripts/fit_home_model.py   # logistic on form features → P(home)
python scripts/fit_home_elo.py     # Elo → calibrated P(home)
```

### 4. Model vs market (home edge bets)
```bash
python scripts/eval_home_edge.py
python scripts/eval_home_edge.py --model both --filter all,home_dog
```
Bets home at `AvgCH` (vig included) when `P_model(home) - p_home_fair ≥ edge`.
Filters: `all`, `home_dog` (`AvgCH > AvgCA`), `home_fav`.

### 5. Multinomial 1X2 model
```bash
python scripts/fit_multinomial_model.py
python scripts/eval_multinomial_edge.py
python scripts/eval_multinomial_edge.py --sides H,A
```
Predicts P(H), P(D), P(A). Bets the side with largest edge vs fair closing probs
when edge ≥ threshold (payout at AvgCH / AvgCD / AvgCA).

### 6. Bayesian Dixon–Coles (hierarchical Poisson)
```bash
python scripts/fit_bayesian_dc.py
python scripts/fit_bayesian_dc.py --test-seasons 2026   # current season YTD
python scripts/fit_bayesian_dc.py --conf 0.7          # only bet when P(edge>0) ≥ 0.7
python scripts/fit_bayesian_dc.py --map-only           # shrinkage MAP, no Laplace
```
Fits attack/defence with Gaussian shrinkage priors plus a **league intercept**
so home advantage is only the extra home boost (not the scoring rate). Optional
Laplace posterior at the MAP; averages scoreline probabilities into 1X2.
Writes `results/tables/bayesian_dc_edge_roi.csv` (and `*_2026.csv` for other holdouts).
See FINDINGS.md for the intercept fix and edge-bet calibration.

### 7. Baselines
```bash
python scripts/eval_home_underdog.py          # home dogs @ AvgCH
python scripts/eval_home_underdog.py --fair   # home dogs @ de-vigged odds
```

## Legacy two-way market (favorite / underdog DC)

Favorite/underdog DC backtests (includes classical Dixon–Coles):

```bash
python scripts/fit_logistic_model.py    # predicts fav_won
python scripts/eval_vs_market.py
python scripts/fit_elo_model.py
python scripts/fit_dixon_coles.py --static
python scripts/eval_underdog_dc.py --side favorite
python scripts/eval_underdog_dc.py --side underdog_dc
```
