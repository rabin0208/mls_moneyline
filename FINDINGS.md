# MLS Moneyline — Findings & Notes

Summary of what worked, what didn’t, and how to use the best model in practice.
Holdout unless noted: **2025** (train through **2024**). Odds: football-data closing averages `AvgCH` / `AvgCD` / `AvgCA`.

Bayesian DC numbers below are from the **league-intercept** specification (MAP, `--map-only`), refit 2026-08-16. Pre-intercept 2025/2026 figures are called out where they still matter.

---

## Bottom line

1. **Reference price = closing market**, not any fitted model. When model and market disagree, the market has been more accurate.
2. **Best model so far = Bayesian Dixon–Coles** (hierarchical Poisson + intercept + home advantage). Closest to the market on log-loss; **not** a license to bet every closing edge.
3. **Unconditional calibration is OK; selective (edge) calibration is not.** Average P(H/D/A) matches the season. On *selected* edge bets, `P_model` is still ~**10pp too high**.
4. **Practical use:** treat Bayes DC as a **fair-price estimate**, compare to **early / soft book** odds, haircut large disagreements, skip draws, and track **CLV** — not short-run ROI alone.

---

## Model ladder (2025 holdout)

| Approach | Role | Rough result vs close |
|----------|------|------------------------|
| Closing fair 1X2 | **Reference** | Best log-loss / accuracy (1.024 / 49.8%) |
| Bayesian Dixon–Coles **(with intercept)** | Best *model* | Log-loss **1.050** vs market 1.024; acc 46.3%; edge=0 ROI **−16.9%** (538 bets) |
| Bayesian Dixon–Coles (no intercept, archived) | Identifiability bug | Log-loss 1.060; picked home **94%** of games; edge=0 looked ~flat (−0.5%) because it over-favored home |
| Classical Dixon–Coles / home Elo | Solid baselines | Still lose on edge filters (~−13% to −17% on older fav/dog evals) |
| Multinomial / home logistic on W–L lags | Weak | Edge ROI ~−17% to −20%+; far from market when they disagree |

Artifacts (current intercept model):

- `results/tables/bayesian_dc_edge_roi.csv`
- `results/tables/bayesian_dc_predictions_holdout.csv`
- `results/models/bayesian_dc_model.pkl` / `bayesian_dc_params.json`

Run:

```bash
python scripts/fit_bayesian_dc.py --map-only
python scripts/fit_bayesian_dc.py            # MAP + Laplace posterior
python scripts/predict_upcoming.py --refit --map-only
```

Old live pickles with `home_adv ≳ 0.4` and `intercept = 0` are treated as legacy and refit.

---

## League intercept (what changed)

The original Dixon–Coles mean had **no intercept** and sum-to-zero attack/defence:

```
λ = exp(attack_home − defence_away + home_adv)
μ = exp(attack_away − defence_home)
```

That forces geometric-mean away xG ≈ 1.0, so `home_adv` absorbed **all home scoring**, not just home advantage. Fitted `home_adv` was **0.48** (equal teams → λ/μ ≈ 1.62, P(home) ≈ 51%). Holdout: mean μ **1.06** vs actual away goals ~1.35; `model_pick` was home in **507 / 538** games (market 434).

**Fix:** add a league intercept γ (standard DC):

```
λ = exp(γ + attack_home − defence_away + δ)
μ = exp(γ + attack_away − defence_home)
```

γ ~ N(0.18, 0.25²), δ ~ N(0.25, 0.40²). Attack/defence still sum-to-zero.

**2025 holdout MAP:** γ = **0.22**, δ = **0.31**, ρ = −0.085.

| | Before (no γ) | After | Market / actual 2025 |
|--|--|--|--|
| Mean P(home / away) | 50% / 24% | **46% / 28%** | 47% / 28% (mkt); 44% / 30% (actual) |
| Mean λ / μ | 1.68 / **1.06** | 1.68 / **1.25** | ~1.66 / 1.35 actual |
| Home as `model_pick` | 94% | **87%** | 81% market |
| 1X2 log-loss | 1.060 | **1.050** | 1.024 |

Equal teams are no longer an automatic ~51% home favorite. The model still picks home often — MLS home teams really do win more — but away sides with a real strength gap can be favorites.

Classical `fit_dixon_coles.py` uses the same intercept parameterization.

---

## What Bayesian DC does

- Home/away goals ~ Poisson with Dixon–Coles low-score correlation ρ
- League intercept γ + home advantage δ (δ is *only* the extra home boost)
- Team attack/defence ~ hierarchical Gaussian priors (shrinkage)
- MAP via L-BFGS-B, optional **Laplace** posterior → average scoreline matrix → P(H), P(D), P(A)
- Edge-betting rule: best side among H/D/A (or H,A) vs de-vigged close, payout at `AvgC*`

This is the soccer analogue of “predict scores → derive moneyline probs,” with uncertainty.

---

## Calibration vs edge (important)

Profitable long-run betting needs:

1. **`P_model` calibrated** (say 45% → happens ~45%)
2. On selected bets, **calibrated P > fair book price** enough to **clear vig** (`hit > 1/odds`)
3. Enough volume for variance to settle

**Unconditional (all 538 games):** average P matches the season; binned reliability is close to the market (home ECE ~4.3% vs market ~3.4%). A ~50% home call hits ~47%; a ~65% home call hits ~64%. Fine as a fair-price sketch.

**Conditional on an edge (the slice you would bet):** still overconfident. `pick_bets` takes `argmax(P_model − P_fair)`. That subsample is where model and book disagree.

| Filter | n | Mean `P_model` | Hit rate | Mean fair market P | P − hit | ROI @ `AvgC*` |
|--------|---|----------------|----------|--------------------|---------|----------------|
| edge ≥ 0 | 538 | 0.39 | 0.29 | 0.32 | **+0.10** | **−16.9%** |
| edge ≥ 0.05 | 310 | 0.41 | 0.31 | 0.31 | **+0.11** | −9.9% |
| edge ≥ 0.08 | 184 | 0.44 | 0.33 | 0.32 | **+0.11** | −1.7% |
| edge ≥ 0.10 | 117 | 0.45 | 0.34 | 0.30 | **+0.11** | +8.5% |

Two things at once:

1. **Overconfidence did not go away.** Raising the threshold leaves `P_model` about **10pp above** the hit rate. Do not Kelly on raw `P_model`.
2. **Unlike the pre-intercept model, some of the edge may be real.** Hit rate now **rises** with the filter, and at ≥ 0.08 it is **above** fair market P. The old table was: claimed edge up, hit stuck on the market (~35%). That “fake edge” pattern is weaker now.

You still have to clear **vig**, not just beat de-vigged fair P. At closing odds ~3.4–3.6 you need ~28–29% hits to break even. Edge ≥ 0 sits on that line and **loses** because true hit rate is below `P_model`. Edge ≥ 0.10 at **+8.5% / 117 bets** is a small-sample hint, not a system.

### By side (edge ≥ 0.08)

| Side | n | Hit | ROI |
|------|---|-----|-----|
| Home | 96 | 40% | **−12%** |
| Draw | 9 | 22% | −17% |
| Away | 79 | 27% | **+12%** |

Draws are the worst part of the 1X2 edge rule. Home edges still lose vs close. Away is the only interesting slice, and even that is noisy. `H,A` only is better than H/D/A at every threshold (edge ≥ 0.08: **+2.5%** / 177 bets; edge ≥ 0.10: **+11.9%** / 117).

**Check before trusting a threshold:** among bets with edge ≥ X, is `mean(P_model) ≈ hit_rate`, and is `hit_rate` above `1/odds` after vig?

---

## Fade / “bet against the model”

On the intercept model, 89 / 538 games (17%) have `model_pick ≠ mkt_pick`. Betting the **market** pick on those was **+5.9% ROI** (2025). Same diagnostic as before: when they disagree, the close is still the better price on this holdout. That can flip season to season — treat it as a check, not a strategy.

---

## Why baseball looked easier than MLS

Compared to `Documents/semantic/baseball_moneyline`:

| | Baseball | MLS (this repo) |
|--|----------|-----------------|
| Outcomes | Binary home/away | 1X2 (~25% draws) |
| Features | ~114 (runs, rest, **pitchers**, …) | ~16 (W–L lags, H2H, season) |
| Sample | ~23k games | ~6k games |
| Model vs market | Nearly tied on log-loss/Brier | Market clearly sharper |
| Edge ROI @ close | Strong on 2026 eval | Negative at edge 0; noisy + at high edge / away-only |

Same modeling *style* does not transfer cleanly: MLS closing 1X2 is tougher, and the feature set here has no pitcher-like discrete signal (lineups / xG / absences).

Bayesian inference helps shrinkage and uncertainty; it does **not** add information the books already price.

---

## How to use the model in practice

### Intended workflow

1. Fit / refresh Bayesian DC on history through “yesterday” (`predict_upcoming.py --refit`).
2. For upcoming games, get **current** soft/early moneylines (not only historical closes).
3. De-vig → compare to `P_model`.
4. Haircut: a 10pp overconfidence gap on selected edges is roughly constant, so a 3% model edge is almost certainly fake; a 12% edge might be a couple of percent real after the cut. Optional: mix `0.7 P_model + 0.3 P_market` before comparing to `1/odds`.
5. Skip draws. If experimenting vs close at all, prefer **away** (maybe home) at a **high** threshold, not every 1X2 edge.
6. Log **CLV**: your odds vs closing line. Positive CLV over many bets ≈ positive EV even when win/loss is noisy.

### Soft books / enhanced odds

Unconditional P is close enough to use as a **rough fair price**. If you regularly get a few percent **better than the close**, that can turn +EV — with limits, caps, and one-shot promos as real constraints.

Do **not** assume “biggest model edge” is the goldmine; those spots are still the most overconfident. Do **not** bet every positive edge at the close (−17% ROI on 2025).

### What “close to the market” means here

- Good enough as a **rough fair price** for early line shopping (mean 1X2 now matches the close)
- **Not** proven sharper than the closing market on log-loss or unfiltered edges
- High-edge / away-only ROI vs close is **suggestive and underpowered**

---

## Suggested next improvements (information, not inference)

Higher leverage than another estimator on the same inputs:

- Goals for/against or xG form (not only W–L)
- Rest / travel
- Key absences / lineup strength
- Walk-forward mid-season refits
- Backtest vs **opening** lines + CLV, not only closes
- Shrink / mix `P_model` toward the market before betting; drop draws from the edge rule

---

## Quick command map

```bash
# data
python scripts/prepare_data.py
python scripts/build_features.py

# best current model
python scripts/fit_bayesian_dc.py --map-only
python scripts/predict_upcoming.py --refit --map-only

# other baselines
python scripts/fit_dixon_coles.py --static
python scripts/fit_multinomial_model.py && python scripts/eval_multinomial_edge.py
python scripts/eval_home_edge.py --model both
python scripts/eval_home_underdog.py
```

Pipeline details live in `README.md`.

---

## 2026 season-to-date check

Intercept model, MAP. Train through **2025**. Test: **269** games **2026-02-21 → 2026-08-08** (no June; World Cup break).

```bash
python scripts/fit_bayesian_dc.py --test-seasons 2026 --map-only
```

| | Bayesian DC (intercept) | Market |
|--|--|--|
| 1X2 accuracy | 47.6% | **50.2%** |
| 1X2 log-loss | 1.044 | **1.002** |
| MAP | γ = 0.26, δ = 0.25, ρ = −0.06 | — |
| Edge=0 ROI (H/D/A) | **−18.4%** (−49.6u / 269) | — |

Default rule = 1u on the best side among H/D/A every game (edge ≥ 0) at closing `AvgC*`.

| Filter | n | H/D/A | Hit | Mean P | Mean mkt P | ROI |
|--------|---|-------|-----|--------|------------|-----|
| edge ≥ 0 | 269 | 116/22/131 | 28.6% | 0.39 | 0.32 | **−18.4%** |
| edge ≥ 0.05 | 153 | 71/1/81 | 25.5% | 0.41 | 0.31 | −25.7% |
| edge ≥ 0.08 | 87 | 40/0/47 | 24.1% | 0.42 | 0.29 | −18.0% |
| edge ≥ 0.10 | 62 | 29/0/33 | 27.4% | 0.42 | 0.28 | −6.9% |

By side at edge ≥ 0: home −16.7% (116), draw −15.1% (22), away −20.5% (131). March–April were the hole (−20u and −25u); May/July roughly flat; August YTD −8.7u on 16 bets.

Unlike 2025, raising the edge threshold did **not** lift hit rate above the market. Same ~10pp overconfidence, plus the books were still sharper. Artifacts: `results/tables/bayesian_dc_edge_roi_2026.csv`, `bayesian_dc_predictions_holdout_2026.csv`.

**Takeaway:** intercept fixed the home-only pick bias (2026 `model_pick` 210 H / 59 A vs market 213 / 56), but closing-line edge betting is still a loser YTD. Do not lean on early/soft-book edges until this re-check looks better.
