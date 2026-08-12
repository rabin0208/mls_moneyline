# MLS Moneyline — Findings & Notes

Summary of what worked, what didn’t, and how to use the best model in practice.
Holdout unless noted: **2025** (train through **2024**). Odds: football-data closing averages `AvgCH` / `AvgCD` / `AvgCA`.

---

## Bottom line

1. **Reference price = closing market**, not any fitted model. When model and market disagree, the market has been more accurate.
2. **Best model so far = Bayesian Dixon–Coles** (hierarchical Poisson + Laplace posterior). Much closer to the market than form logistic / multinomial; roughly flat if you bet every game’s top edge at the close.
3. **“Large edge” is not enough.** Edge = `P_model − P_fair` is only useful if `P_model` is calibrated. On this holdout the Bayes model was **overconfident** on high-edge bets.
4. **Practical use:** treat Bayes DC as a **fair-price estimate**, compare to **early / soft book** odds, bet when the book is out of line, and track **CLV** (your price vs eventual close) plus calibration — not short-run ROI alone.

---

## Model ladder (2025 holdout)

| Approach | Role | Rough result vs close |
|----------|------|------------------------|
| Closing fair 1X2 | **Reference** | Best log-loss / accuracy |
| Bayesian Dixon–Coles | Best *model* | Log-loss ~1.060 vs market ~1.024; edge=0 ROI **≈ −0.5%** (538 bets) |
| Classical Dixon–Coles / home Elo | Solid baselines | Still lose on edge filters (~−13% to −17% on older fav/dog evals) |
| Multinomial / home logistic on W–L lags | Weak | Edge ROI ~−17% to −20%+; far from market when they disagree |

Artifacts:

- `results/tables/bayesian_dc_edge_roi.csv`
- `results/tables/bayesian_dc_predictions_holdout.csv`
- `results/models/bayesian_dc_model.pkl`

Run:

```bash
python scripts/fit_bayesian_dc.py
python scripts/fit_bayesian_dc.py --conf 0.7    # posterior P(edge>0) filter (did not help)
python scripts/fit_bayesian_dc.py --map-only   # shrinkage only, no Laplace
```

### What Bayesian DC does

- Home/away goals ~ Poisson with Dixon–Coles low-score correlation ρ  
- Team attack/defence ~ hierarchical Gaussian priors (shrinkage)  
- MAP via L-BFGS-B, then **Laplace** posterior → average scoreline matrix → P(H), P(D), P(A)  
- Same edge-betting rule as the multinomial pipeline (best side among H/D/A)

This is the soccer analogue of “predict scores → derive moneyline probs,” with uncertainty.

---

## Calibration vs edge (important)

Profitable long-run betting needs:

1. **`P_model` calibrated** (say 45% → happens ~45%)  
2. On selected bets, **calibrated P > fair book price** enough to clear vig  
3. Enough volume for variance to settle  

On Bayesian DC **2025**, high “edge” bets were mostly fake edge (overconfidence):

| Filter | Mean `P_model` | Actual hit rate | Mean fair market P |
|--------|----------------|-----------------|--------------------|
| edge ≥ 0 | 0.43 | 0.37 | 0.35 |
| edge ≥ 0.05 | 0.47 | 0.35 | 0.36 |
| edge ≥ 0.08 | 0.51 | 0.37 | 0.38 |
| edge ≥ 0.10 | 0.53 | 0.35 | 0.38 |

So raising the edge threshold made the *reported* edge larger while hit rate stayed ~market — the model was too high, not the books too low.

**Check before trusting a threshold:** among bets with edge ≥ X, is `mean(P_model) ≈ hit_rate`, and is `hit_rate` above the fair (or soft) price after vig?

---

## Fade / “bet against the model”

On multinomial 2025, betting the **market** on model–market disagreements looked good (+11% ROI). On a clean **2024** holdout (retrain through 2023), the same rule was **not** profitable (~−4%). Treat that as season noise / diagnostic, not a strategy.

Disagreement test is still useful: if fading the model toward the market helps, the model is the weaker price.

---

## Why baseball looked easier than MLS

Compared to `Documents/semantic/baseball_moneyline`:

| | Baseball | MLS (this repo) |
|--|----------|-----------------|
| Outcomes | Binary home/away | 1X2 (~25% draws) |
| Features | ~114 (runs, rest, **pitchers**, …) | ~16 (W–L lags, H2H, season) |
| Sample | ~23k games | ~6k games |
| Model vs market | Nearly tied on log-loss/Brier | Market clearly sharper |
| Edge ROI @ close | Strong on 2026 eval | Flat at best (Bayes), negative otherwise |

Same modeling *style* does not transfer cleanly: MLS closing 1X2 is tougher, and the feature set here has no pitcher-like discrete signal (lineups / xG / absences).

Bayesian inference helps shrinkage and uncertainty; it does **not** add information the books already price.

---

## How to use the model in practice

### Intended workflow

1. Fit / refresh Bayesian DC on history through “yesterday.”  
2. For upcoming games, get **current** soft/early moneylines (not only historical closes).  
3. De-vig → compare to `P_model`.  
4. Bet when the book is meaningfully worse than the model **and** the edge survives a sanity check (not only huge disagreement).  
5. Log **CLV**: your odds vs closing line. Positive CLV over many bets ≈ positive EV even when win/loss is noisy.

### Soft books / enhanced odds

Bayes DC was ~**flat vs closing averages** at edge 0. If you regularly get prices a few percent **better** than that close (soft books, boosts), the *same* pricing view can turn +EV — with limits, caps, and one-shot promos as real constraints.

Do **not** assume “biggest model edge” is the soft-book goldmine; those spots were the most overconfident on 2025.

### What “close to the market” means here

- Good enough as a **rough fair price** for early line shopping  
- **Not** proven sharper than the closing market on log-loss or selective edges  

---

## Suggested next improvements (information, not inference)

Higher leverage than another estimator on the same inputs:

- Goals for/against or xG form (not only W–L)  
- Rest / travel  
- Key absences / lineup strength  
- Walk-forward mid-season refits  
- Backtest vs **opening** lines + CLV, not only closes  

---

## Quick command map

```bash
# data
python scripts/prepare_data.py
python scripts/build_features.py

# best current model
python scripts/fit_bayesian_dc.py

# other baselines
python scripts/fit_dixon_coles.py --static
python scripts/fit_multinomial_model.py && python scripts/eval_multinomial_edge.py
python scripts/eval_home_edge.py --model both
python scripts/eval_home_underdog.py
```

Pipeline details live in `README.md`.

---

## 2026 season-to-date check

Data refreshed from football-data through **2026-08-08** (**269** completed games with closing odds).

Train: all seasons through **2025**. Test: **2026** YTD.

```bash
# refresh raw file then rebuild
curl -sL -A "Mozilla/5.0" "https://www.football-data.co.uk/new/USA.csv" -o data/USA.csv
python scripts/prepare_data.py
python scripts/fit_bayesian_dc.py --test-seasons 2026
```

| | Bayesian DC | Market |
|--|--|--|
| 1X2 accuracy | 48.3% | **50.2%** |
| 1X2 log-loss | 1.046 | **1.002** |
| Edge=0 ROI (H/D/A) | **−11.9%** (−31.9u / 269) | — |

So 2026 YTD is **worse** than the near-flat 2025 holdout (−0.5%). Same pattern: market still sharper; model still overconfident on selected edges (e.g. edge≥0.05: mean `P_model` ≈ 0.49 vs hit rate ≈ 0.36).

Artifacts: `results/tables/bayesian_dc_edge_roi_2026.csv`, `bayesian_dc_predictions_holdout_2026.csv`.

**Takeaway:** “close to market” on 2025 did not fully carry into 2026 so far. Re-check after more games / mid-season refits before leaning on early soft-book edges.
