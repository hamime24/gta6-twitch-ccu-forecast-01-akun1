# GTA VI Twitch CCU forecast — akun 1 (hamime24 / lively-sparrow-53), rebuild

Competition: gta6-twitch-ccu-forecast-01. Forecast Twitch concurrent viewers in
the Grand Theft Auto VI category, hours 2-24 of release day (2026-11-19 UTC),
as five quantiles per hour.

Model: per-hour recency-weighted OLS log regression with a sharp spread. For
each hour h, fit log(viewers_h) = a_h + b_h * log(viewers_1) across the
rolling-midnight comparables, each launch weighted by its release year (2018
counts 0.5, 2026 counts 1.5, linear in between); predict at GTA VI's hour-1
print. Spread = residual std scaled by 0.75, clamped to [0.20, 1.00].
Quantiles: q_tau(h) = exp(pred_h + z_tau * s_h).

This rebuild replaces the account's first entry (all-comparables median
quantiles, practice loss 0.293). Leave-one-out backtest over the 12 rolling
comparables: pinball loss 0.257 vs the house baseline's 0.357; the practice
board is the stronger signal here — the sharp-spread OLS sibling scored 0.157
against the plain OLS sibling's 0.162, and the recency-weighted center matched
plain OLS at 0.162.

Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
