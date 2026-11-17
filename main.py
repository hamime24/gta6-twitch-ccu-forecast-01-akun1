"""GTA VI launch-day Twitch CCU forecast, akun 1 (hamime24 / lively-sparrow-53).

Competition: gta6-twitch-ccu-forecast-01. The task: for each of hours 2-24 of
GTA VI's release day (2026-11-19 UTC), forecast five quantiles of the Twitch
category's concurrent viewers, given hour 1's print and the case's comparables.

Model (per-hour weighted OLS log regression, recency-weighted, sharp spread).
This is a rebuild of the account's first entry, which forecast the median
shape of all comparables and scored a practice loss of 0.293. The rebuild
fits, for each hour h, a least-squares line across the rolling-midnight
comparables: log(viewers_h) = a_h + b_h * log(viewers_1), each launch weighted
by how recent it is (a 2018 launch counts half, a 2026 launch counts one and
a half, rising linearly in between), and predicts at GTA VI's hour-1 print.
The spread is the residual standard deviation scaled by 0.75 before clamping
to [0.20, 1.00], so every hour's band is a quarter narrower in log terms than
the plain recency-weighted entry's.

Why this combination? Two practice signals point the same way. First, the
recency-weighted center matched the plain OLS center on the practice board
(loss 0.162 against 0.162), so weighting by year costs nothing and listens
more to recent launches, when Twitch's audience and launch marketing looked
most like 2026. Second, the sharp-spread OLS entry beat its plain sibling on
the practice board (0.157 against 0.162): the pinball loss rewards tight
intervals when the center is right, and a well-calibrated center does not
need the extra width. This entry stacks the two: the recency-weighted center
with the sharp spread. The leave-one-out backtest over the 12 rolling
comparables puts this variant at mean pinball loss 0.257 against the plain
OLS entry's 0.247 and the house baseline's 0.357, winning 5 of 12 cases —
the backtest rates sharpness lower than the practice board does, and this
entry bets the practice board's signal, which is graded on real launches,
over the backtest's pseudo-cases.

Quantiles: q_tau(h) = exp(pred_h + z_tau * s_h), with s_h the scaled, clamped
residual spread. Deterministic, standard library only, no network, no clock.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. It reads every <id>.in in the input directory and writes
<id>.out to the output directory: {"hours": [{"hour": h, "q": [q05, q25, q50,
q75, q95]}, ...]}, exactly 23 rows, hours 2-24 each once, each q five finite
numbers above 0 and at most 1e9, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The residual spread of an hour's fit, scaled by SPREAD_SCALE then clamped.
SPREAD_MIN = 0.20
SPREAD_MAX = 1.00
SPREAD_SCALE = 0.75
# The recency weight's endpoints: a launch from WEIGHT_YEAR_BASE counts
# WEIGHT_OLD, and the weight rises WEIGHT_PER_YEAR each year after.
WEIGHT_YEAR_BASE = 2018
WEIGHT_OLD = 0.5
WEIGHT_PER_YEAR = 0.125
# A comparable needs at least this many release-day hours to count.
MIN_HOURS = 20
# The baseline's spread, used as a fallback when no comparable qualifies.
BASELINE_SIGMA = 0.15


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp ("...Z") as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def unlock_kind(comparable: dict) -> str:
    """How the launch unlocked: 'rolling', 'global', or 'other'.

    GTA VI unlocks at midnight local, rolling country by country, so only
    'rolling' comparables share its daily shape. The wording comes from the
    kit's own unlock field.
    """
    text = (comparable.get("unlock") or "").lower()
    if "midnight local" in text or "rolling midnight" in text:
        return "rolling"
    if "global" in text or "simultaneous" in text or "single global" in text:
        return "global"
    return "other"


def recency_weight(year: int) -> float:
    """A launch's weight in the fit, rising linearly with its release year.

    A 2018 launch counts half; a 2026 launch counts one and a half. Recent
    launches say more about how GTA VI's day will go.
    """
    return WEIGHT_OLD + (year - WEIGHT_YEAR_BASE) * WEIGHT_PER_YEAR


def release_day_curve(comparable: dict) -> tuple[dict, int] | None:
    """The comparable's release-day hourly viewers and its release year.

    Returns ({hour 1..24: viewers}, year), or None when hour 1 is missing or
    zero, or fewer than MIN_HOURS of the release day's hours are present (e.g.
    the GTA VI pre-release comparable, which has no release-day data yet).
    """
    day_start = parse_time(comparable["day_start"])
    by_hour: dict[int, float] = {}
    for point in comparable.get("hourly", []):
        at = parse_time(point["from"])
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = point["viewers"]
    if by_hour.get(1, 0) <= 0 or len(by_hour) < MIN_HOURS:
        return None
    return by_hour, day_start.year


def wols_fit(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """The weighted least-squares line y = a + b*x through the points (xs, ys).

    With no weighted spread in x (every launch had the same hour-1 print), the
    slope is 0 and the intercept is the weighted mean of y: the best flat
    prediction.
    """
    sw = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / sw
    mean_y = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return mean_y, 0.0
    slope = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys)) / sxx
    return mean_y - slope * mean_x, slope


def fit_hourly(curves: list[tuple[dict, int]]) -> dict[int, tuple[float, float, float]]:
    """For each hour 2-24, the weighted OLS fit (intercept, slope, spread).

    Each hour's line is fit on log(viewers_h) against log(viewers_1) across
    the given (curve, year) pairs, each launch weighted by its recency. The
    spread is the fitted line's residual standard deviation, scaled by
    SPREAD_SCALE, then clamped to [SPREAD_MIN, SPREAD_MAX]. An hour fewer
    than 3 launches cover falls back to the previous hour's fit, or a flat
    line (intercept 0, slope 1) at hour 2.
    """
    log_p1 = [math.log(c[1]) for c, _ in curves]
    weights = [recency_weight(year) for _, year in curves]
    fits: dict[int, tuple[float, float, float]] = {}
    prev = (0.0, 1.0, SPREAD_MIN)
    for h in range(2, 25):
        xs: list[float] = []
        ys: list[float] = []
        ws: list[float] = []
        for (curve, _), x, w in zip(curves, log_p1, weights):
            if h in curve and curve[h] > 0:
                xs.append(x)
                ys.append(math.log(curve[h]))
                ws.append(w)
        if len(xs) < 3:
            fits[h] = prev
            continue
        intercept, slope = wols_fit(xs, ys, ws)
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        mean_resid = sum(residuals) / len(residuals)
        variance = sum((r - mean_resid) ** 2 for r in residuals) / len(residuals)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, math.sqrt(variance) * SPREAD_SCALE))
        fits[h] = (intercept, slope, spread)
        prev = fits[h]
    return fits


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_quantiles(p1: float, h: int, fits: dict[int, tuple[float, float, float]]) -> list[float | int]:
    """The five quantiles for hour h: the weighted OLS prediction plus a log-normal spread."""
    intercept, slope, spread = fits[h]
    center = math.exp(intercept + slope * math.log(p1))
    out = []
    for z in Z:
        q = center * math.exp(z * spread)
        # Every quantile must be a finite number above 0 and at most 1e9.
        q = max(0.01, min(1_000_000_000, q))
        out.append(round2(q))
    # Rounding can only move a quantile by a cent; keep them non-decreasing.
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # Only rolling-midnight comparables share GTA VI's daily shape; with
    # fewer than 5 of them, fall back to every usable comparable.
    rolling = []
    all_curves = []
    for comparable in case.get("comparables", []):
        item = release_day_curve(comparable)
        if item is None:
            continue
        all_curves.append(item)
        if unlock_kind(comparable) == "rolling":
            rolling.append(item)
    curves = rolling if len(rolling) >= 5 else all_curves
    if curves:
        fits = fit_hourly(curves)
    else:
        # No comparable at all: fall back to the house baseline's flat shape.
        fits = {h: (0.0, 1.0, BASELINE_SIGMA * math.sqrt(h - 1)) for h in range(2, 25)}

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], fits)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
