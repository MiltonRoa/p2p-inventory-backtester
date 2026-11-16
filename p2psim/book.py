# -*- coding: utf-8 -*-
"""Synthetic P2P order-book snapshots.

The real study used one snapshot per minute of the public PYG/USDT book on
Binance P2P (about 113 days, 4.7 million rows).  This generator produces
snapshots with the same *shape* so the simulator and its tests do not depend
on proprietary data:

* a mid price that follows a slow random walk with occasional regime jumps,
* a bid/ask spread drawn around a configurable median (the real book had a
  median spread of about 1.0 % with a fat right tail at night),
* ``n_levels`` sellers above the mid and buyers below it, each level a few
  guaraníes apart, with per-level jitter,
* integer prices (the real book quotes PYG per USDT in whole units).

Returned structure (chronological):

    times : pandas.DatetimeIndex
    sells : list[np.ndarray]  ascending ask prices per snapshot
    buys  : list[np.ndarray]  ascending bid prices per snapshot
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BookParams:
    days: int = 30
    step_minutes: int = 1
    mid0: float = 6_200.0            # starting mid price (PYG per USDT)
    drift_bps_day: float = 0.0       # deterministic drift per day, basis points
    vol_bps_day: float = 40.0        # daily volatility of the mid, basis points
    jump_prob_day: float = 0.05      # probability of a regime jump per day
    jump_bps: float = 150.0          # typical jump size, basis points
    spread_median: float = 0.010     # median (ask - bid) / bid
    spread_sigma: float = 0.35       # log-normal dispersion of the spread
    n_levels: int = 25               # levels per side
    level_gap: float = 2.0           # mean price gap between consecutive levels (PYG)
    seed: int = 0


def synth_book(p: BookParams) -> Tuple[pd.DatetimeIndex, List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(p.seed)
    n = int(p.days * 24 * 60 / p.step_minutes)
    dt_day = p.step_minutes / (24 * 60)

    # Mid path: geometric random walk + Poisson jumps.
    rets = rng.normal(p.drift_bps_day * 1e-4 * dt_day,
                      p.vol_bps_day * 1e-4 * np.sqrt(dt_day), n)
    jumps = rng.random(n) < p.jump_prob_day * dt_day
    rets[jumps] += rng.normal(0, p.jump_bps * 1e-4, jumps.sum())
    mid = p.mid0 * np.exp(np.cumsum(rets))

    # Spread per snapshot, log-normal around the median, wider at "night"
    # (hours 2-8 UTC-3) to mimic a thin book.
    times = pd.date_range("2026-01-01", periods=n, freq=f"{p.step_minutes}min")
    hour_local = (times.hour - 3) % 24
    night = ((hour_local >= 2) & (hour_local < 8)).astype(float)
    spread = p.spread_median * np.exp(rng.normal(0, p.spread_sigma, n)) * (1 + 0.5 * night)

    sells: List[np.ndarray] = []
    buys: List[np.ndarray] = []
    k = np.arange(p.n_levels)
    for i in range(n):
        half = mid[i] * spread[i] / 2
        gaps_a = rng.exponential(p.level_gap, p.n_levels)
        gaps_b = rng.exponential(p.level_gap, p.n_levels)
        asks = np.floor(mid[i] + half + np.cumsum(gaps_a) - gaps_a[0] * 0.5 + 0 * k)
        bids = np.floor(mid[i] - half - np.cumsum(gaps_b) + gaps_b[0] * 0.5 + 0 * k)
        sells.append(np.sort(asks))
        buys.append(np.sort(bids))
    return times, sells, buys
