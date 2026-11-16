# -*- coding: utf-8 -*-
"""Desk model: pricing rules, queue-position fill hazards, inventory caps.

Pricing (same rules as the live repricer)
-----------------------------------------
* SELL ad:  ``price = max(best_ask - 1, floor)`` where
  ``floor = int(avg_cost * (1 + fee + margin)) + 1``.  If that price would
  leave the ad outside the top ``top_pos`` of the book, it is re-priced one
  unit below the ``top_pos``-th ask but never below ``break_even``
  (``avg_cost * (1 + fee + min_margin)``).
* BUY ad:   ``price = min(best_bid + 1, ceiling)`` where
  ``ceiling = int(max(best_ask, median_24h(best_ask)) * (1 - 2 fee - margin))``
  so a purchase can always be resold at the required margin.

Fills
-----
Fills arrive as a Poisson process whose intensity depends on the ad's queue
position (rank 1 is the best price).  The hazards below were measured on real
snapshots (orders per hour at each rank) and are scaled by ``calibration``.
An ad priced more than 0.3 % away from the fair side of the book gets the
"off-market" hazard.  Order sizes are sampled from a log-normal roughly
matching the empirical ticket distribution (median about 500 USDT).

Inventory cap ("tope 90/70/40")
-------------------------------
The desk looks at where the current best bid sits inside its rolling 72 h
range.  In the lower third it may hold up to ``caps[0]`` of total capital as
inventory, in the middle third ``caps[1]``, in the upper third ``caps[2]``.
This is the rule that stopped the real desk from filling up at local tops.

Accounting identity (tested): ``cash + cost_basis == initial_cash + realised_pnl``
at every step.  Mark-to-market of the remaining inventory is reported
separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Orders per hour by queue rank, measured on the live book (2026-09-25).
HZ_BUY = {1: 0.128, 2: 0.092, 3: 0.092, 4: 0.048, 5: 0.048, 6: 0.048}
HZ_SELL = {1: 0.095, 2: 0.064, 3: 0.064, 4: 0.014, 5: 0.014, 6: 0.014}


def hazard(rank: int, side: str) -> float:
    tab = HZ_BUY if side == "BUY" else HZ_SELL
    if rank in tab:
        return tab[rank]
    if rank <= 12:
        return 0.044 if side == "BUY" else 0.019
    if rank <= 20:
        return 0.020 if side == "BUY" else 0.010
    return 0.007 if side == "BUY" else 0.004


@dataclass(frozen=True)
class DeskParams:
    initial_cash: float = 40_000_000.0       # PYG
    fee: float = 0.0014                      # per side (Silver merchant tier)
    margin: float = 0.0020                   # target net margin per round trip
    min_margin: float = 0.0010               # never sell below cost*(1+fee+min_margin)
    top_pos: int = 3                         # keep the SELL ad inside the top-N
    caps: Optional[Tuple[float, float, float]] = (0.9, 0.7, 0.4)   # None = no cap
    min_ticket_pyg: float = 300_000.0        # exchange minimum per order
    min_inventory_to_sell: float = 50.0      # USDT
    size_median_usdt: float = 500.0
    size_sigma: float = 0.9
    calibration: float = 1.0                 # hazard multiplier
    spread_gate: Optional[float] = None      # do not BUY while spread < gate (tested, rejected)


@dataclass
class DeskResult:
    realised_pnl: float
    mtm: float
    cash: float
    inventory: float
    avg_cost: float
    buys: int
    sells: int
    stuck_hours: float          # SELL ad parked above best ask because of the floor
    underwater_hours: float     # stuck while inventory >= 50 % of capital
    no_stock_hours: float       # cash idle with (almost) no inventory
    max_inventory_share: float  # max inventory value / capital seen at a buy decision
    daily_pnl: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.realised_pnl + self.mtm


def _prep(times: pd.DatetimeIndex, sells: Sequence[np.ndarray], buys: Sequence[np.ndarray]):
    """Derive best quotes, fair references and rolling stats; mark valid snapshots."""
    va = np.array([s[0] if s is not None and len(s) >= 5 else np.nan for s in sells])
    vb = np.array([b[-1] if b is not None and len(b) >= 5 else np.nan for b in buys])
    sp = (va - vb) / vb
    valid = ~np.isnan(sp) & (sp >= 0) & (sp <= 0.025)
    va = np.where(valid, va, np.nan)
    vb = np.where(valid, vb, np.nan)
    fair_s = np.array([np.median(s[:5]) if s is not None and len(s) >= 5 else np.nan for s in sells])
    fair_b = np.array([np.median(b[-5:]) if b is not None and len(b) >= 5 else np.nan for b in buys])
    va_s = pd.Series(va).ffill()
    vb_s = pd.Series(vb).ffill()
    step_min = max(int(round((times[1] - times[0]).total_seconds() / 60)), 1) if len(times) > 1 else 1
    w24 = max(1440 // step_min, 2)
    w72 = max(4320 // step_min, 2)
    va_med24 = va_s.rolling(w24, min_periods=max(w24 // 24, 1)).median().values
    vb_min72 = vb_s.rolling(w72, min_periods=max(w72 // 36, 1)).min().values
    vb_max72 = vb_s.rolling(w72, min_periods=max(w72 // 36, 1)).max().values
    dt_h = np.r_[step_min / 60, np.diff(times.values).astype("timedelta64[s]").astype(float) / 3600]
    dt_h = np.clip(dt_h, 0, 0.25)
    return va_s.values, vb_s.values, valid, fair_s, fair_b, va_med24, vb_min72, vb_max72, dt_h


def simulate(times: pd.DatetimeIndex, sells: Sequence[np.ndarray], buys: Sequence[np.ndarray],
             p: DeskParams, seed: int = 0) -> DeskResult:
    rng = np.random.default_rng(seed)
    va, vb, valid, fair_s, fair_b, va_med24, vb_min72, vb_max72, dt_h = _prep(times, sells, buys)
    n = len(times)

    cash = p.initial_cash
    inv = 0.0        # USDT held
    basis = 0.0      # PYG cost of the inventory
    pnl = 0.0
    n_buy = n_sell = 0
    stuck = under = nostock = 0.0
    max_share = 0.0
    daily: Dict[str, float] = {}

    def draw_size() -> float:
        return float(np.exp(np.log(p.size_median_usdt) + p.size_sigma * rng.normal()))

    for i in range(n):
        if not valid[i] or np.isnan(va[i]) or np.isnan(vb[i]):
            continue
        t = times[i]
        day = str(t.date())
        f, mg = p.fee, p.margin
        cm = basis / inv if inv > 0 else 0.0
        ref = max(va[i], va_med24[i]) if not np.isnan(va_med24[i]) else va[i]
        ceiling = int(ref * (1 - 2 * f - mg))
        capital = cash + inv * vb[i]
        spread = (va[i] - vb[i]) / vb[i]

        # ---- SELL ad ----------------------------------------------------
        if inv >= p.min_inventory_to_sell:
            floor = int(cm * (1 + f + mg)) + 1
            be = int(cm * (1 + f + p.min_margin)) + 1
            px_s = max(va[i] - 1, floor)
            if px_s > va[i] - 1:
                rank_full = 1 + int(np.searchsorted(sells[i], px_s, side="right"))
                if rank_full > p.top_pos:
                    r_top = sells[i][p.top_pos - 1] if len(sells[i]) >= p.top_pos else va[i]
                    px_s = max(int(r_top) - 1, be)
            rank_s = 1 + int(np.searchsorted(sells[i], px_s, side="right"))
            h_s = hazard(rank_s, "SELL") if px_s <= fair_s[i] * 1.003 else hazard(99, "SELL")
            if rng.random() < 1 - np.exp(-h_s * p.calibration * dt_h[i]):
                q = min(draw_size(), inv)
                if q * px_s >= p.min_ticket_pyg:
                    tot = q * px_s
                    out = min(q + q * f, inv)          # fee paid in USDT
                    g = tot - out * cm
                    pnl += g
                    basis -= out * cm
                    inv -= out
                    cash += tot
                    n_sell += 1
                    daily[day] = daily.get(day, 0.0) + g
            if px_s > va[i] - 1 and inv >= 1000:
                stuck += dt_h[i]
                if inv * vb[i] >= 0.5 * capital:
                    under += dt_h[i]
        if inv < 100 and cash > 1_000_000:
            nostock += dt_h[i]

        # ---- BUY ad -----------------------------------------------------
        q_max = cash * 0.95
        if p.caps is not None:
            rng72 = vb_max72[i] - vb_min72[i]
            pos72 = (vb[i] - vb_min72[i]) / rng72 if rng72 > 0 and not np.isnan(rng72) else 0.5
            c = p.caps[0] if pos72 < 1 / 3 else (p.caps[1] if pos72 < 2 / 3 else p.caps[2])
            q_max = min(q_max, max(c * capital - inv * vb[i], 0.0))
        if p.spread_gate is not None and spread < p.spread_gate:
            q_max = 0.0
        if q_max >= p.min_ticket_pyg:
            if vb[i] > ceiling:
                reach = buys[i][buys[i] <= ceiling - 1]
                px_b = min(reach[-1] + 1, ceiling) if len(reach) else ceiling
            else:
                px_b = min(vb[i] + 1, ceiling)
            rank_b = 1 + int(len(buys[i]) - np.searchsorted(buys[i], px_b, side="left"))
            h_b = hazard(rank_b, "BUY") if px_b >= fair_b[i] * 0.997 else hazard(99, "BUY")
            if rng.random() < 1 - np.exp(-h_b * p.calibration * dt_h[i]):
                q = min(draw_size(), q_max / px_b)
                if q * px_b >= p.min_ticket_pyg:
                    tot = q * px_b
                    cash -= tot
                    inv += q * (1 - f)
                    basis += tot
                    n_buy += 1
                    max_share = max(max_share, (inv * vb[i]) / max(capital, 1.0))

        # Accounting identity: cash + basis == initial_cash + pnl (float noise aside).
        assert abs((cash + basis) - (p.initial_cash + pnl)) < 1e-3 * max(p.initial_cash, 1.0)

    cm_f = basis / inv if inv else 0.0
    last_ask = va[~np.isnan(va)][-1] if np.any(~np.isnan(va)) else 0.0
    mtm = inv * (last_ask * (1 - p.fee) - cm_f)
    return DeskResult(pnl, mtm, cash, inv, cm_f, n_buy, n_sell, stuck, under, nostock,
                      max_share, daily)


def run_variants(times, sells, buys, variants: Dict[str, DeskParams], seeds: int = 8
                 ) -> Dict[str, List[DeskResult]]:
    """Run every variant with the same seeds (paired comparison)."""
    return {name: [simulate(times, sells, buys, prm, seed=s) for s in range(seeds)]
            for name, prm in variants.items()}
