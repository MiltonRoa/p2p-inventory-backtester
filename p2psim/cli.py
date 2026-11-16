# -*- coding: utf-8 -*-
"""Run the desk model on a synthetic book and compare variants with paired seeds.

    python -m p2psim.cli --days 60 --seeds 8
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np

from .book import BookParams, synth_book
from .model import DeskParams, run_variants


def fmt(x: float) -> str:
    return f"{x:+,.0f}"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P2P desk Monte Carlo on a synthetic order book")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--book-seed", type=int, default=0)
    ap.add_argument("--spread", type=float, default=0.010, help="median spread, e.g. 0.01 = 1 %%")
    ap.add_argument("--vol", type=float, default=40.0, help="daily vol of the mid in bps")
    ap.add_argument("--cash", type=float, default=40_000_000.0)
    a = ap.parse_args(argv)

    times, sells, buys = synth_book(BookParams(days=a.days, spread_median=a.spread,
                                               vol_bps_day=a.vol, seed=a.book_seed))
    base = dict(initial_cash=a.cash)
    variants = {
        "no cap (buy with 95 % of cash)": DeskParams(caps=None, **base),
        "cap 100/70/40 of capital": DeskParams(caps=(1.0, 0.7, 0.4), **base),
        "cap 90/70/40 (live rule)": DeskParams(caps=(0.9, 0.7, 0.4), **base),
        "cap 80/70/40": DeskParams(caps=(0.8, 0.7, 0.4), **base),
        "live rule + no-buy gate at spread < 0.5 %": DeskParams(caps=(0.9, 0.7, 0.4), spread_gate=0.005, **base),
    }
    res = run_variants(times, sells, buys, variants, seeds=a.seeds)
    ref = res["cap 90/70/40 (live rule)"]

    print(f"synthetic book: {a.days} days, median spread {a.spread*100:.2f} %, vol {a.vol:.0f} bps/day, "
          f"{a.seeds} paired seeds, capital {a.cash:,.0f} PYG\n")
    print(f"  {'variant':<44} {'total PnL':>12} {'+-':>10} {'vs live':>10} {'wins':>5} "
          f"{'buys':>5} {'sells':>5} {'stuck h':>8} {'under h':>8} {'idle h':>7} {'max inv':>8}")
    for name, rs in res.items():
        tot = np.array([r.total for r in rs])
        d = tot - np.array([r.total for r in ref])
        wins = int((d > 0).sum())
        print(f"  {name:<44} {fmt(tot.mean()):>12} {fmt(tot.std()):>10} {fmt(d.mean()):>10} "
              f"{wins:>3}/{len(rs):<2} {np.mean([r.buys for r in rs]):>5.0f} "
              f"{np.mean([r.sells for r in rs]):>5.0f} {np.mean([r.stuck_hours for r in rs]):>8.0f} "
              f"{np.mean([r.underwater_hours for r in rs]):>8.0f} "
              f"{np.mean([r.no_stock_hours for r in rs]):>7.0f} "
              f"{np.mean([r.max_inventory_share for r in rs])*100:>7.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
