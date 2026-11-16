# -*- coding: utf-8 -*-
import numpy as np
import pytest

from p2psim import BookParams, DeskParams, simulate, synth_book
from p2psim.model import hazard, run_variants


@pytest.fixture(scope="module")
def book():
    return synth_book(BookParams(days=10, step_minutes=5, seed=1))


def test_synthetic_book_shape_and_order(book):
    times, sells, buys = book
    assert len(times) == len(sells) == len(buys) == 10 * 24 * 12
    for s, b in zip(sells[:50], buys[:50]):
        assert np.all(np.diff(s) >= 0) and np.all(np.diff(b) >= 0)
        assert s[0] > b[-1]                      # best ask above best bid
        assert float(s[0]).is_integer()          # integer PYG prices


def test_hazard_is_highest_at_rank_one_and_lowest_off_market():
    # The measured table is not strictly monotonic (rank 4-6 SELL was observed
    # below rank 7-12), so only the endpoints are pinned down.
    for side in ("BUY", "SELL"):
        hz = {r: hazard(r, side) for r in (1, 2, 4, 8, 15, 30, 99)}
        assert hz[1] == max(hz.values()) > 0
        assert hz[99] == min(hz.values()) > 0
        assert hz[1] > hz[8] > hz[30]


def test_zero_calibration_means_no_fills(book):
    times, sells, buys = book
    r = simulate(times, sells, buys, DeskParams(calibration=0.0), seed=0)
    assert r.buys == 0 and r.sells == 0 and r.realised_pnl == 0.0
    assert r.cash == DeskParams().initial_cash


def test_accounting_identity_and_activity(book):
    times, sells, buys = book
    p = DeskParams(calibration=3.0)
    r = simulate(times, sells, buys, p, seed=3)
    assert r.buys > 0 and r.sells > 0
    # cash + cost basis == initial cash + realised pnl (the simulator asserts this every step)
    assert abs((r.cash + r.avg_cost * r.inventory) - (p.initial_cash + r.realised_pnl)) < 1.0
    assert r.inventory >= 0 and r.cash >= 0


def test_every_sale_is_above_cost_plus_fee(book):
    """The floor rule means realised PnL per sale is never negative."""
    times, sells, buys = book
    r = simulate(times, sells, buys, DeskParams(calibration=3.0), seed=5)
    assert all(v >= -1e-6 for v in r.daily_pnl.values())


def test_inventory_cap_limits_exposure_vs_no_cap(book):
    times, sells, buys = book
    res = run_variants(times, sells, buys, {
        "nocap": DeskParams(caps=None, calibration=3.0),
        "cap": DeskParams(caps=(0.5, 0.3, 0.2), calibration=3.0),
    }, seeds=4)
    share_nocap = np.mean([r.max_inventory_share for r in res["nocap"]])
    share_cap = np.mean([r.max_inventory_share for r in res["cap"]])
    assert share_cap < share_nocap
    assert share_cap <= 0.5 + 0.1   # cap plus one ticket of slack


def test_spread_gate_reduces_buys(book):
    times, sells, buys = book
    a = simulate(times, sells, buys, DeskParams(calibration=3.0), seed=2)
    b = simulate(times, sells, buys, DeskParams(calibration=3.0, spread_gate=0.02), seed=2)
    assert b.buys <= a.buys


def test_same_seed_is_reproducible(book):
    times, sells, buys = book
    r1 = simulate(times, sells, buys, DeskParams(), seed=9)
    r2 = simulate(times, sells, buys, DeskParams(), seed=9)
    assert r1.total == r2.total and r1.buys == r2.buys
