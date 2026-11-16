# p2p-inventory-backtester

Monte Carlo model of a **peer-to-peer stablecoin desk**: a merchant who posts
one BUY and one SELL advertisement on a P2P order book (the study case is
PYG/USDT on Binance P2P), prices them against the live book and gets filled at
a rate that depends on queue position. The model was built to make decisions
for a desk running with real capital, where the expensive mistakes were not
bad prices but **inventory**: buying too much at a local top and sitting on it
for days.

The production repricer, advisor and nightly reconciliation that this model
served are private (they contain the desk's live pricing policy). What is
public here is the decision tool, with a synthetic order-book generator in
place of the proprietary one-snapshot-per-minute dataset.

## What the model captures

* **Pricing rules identical to the live bot.** SELL at `best_ask - 1` but
  never below `cost * (1 + fee + margin)`; if that leaves the ad outside the
  top 3, step to one unit below the third ask but never below break-even.
  BUY at `best_bid + 1` but never above a ceiling derived from the 24 h median
  ask, so anything bought can be resold at the target margin.
* **Queue-position fill hazards** measured on the real book (orders per hour
  at rank 1, 2-3, 4-6, 7-12, 13-20, 20+), scaled by a calibration factor.
  Ads priced off-market get the off-market hazard.
* **Inventory cap by position in the 72 h range** ("90/70/40"): in the lower
  third of the range the desk may hold up to 90 % of capital in inventory,
  70 % in the middle third, 40 % in the upper third.
* **Spread gate** (do not buy while the spread is narrow), kept as an option
  because it was proposed, tested with paired seeds and rejected.
* Exchange constraints: minimum ticket, fee paid in the asset on sales,
  integer prices.
* **Accounting identity** asserted at every step:
  `cash + cost_basis == initial_cash + realised_pnl`.

## What it decided, on the real data

Paired-seed runs on 113 days of real snapshots (8 seeds, same random draws
for every variant):

| Question | Result | Decision |
|---|---|---|
| Inventory cap 100/70/40 vs no cap | less time stuck above market (216 h to 88 h), smaller worst gap between floor and market, same PnL | adopted |
| 90/70/40 (always keep 10 % cash) vs 100/70/40 | costs about 0.1 M PYG over the period, buys optionality | adopted |
| 80/70/40 (always keep 20 % cash) | -0.9 M to -1.5 M PYG: idle cash on good days costs more than it saves on bad ones | rejected |
| "Pause when the spread is narrow" (gate 0.4-0.8 %) | -0.2 M to -4.2 M PYG, 0-5 wins out of 8, non-monotonic in the threshold | rejected as noise |
| Fixed capital tranches ("two lots of 20 M and one of 10 M") | -1.7 M to -2.8 M PYG, 1 win out of 8 | rejected |
| Hour-of-day effects | mid moves +-0.02 %/h, "cheap" nights are a thin book without flow | no time rule |

The backtest inflates absolute PnL about 2.5x (no cancellations, no dry
spells), so it is used **in relative terms only**: same seeds, same book,
difference between variants.

## Run

```
pip install -r requirements.txt
pytest -q
python -m p2psim.cli --days 60 --seeds 8
```

The CLI prints total PnL, dispersion, difference against the live rule,
wins out of N seeds, number of buys and sells, hours stuck above market,
hours under water, idle hours and maximum inventory share for each variant.
Numbers on the synthetic book are illustrative; the point is the machinery.

## Layout

```
p2psim/book.py   synthetic snapshot generator (random-walk mid, log-normal spread, N levels per side)
p2psim/model.py  desk rules, hazards, inventory cap, accounting
p2psim/cli.py    paired-seed comparison of variants
tests/           shape of the book, hazard monotonicity, no fills at zero hazard,
                 accounting identity, floor rule, cap limits exposure, gate reduces buys, reproducibility
```

MIT License.
