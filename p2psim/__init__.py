"""p2psim: Monte Carlo backtester for a P2P (fiat/stablecoin) market-making desk.

Models a merchant who posts one BUY and one SELL advertisement on a peer-to-peer
exchange order book, prices them against the live book, and gets filled at a
rate that depends on queue position.  The simulator was built to decide
inventory-cap and pricing rules for a real desk running on Binance P2P
(PYG/USDT) with its own capital; this package ships with a synthetic order-book
generator so the model can be run and tested without the proprietary
snapshots.
"""

from .book import BookParams, synth_book
from .model import DeskParams, simulate

__all__ = ["BookParams", "synth_book", "DeskParams", "simulate"]
__version__ = "1.0.0"
