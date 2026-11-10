#!/usr/bin/env python3
"""
penny_trader.py - Intraday penny-stock bot (Alpaca) with 4 strategies:

  1. Gap & Go           gap > 4%, RVOL > 3x, low float; ORB-5 break or VWAP pullback
  2. VWAP Momentum      cross above VWAP + volume spike + expanding MACD histogram
  3. Opening Range Breakout (15 min by default) with volume confirmation
  4. Panic Dip Buy      RSI < 25 on heavy volume, buy first bullish reversal, target VWAP

SETUP
  pip install alpaca-py pandas numpy yfinance        (yfinance is optional, for float data)
  Create a file named .env next to this script (see .env.example) with your PAPER keys.

RUN
  python penny_trader.py --dry-run                   # logs signals only, no orders
  python penny_trader.py                             # PAPER trading (default)
  python penny_trader.py --watchlist my_symbols.txt  # add your own tickers (one per line)
  python penny_trader.py --live                      # REAL money (asks for confirmation)

NOTES
  * Educational software, not financial advice. Penny stocks are extremely risky.
  * Free Alpaca data uses the IEX feed (a small slice of total volume). RVOL is computed
    against IEX history so it is self-consistent, but absolute volumes look small.
  * Alpaca does not trade most OTC/pink-sheet stocks; this targets exchange-listed
    low-priced stocks ($0.50-$5).
  * Backtest and paper trade for weeks before risking real money.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.enums import MostActivesBy
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

ET = ZoneInfo("America/New_York")
OPEN_MOD = 9 * 60 + 30  # minutes since midnight
log = logging.getLogger("penny")


# --------------------------------------------------------------------------- config
SEED_HIGH_LIQUIDITY = ["PLUG", "OPEN", "NIO", "RIG"]                 # ORB / VWAP testing
SEED_LOW_FLOAT_MOMO = ["SOUN", "BBAI", "RGTI", "SDA", "SNDL"]         # Gap & Go
SEED_DIP_BUY = ["ACRS", "STTK", "DBGI"]                               # Panic Dip
SEEDS = SEED_HIGH_LIQUIDITY + SEED_LOW_FLOAT_MOMO + SEED_DIP_BUY


def load_env(path: str = ".env"):
    """Tiny .env loader so keys never live in the source code."""
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Config:
    # universe
    min_price: float = 0.50
    max_price: float = 5.00
    max_universe: int = 40
    screen_min_volume: int = 1_000_000   # daily volume filter for scanned symbols
    screen_min_rvol: float = 2.0         # RVOL filter for scanned symbols
    # Gap & Go
    gap_min_pct: float = 4.0
    rvol_min: float = 3.0
    max_float: float = 20_000_000
    allow_unknown_float: bool = True   # float data is often missing; set False to be strict
    gap_orb_minutes: int = 5
    # ORB
    orb_minutes: int = 15
    orb_vol_mult: float = 1.5
    # VWAP momentum
    vwap_vol_mult: float = 2.0
    # Panic dip
    rsi_oversold: float = 25.0
    dip_min_drop_pct: float = 8.0
    dip_vol_mult: float = 2.0
    # risk
    risk_per_trade_pct: float = 0.5     # % of equity risked per trade
    max_position_pct: float = 10.0      # max % of equity in one position
    max_open_positions: int = 3
    max_trades_per_day: int = 6
    daily_loss_limit_pct: float = 3.0   # stop trading for the day past this drawdown
    reward_risk: float = 2.0
    max_stop_pct: float = 8.0           # cap stop distance
    flatten_at: dtime = dtime(15, 50)   # close everything before the close


@dataclass
class Signal:
    symbol: str
    strategy: str
    entry: float
    stop: float
    target: float
    reason: str


@dataclass
class Ctx:
    symbol: str
    rth: pd.DataFrame
    gap_pct: float
    rvol: float
    flt: Optional[float]
    screened: bool = False   # came from the live scanner (not the fixed seed list)


# --------------------------------------------------------------------------- indicators
def add_indicators(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    tp = (r.high + r.low + r.close) / 3
    r["vwap"] = (tp * r.volume).cumsum() / r.volume.cumsum().replace(0, np.nan)
    delta = r.close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    r["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)
    macd = r.close.ewm(span=12, adjust=False).mean() - r.close.ewm(span=26, adjust=False).mean()
    r["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    r["vol_avg"] = r.volume.rolling(20, min_periods=5).mean().shift(1)
    r["mod"] = r.index.hour * 60 + r.index.minute
    return r


def float_ok(c: Ctx, cfg: Config) -> bool:
    return cfg.allow_unknown_float if c.flt is None else c.flt <= cfg.max_float


def cap_stop(entry: float, stop: float, cfg: Config) -> float:
    return max(stop, entry * (1 - cfg.max_stop_pct / 100))


# --------------------------------------------------------------------------- strategies
def gap_and_go(c: Ctx, cfg: Config) -> Optional[Signal]:
    r = c.rth
    end = OPEN_MOD + cfg.gap_orb_minutes
    if c.gap_pct < cfg.gap_min_pct or not c.rvol >= cfg.rvol_min or not float_ok(c, cfg):
        return None
    rng = r[r["mod"] < end]
    if rng.empty or r["mod"].iloc[-1] < end or r["mod"].iloc[-1] > OPEN_MOD + 60:
        return None
    hi, lo = rng.high.max(), rng.low.min()
    a, b = r.iloc[-2], r.iloc[-1]
    # (a) opening-range breakout
    if b.close > hi and (r.close.iloc[-4:-1] <= hi).any() and b.close > b.vwap:
        entry = b.close
        stop = cap_stop(entry, max(lo, b.vwap * 0.99), cfg)
        return Signal(c.symbol, "GapAndGo-ORB", entry, stop, entry + cfg.reward_risk * (entry - stop),
                      f"gap {c.gap_pct:.1f}% rvol {c.rvol:.1f} broke {end - OPEN_MOD}m high {hi:.4f}")
    # (b) early VWAP pullback after having broken out
    if (r.high.max() > hi and b.low <= b.vwap * 1.005 and b.close > b.vwap
            and b.close > b.open and a.close < a.open * 1.02):
        entry = b.close
        stop = cap_stop(entry, min(b.low, b.vwap) * 0.995, cfg)
        return Signal(c.symbol, "GapAndGo-VWAPpullback", entry, stop,
                      entry + cfg.reward_risk * (entry - stop),
                      f"gap {c.gap_pct:.1f}% rvol {c.rvol:.1f} VWAP pullback bounce")
    return None


def vwap_momentum(c: Ctx, cfg: Config) -> Optional[Signal]:
    r = c.rth
    if len(r) < 35:
        return None
    a, b = r.iloc[-2], r.iloc[-1]
    crossed = a.close <= a.vwap and b.close > b.vwap
    spike = pd.notna(b.vol_avg) and b.volume >= cfg.vwap_vol_mult * b.vol_avg
    macd_up = b.macd_hist > 0 and b.macd_hist > a.macd_hist
    if crossed and spike and macd_up:
        entry = b.close
        stop = cap_stop(entry, min(r.low.iloc[-5:].min(), b.vwap * 0.99), cfg)
        return Signal(c.symbol, "VWAP-Momentum", entry, stop,
                      entry + cfg.reward_risk * (entry - stop),
                      f"VWAP cross, vol {b.volume / b.vol_avg:.1f}x, MACD hist rising")
    return None


def opening_range_breakout(c: Ctx, cfg: Config) -> Optional[Signal]:
    r = c.rth
    end = OPEN_MOD + cfg.orb_minutes
    rng = r[r["mod"] < end]
    if len(rng) < 3 or r["mod"].iloc[-1] < end or r["mod"].iloc[-1] > 11 * 60 + 30:
        return None
    hi, lo = rng.high.max(), rng.low.min()
    a, b = r.iloc[-2], r.iloc[-1]
    spike = pd.notna(b.vol_avg) and b.volume >= cfg.orb_vol_mult * b.vol_avg
    if b.close > hi and a.close <= hi and spike:
        entry = b.close
        stop = cap_stop(entry, lo, cfg)
        return Signal(c.symbol, f"ORB-{cfg.orb_minutes}", entry, stop,
                      entry + cfg.reward_risk * (entry - stop),
                      f"broke {cfg.orb_minutes}m range high {hi:.4f}, vol {b.volume / b.vol_avg:.1f}x")
    return None


def panic_dip(c: Ctx, cfg: Config) -> Optional[Signal]:
    r = c.rth
    if len(r) < 20:
        return None
    a, b = r.iloc[-2], r.iloc[-1]
    drop = (r.high.max() - r.low.iloc[-6:].min()) / r.high.max() * 100
    was_oversold = r.rsi.iloc[-4:-1].min() < cfg.rsi_oversold
    heavy = ((r.volume.iloc[-6:] >= cfg.dip_vol_mult * r.vol_avg.iloc[-6:]).any())
    reversal = b.close > b.open and a.close < a.open and b.close > a.close
    if drop >= cfg.dip_min_drop_pct and was_oversold and heavy and reversal and b.close < b.vwap:
        entry = b.close
        stop = cap_stop(entry, r.low.iloc[-4:].min() * 0.995, cfg)
        target = b.vwap  # mean-reversion target
        if target - entry < (entry - stop):  # require at least 1R to VWAP
            return None
        return Signal(c.symbol, "PanicDip", entry, stop, target,
                      f"drop {drop:.1f}% RSI<{cfg.rsi_oversold:.0f} reversal candle, target VWAP")
    return None


STRATEGIES = [gap_and_go, opening_range_breakout, vwap_momentum, panic_dip]


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self, key: str, secret: str, cfg: Config):
        self.cfg = cfg
        self.hist = StockHistoricalDataClient(key, secret)
        self.screener = ScreenerClient(key, secret)
        self.profiles: dict[str, Optional[dict]] = {}

    def bars(self, symbols, start, end, tf=TimeFrame.Minute) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf,
                               start=start, end=end, feed=DataFeed.IEX)
        df = self.hist.get_stock_bars(req).df
        out = {}
        if df.empty:
            return out
        present = set(df.index.get_level_values(0))
        for s in symbols:
            if s in present:
                d = df.xs(s, level=0).copy()
                d.index = d.index.tz_convert(ET)
                out[s] = d
        return out

    def universe(self, watchlist: list[str]) -> tuple[list[str], set[str]]:
        """Fixed seeds + live scan (price < max, volume > 1M). Returns (symbols, scanned)."""
        cfg = self.cfg
        scanned: set[str] = set()
        try:
            movers = self.screener.get_market_movers(MarketMoversRequest(top=50))
            for g in movers.gainers:
                if cfg.min_price <= g.price <= cfg.max_price:
                    scanned.add(g.symbol)
        except Exception as e:
            log.warning("movers screener unavailable: %s", e)
        try:
            act = self.screener.get_most_actives(MostActivesRequest(by=MostActivesBy.VOLUME, top=50))
            for a in act.most_actives:
                if a.volume >= cfg.screen_min_volume:
                    scanned.add(a.symbol)
        except Exception as e:
            log.warning("most-actives screener unavailable: %s", e)
        seeds = set(watchlist)
        syms = sorted(seeds | scanned)[: cfg.max_universe + len(seeds)]
        return syms, scanned - seeds

    @staticmethod
    def _float(sym: str) -> Optional[float]:
        try:
            import yfinance as yf
            f = yf.Ticker(sym).info.get("floatShares")
            return float(f) if f else None
        except Exception:
            return None

    def build_profiles(self, syms: list[str], now: datetime):
        need = [s for s in syms if s not in self.profiles or
                (self.profiles[s] or {}).get("date") != now.date()]
        need = [s for s in need if not (s in self.profiles and self.profiles[s] is None)]
        if not need:
            return
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hist = self.bars(need, midnight - timedelta(days=12), midnight)
        for s in need:
            d = hist.get(s)
            if d is None or d.empty:
                self.profiles[s] = None
                continue
            d["date"] = d.index.date
            d["mod"] = d.index.hour * 60 + d.index.minute
            curves = [g.groupby("mod")["volume"].sum().reindex(range(240, 960), fill_value=0).cumsum()
                      for _, g in d.groupby("date")]
            rth = d.between_time("09:30", "15:59")
            if rth.empty:
                self.profiles[s] = None
                continue
            self.profiles[s] = {
                "date": now.date(),
                "avg_curve": pd.concat(curves, axis=1).mean(axis=1),
                "prior_close": float(rth.close.iloc[-1]),
                "float": self._float(s),
            }

    def contexts(self, syms: list[str], now: datetime, scanned: set[str] = frozenset()) -> list[Ctx]:
        self.build_profiles(syms, now)
        start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        today = self.bars(syms, start, now)
        cur_min = now.replace(second=0, microsecond=0)
        ctxs = []
        for s, d in today.items():
            p = self.profiles.get(s)
            d = d[d.index < cur_min]  # drop the still-forming bar
            if p is None or d.empty:
                continue
            rth = d.between_time("09:30", "15:59")
            if len(rth) < 3:
                continue
            rth = add_indicators(rth)
            if not (self.cfg.min_price <= rth.close.iloc[-1] <= self.cfg.max_price):
                continue
            gap = (rth.open.iloc[0] - p["prior_close"]) / p["prior_close"] * 100
            mod_now = int(d.index[-1].hour * 60 + d.index[-1].minute)
            expected = p["avg_curve"].get(mod_now, np.nan)
            rvol = float(d.volume.sum() / expected) if expected and expected > 0 else float("nan")
            ctxs.append(Ctx(s, rth, float(gap), rvol, p["float"], s in scanned))
        return ctxs


# --------------------------------------------------------------------------- execution
def px(p: float) -> float:
    return round(p, 2) if p >= 1 else round(p, 4)


class Trader:
    def __init__(self, cfg: Config, tc: TradingClient, dry_run: bool):
        self.cfg, self.tc, self.dry = cfg, tc, dry_run
        self.done: set[str] = set()     # one trade per symbol per day
        self.trades_today = 0
        self.day = None

    def reset_if_new_day(self, d):
        if self.day != d:
            self.day, self.done, self.trades_today = d, set(), 0

    def daily_loss_hit(self) -> bool:
        a = self.tc.get_account()
        last = float(a.last_equity)
        pnl = (float(a.equity) - last) / last * 100 if last else 0
        if pnl <= -self.cfg.daily_loss_limit_pct:
            log.warning("Daily loss limit hit (%.2f%%) - no new trades", pnl)
            return True
        return False

    def submit(self, s: Signal):
        cfg = self.cfg
        acct = self.tc.get_account()
        equity = float(acct.equity)
        entry = s.entry * 1.005                      # marketable limit
        stop = min(s.stop, entry - 0.01)
        target = max(s.target, entry + 0.01)
        rps = entry - stop
        qty = min(int(equity * cfg.risk_per_trade_pct / 100 / rps),
                  int(equity * cfg.max_position_pct / 100 / entry),
                  int(float(acct.buying_power) * 0.95 / entry))
        if qty < 1:
            log.info("%s skipped: size < 1 share", s.symbol)
            return
        log.info("SIGNAL %s %s qty=%d entry=%.4f stop=%.4f target=%.4f | %s",
                 s.symbol, s.strategy, qty, entry, stop, target, s.reason)
        self._csv(s, qty, entry, stop, target)
        self.done.add(s.symbol)
        if self.dry:
            return
        self.trades_today += 1
        self.tc.submit_order(LimitOrderRequest(
            symbol=s.symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            limit_price=px(entry), order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=px(target)),
            stop_loss=StopLossRequest(stop_price=px(stop)),
        ))

    def _csv(self, s, qty, entry, stop, target):
        new = not os.path.exists("signals.csv")
        with open("signals.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "strategy", "qty", "entry", "stop", "target", "reason"])
            w.writerow([datetime.now(ET).isoformat(timespec="seconds"), s.symbol, s.strategy,
                        qty, px(entry), px(stop), px(target), s.reason])

    def busy_symbols(self) -> set[str]:
        held = {p.symbol for p in self.tc.get_all_positions()}
        orders = self.tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return held | {o.symbol for o in orders}

    def flatten(self):
        log.info("Flattening all positions and cancelling orders")
        if not self.dry:
            self.tc.cancel_orders()
            self.tc.close_all_positions(cancel_orders=True)


# --------------------------------------------------------------------------- main loop
def load_watchlist(path: Optional[str]) -> list[str]:
    if not path:
        return []
    with open(path) as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def run(args):
    load_env()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Put ALPACA_API_KEY and ALPACA_SECRET_KEY in a .env file next to this script.")
    if args.live:
        if input("LIVE TRADING with real money. Type 'I ACCEPT THE RISK' to continue: ") != "I ACCEPT THE RISK":
            sys.exit("Aborted.")
    cfg = Config()
    tc = TradingClient(key, secret, paper=not args.live)
    data = Data(key, secret, cfg)
    trader = Trader(cfg, tc, args.dry_run)
    watch = sorted(set(SEEDS) | set(load_watchlist(args.watchlist)))
    log.info("Mode: %s%s", "LIVE" if args.live else "PAPER", " (dry-run)" if args.dry_run else "")

    last_universe: tuple[float, list[str], set[str]] = (0.0, [], set())
    while True:
        try:
            clock = tc.get_clock()
            now = datetime.now(ET)
            if not clock.is_open:
                log.info("Market closed. Next open %s", clock.next_open)
                time.sleep(300)
                continue
            trader.reset_if_new_day(now.date())

            if now.time() >= cfg.flatten_at:
                trader.flatten()
                time.sleep(600)
                continue

            if time.time() - last_universe[0] > 300:
                syms, scanned = data.universe(watch)
                last_universe = (time.time(), syms, scanned)
                log.info("Universe (%d, %d scanned): %s", len(syms), len(scanned), ", ".join(syms))

            if (not trader.daily_loss_hit() and trader.trades_today < cfg.max_trades_per_day):
                busy = trader.busy_symbols()
                open_slots = cfg.max_open_positions - len({p.symbol for p in tc.get_all_positions()})
                for ctx in data.contexts(last_universe[1], now, last_universe[2]):
                    if open_slots <= 0 or ctx.symbol in busy or ctx.symbol in trader.done:
                        continue
                    if ctx.screened and not ctx.rvol >= cfg.screen_min_rvol:
                        continue  # scanned names must show RVOL > 2; seed names are always eligible
                    for strat in STRATEGIES:
                        sig = strat(ctx, cfg)
                        if sig:
                            trader.submit(sig)
                            open_slots -= 1
                            break
        except KeyboardInterrupt:
            log.info("Stopped by user")
            return
        except Exception as e:
            log.exception("loop error: %s", e)
        # wake a few seconds after the next minute bar closes
        time.sleep(max(5, 63 - datetime.now().second))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Penny stock strategy bot (Alpaca)")
    ap.add_argument("--live", action="store_true", help="trade real money (default: paper)")
    ap.add_argument("--dry-run", action="store_true", help="log signals only; place no orders")
    ap.add_argument("--watchlist", help="text file with one ticker per line")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(ap.parse_args())
