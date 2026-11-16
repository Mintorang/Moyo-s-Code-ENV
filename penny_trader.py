#!/usr/bin/env python3
"""
penny_trader.py - ONE-SHOT penny-stock bot for GitHub Actions (runs hourly, then exits).

Each run: check market clock -> cancel stale entries -> (flatten near the close) ->
scan seeds + live movers -> look back over the last ~75 minutes of 1-min bars for any
valid trigger from 4 strategies -> place Alpaca BRACKET orders (stop + target live
server-side, so exits happen between hourly runs).

Strategies: Gap & Go, VWAP Momentum, Opening Range Breakout, Panic Dip Buy.

Secrets (GitHub -> Settings -> Secrets and variables -> Actions):
  ALPACA_API_KEY, ALPACA_SECRET_KEY            (repository secrets)
Optional repo variables: ALPACA_PAPER (default true), DRY_RUN, RISK_PER_TRADE_PCT,
  MAX_OPEN_POSITIONS, REWARD_RISK, EXTRA_TICKERS (comma separated)

Educational software, not financial advice. Profit is NOT guaranteed.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed, MostActivesBy
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
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
OPEN_MOD = 9 * 60 + 30
log = logging.getLogger("penny")

SEED_HIGH_LIQUIDITY = ["PLUG", "OPEN", "NIO", "RIG"]
SEED_LOW_FLOAT_MOMO = ["SOUN", "BBAI", "RGTI", "SDA", "SNDL"]
SEED_DIP_BUY = ["ACRS", "STTK", "DBGI"]
SEEDS = SEED_HIGH_LIQUIDITY + SEED_LOW_FLOAT_MOMO + SEED_DIP_BUY


def env(name: str, default):
    v = os.getenv(name)
    return default if v in (None, "") else type(default)(v)


@dataclass
class Config:
    # universe
    min_price: float = 0.50
    max_price: float = 5.00
    max_scanned: int = 30
    screen_min_volume: int = 1_000_000
    screen_min_rvol: float = 2.0
    # Gap & Go
    gap_min_pct: float = 4.0
    rvol_min: float = 3.0
    max_float: float = 20_000_000
    allow_unknown_float: bool = True
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
    # hourly-run behaviour
    lookback: int = 75               # accept triggers from the last N minutes
    max_chase_pct: float = 3.0       # skip if price already ran >3% past the trigger
    entry_slip_pct: float = 0.75     # marketable-limit slack so orders actually fill
    stale_entry_minutes: int = 30    # cancel unfilled entries older than this
    flatten_minutes: int = 35        # flatten when <= this many minutes to the close
    # risk
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 10.0
    max_open_positions: int = 3
    max_trades_per_day: int = 6
    daily_loss_limit_pct: float = 3.0
    reward_risk: float = 1.5
    max_stop_pct: float = 8.0

    def __post_init__(self):
        self.risk_per_trade_pct = env("RISK_PER_TRADE_PCT", self.risk_per_trade_pct)
        self.max_open_positions = env("MAX_OPEN_POSITIONS", self.max_open_positions)
        self.reward_risk = env("REWARD_RISK", self.reward_risk)


@dataclass
class Signal:
    symbol: str
    strategy: str
    entry: float
    stop: float
    target: float
    rvol: float
    reason: str


@dataclass
class Ctx:
    symbol: str
    rth: pd.DataFrame
    gap_pct: float
    rvol: float
    flt: Optional[float]
    screened: bool = False


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
    r["pos"] = np.arange(len(r))
    return r


def last_true(mask: pd.Series, lookback: int) -> Optional[int]:
    """Position of the most recent True within the last `lookback` bars."""
    m = mask.fillna(False).astype(bool).iloc[-lookback:]
    idx = np.flatnonzero(m.values)
    return len(mask) - len(m) + int(idx[-1]) if len(idx) else None


# --------------------------------------------------------------------------- strategies
# each returns (name, trigger_pos, stop, target_or_None, reason) or None
def gap_and_go(c: Ctx, cfg: Config):
    r = c.rth
    if c.gap_pct < cfg.gap_min_pct or not c.rvol >= cfg.rvol_min:
        return None
    if (cfg.allow_unknown_float if c.flt is None else c.flt <= cfg.max_float) is False:
        return None
    end = OPEN_MOD + cfg.gap_orb_minutes
    rng = r[r["mod"] < end]
    if rng.empty or r["mod"].iloc[-1] < end:
        return None
    hi, lo = rng.high.max(), rng.low.min()
    brk = ((r.close > hi) & (r.close.shift() <= hi) & (r.close > r.vwap)
           & (r["mod"] >= end) & (r["mod"] <= OPEN_MOD + 90))
    pb = ((r.low <= r.vwap * 1.005) & (r.close > r.vwap) & (r.close > r.open)
          & (r.high.cummax().shift() > hi) & (r["mod"] <= OPEN_MOD + 120))
    i = last_true(brk | pb, cfg.lookback)
    if i is None:
        return None
    row = r.iloc[i]
    if brk.iloc[i]:
        return ("GapAndGo-ORB", i, max(lo, row.vwap * 0.99), None,
                f"gap {c.gap_pct:.1f}% rvol {c.rvol:.1f} broke {hi:.4f}")
    return ("GapAndGo-VWAPpullback", i, min(row.low, row.vwap) * 0.995, None,
            f"gap {c.gap_pct:.1f}% rvol {c.rvol:.1f} VWAP pullback")


def orb(c: Ctx, cfg: Config):
    r = c.rth
    end = OPEN_MOD + cfg.orb_minutes
    rng = r[r["mod"] < end]
    if len(rng) < 3 or r["mod"].iloc[-1] < end:
        return None
    hi, lo = rng.high.max(), rng.low.min()
    trig = ((r.close > hi) & (r.close.shift() <= hi)
            & (r.volume >= cfg.orb_vol_mult * r.vol_avg)
            & (r["mod"] >= end) & (r["mod"] <= 12 * 60))
    i = last_true(trig, cfg.lookback)
    if i is None:
        return None
    return (f"ORB-{cfg.orb_minutes}", i, lo, None, f"broke {cfg.orb_minutes}m high {hi:.4f}")


def vwap_momentum(c: Ctx, cfg: Config):
    r = c.rth
    cross = (r.close.shift() <= r.vwap.shift()) & (r.close > r.vwap)
    spike = r.volume >= cfg.vwap_vol_mult * r.vol_avg
    macd_up = (r.macd_hist > 0) & (r.macd_hist > r.macd_hist.shift())
    i = last_true(cross & spike & macd_up & (r["pos"] >= 30), cfg.lookback)
    if i is None:
        return None
    stop = min(r.low.rolling(5).min().iloc[i], r.vwap.iloc[i] * 0.99)
    return ("VWAP-Momentum", i, stop, None, "VWAP cross + vol spike + MACD hist rising")


def panic_dip(c: Ctx, cfg: Config):
    r = c.rth
    hh = r.high.cummax()
    drop = (hh - r.low.rolling(6).min()) / hh * 100
    oversold = r.rsi.shift().rolling(3).min() < cfg.rsi_oversold
    heavy = (r.volume >= cfg.dip_vol_mult * r.vol_avg).astype(float).rolling(6, min_periods=1).max() > 0
    rev = (r.close > r.open) & (r.close.shift() < r.open.shift()) & (r.close > r.close.shift())
    trig = (drop >= cfg.dip_min_drop_pct) & oversold & heavy & rev & (r.close < r.vwap) & (r["pos"] >= 20)
    i = last_true(trig, cfg.lookback)
    if i is None:
        return None
    stop = r.low.rolling(4).min().iloc[i] * 0.995
    return ("PanicDip", i, stop, float(r.vwap.iloc[-1]), f"drop {drop.iloc[i]:.1f}% RSI<25 reversal, target VWAP")


STRATEGIES = [gap_and_go, orb, vwap_momentum, panic_dip]


def evaluate(c: Ctx, cfg: Config) -> Optional[Signal]:
    r = c.rth
    entry = float(r.close.iloc[-1])
    for fn in STRATEGIES:
        out = fn(c, cfg)
        if not out:
            continue
        name, i, stop, target, reason = out
        if entry > float(r.close.iloc[i]) * (1 + cfg.max_chase_pct / 100):
            log.info("%s %s trigger found but price already ran > %.1f%%", c.symbol, name, cfg.max_chase_pct)
            continue
        stop = max(float(stop), entry * (1 - cfg.max_stop_pct / 100))
        stop = min(stop, entry * 0.99)               # at least a 1% stop
        if not stop < entry:
            continue
        risk = entry - stop
        if target is None:
            target = entry + cfg.reward_risk * risk
        elif target - entry < risk:                  # need >= 1R to VWAP
            continue
        return Signal(c.symbol, name, entry, stop, float(target), c.rvol, reason)
    return None


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self, key: str, secret: str, cfg: Config):
        self.cfg = cfg
        self.hist = StockHistoricalDataClient(key, secret)
        self.screener = ScreenerClient(key, secret)
        self.profiles: dict[str, Optional[dict]] = {}
        self._floats: dict[str, Optional[float]] = {}

    def bars(self, symbols, start, end) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        syms = list(symbols)
        for k in range(0, len(syms), 40):
            chunk = syms[k:k + 40]
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                   start=start, end=end, feed=DataFeed.IEX)
            df = self.hist.get_stock_bars(req).df
            if df.empty:
                continue
            present = set(df.index.get_level_values(0))
            for s in chunk:
                if s in present:
                    d = df.xs(s, level=0).copy()
                    d.index = d.index.tz_convert(ET)
                    out[s] = d
        return out

    def universe(self, seeds: list[str]) -> tuple[list[str], set[str]]:
        cfg, scanned = self.cfg, set()
        try:
            for g in self.screener.get_market_movers(MarketMoversRequest(top=50)).gainers:
                if cfg.min_price <= g.price <= cfg.max_price:
                    scanned.add(g.symbol)
        except Exception as e:
            log.warning("movers screener failed: %s", e)
        try:
            acts = self.screener.get_most_actives(MostActivesRequest(by=MostActivesBy.VOLUME, top=50))
            scanned |= {a.symbol for a in acts.most_actives if a.volume >= cfg.screen_min_volume}
        except Exception as e:
            log.warning("most-actives screener failed: %s", e)
        extra = sorted(scanned - set(seeds))[: cfg.max_scanned]
        return list(seeds) + extra, set(extra)

    def _float(self, sym: str) -> Optional[float]:
        if sym not in self._floats:
            try:
                import yfinance as yf
                f = yf.Ticker(sym).info.get("floatShares")
                self._floats[sym] = float(f) if f else None
            except Exception:
                self._floats[sym] = None
        return self._floats[sym]

    def _profiles(self, syms, now):
        need = [s for s in syms if s not in self.profiles]
        if not need:
            return
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hist = self.bars(need, midnight - timedelta(days=10), midnight)
        for s in need:
            d = hist.get(s)
            rth = d.between_time("09:30", "15:59") if d is not None and not d.empty else None
            if rth is None or rth.empty:
                self.profiles[s] = None
                continue
            d = d.assign(date=d.index.date, mod=d.index.hour * 60 + d.index.minute)
            curves = [g.groupby("mod")["volume"].sum().reindex(range(240, 960), fill_value=0).cumsum()
                      for _, g in d.groupby("date")]
            self.profiles[s] = {"avg_curve": pd.concat(curves, axis=1).mean(axis=1),
                                "prior_close": float(rth.close.iloc[-1])}

    def contexts(self, syms, now, scanned) -> list[Ctx]:
        cfg = self.cfg
        self._profiles(syms, now)
        today = self.bars(syms, now.replace(hour=4, minute=0, second=0, microsecond=0), now)
        cur_min = now.replace(second=0, microsecond=0)
        ctxs, skipped = [], {}
        for s in syms:
            p, d = self.profiles.get(s), today.get(s)
            if p is None:
                skipped[s] = "no history"
                continue
            if d is None or d.empty:
                skipped[s] = "no bars today"
                continue
            d = d[d.index < cur_min]
            rth = d.between_time("09:30", "15:59")
            if len(rth) < 3:
                skipped[s] = "<3 regular-hours bars"
                continue
            price = float(rth.close.iloc[-1])
            if not cfg.min_price <= price <= cfg.max_price:
                skipped[s] = f"price ${price:.2f} outside range"
                continue
            rth = add_indicators(rth)
            gap = (float(rth.open.iloc[0]) - p["prior_close"]) / p["prior_close"] * 100
            mod_now = int(d.index[-1].hour * 60 + d.index[-1].minute)
            exp = p["avg_curve"].get(mod_now, np.nan)
            rvol = float(d.volume.sum() / exp) if exp and exp > 0 else float("nan")
            flt = self._float(s) if gap >= cfg.gap_min_pct else None
            ctxs.append(Ctx(s, rth, gap, rvol, flt, s in scanned))
            log.info("  %-6s $%.2f gap %+5.1f%% rvol %4.1f bars %d", s, price, gap, rvol, len(rth))
        if skipped:
            log.info("Skipped: %s", "; ".join(f"{k} ({v})" for k, v in skipped.items()))
        return ctxs


# --------------------------------------------------------------------------- execution
def px(p: float) -> float:
    return round(p, 2) if p >= 1 else round(p, 4)


class Trader:
    def __init__(self, cfg: Config, tc: TradingClient, dry: bool):
        self.cfg, self.tc, self.dry = cfg, tc, dry
        self.events: list[str] = []

    def note(self, msg: str):
        log.info(msg)
        self.events.append(msg)

    def daily_loss_hit(self, acct) -> bool:
        last = float(acct.last_equity or 0)
        pnl = (float(acct.equity) - last) / last * 100 if last else 0
        self.note(f"Equity ${float(acct.equity):,.2f} (day {pnl:+.2f}%)")
        return pnl <= -self.cfg.daily_loss_limit_pct

    def cancel_stale_entries(self):
        now = datetime.now(timezone.utc)
        for o in self.tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            age = (now - o.submitted_at).total_seconds() / 60 if o.submitted_at else 0
            if o.side == OrderSide.BUY and float(o.filled_qty or 0) == 0 and age > self.cfg.stale_entry_minutes:
                self.note(f"Cancelling stale unfilled entry {o.symbol} ({age:.0f} min old)")
                if not self.dry:
                    try:
                        self.tc.cancel_order_by_id(o.id)
                    except Exception as e:
                        log.warning("cancel failed: %s", e)

    def flatten(self):
        self.note("Near close: cancelling orders and closing ALL positions")
        if not self.dry:
            try:
                self.tc.close_all_positions(cancel_orders=True)
            except Exception as e:
                log.error("flatten failed: %s", e)

    def todays_buys(self) -> set[str]:
        midnight = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
        orders = self.tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=midnight, limit=500))
        return {o.symbol for o in orders if o.side == OrderSide.BUY}

    def submit(self, s: Signal, acct) -> bool:
        cfg = self.cfg
        equity = float(acct.equity)
        limit = s.entry * (1 + cfg.entry_slip_pct / 100)
        stop = min(s.stop, limit - 0.01)
        target = max(s.target, limit + 0.01)
        qty = min(int(equity * cfg.risk_per_trade_pct / 100 / (limit - stop)),
                  int(equity * cfg.max_position_pct / 100 / limit),
                  int(float(acct.buying_power) * 0.95 / limit))
        if qty < 1:
            self.note(f"{s.symbol}: {s.strategy} signal but position size < 1 share")
            return False
        self.note(f"BUY {s.symbol} x{qty} limit {px(limit)} stop {px(stop)} target {px(target)} "
                  f"[{s.strategy}] {s.reason}")
        self._csv(s, qty, limit, stop, target)
        if self.dry:
            return True
        try:
            o = self.tc.submit_order(LimitOrderRequest(
                symbol=s.symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                limit_price=px(limit), order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=px(target)),
                stop_loss=StopLossRequest(stop_price=px(stop))))
            self.note(f"  -> order accepted id={o.id} status={o.status}")
            return True
        except Exception as e:
            self.note(f"  -> ORDER REJECTED for {s.symbol}: {e}")
            return False

    @staticmethod
    def _csv(s, qty, limit, stop, target):
        new = not os.path.exists("signals.csv")
        with open("signals.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "strategy", "qty", "limit", "stop", "target", "reason"])
            w.writerow([datetime.now(ET).isoformat(timespec="seconds"), s.symbol, s.strategy,
                        qty, px(limit), px(stop), px(target), s.reason])

    def write_summary(self):
        path = os.getenv("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a") as f:
                f.write("### Penny trader run\n" + "\n".join(f"- {e}" for e in self.events) + "\n")


# --------------------------------------------------------------------------- one run
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY missing - add them as GitHub repository secrets.")
        return 1
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    dry = os.getenv("DRY_RUN", "false").lower() == "true"
    cfg = Config()
    tc = TradingClient(key, secret, paper=paper)
    trader = Trader(cfg, tc, dry)
    trader.note(f"Mode: {'PAPER' if paper else 'LIVE'}{' (dry-run: no orders sent)' if dry else ''}")

    clock = tc.get_clock()
    if not clock.is_open:
        trader.note(f"Market closed (next open {clock.next_open}). Nothing to do.")
        trader.write_summary()
        return 0
    mins_left = (clock.next_close - clock.timestamp).total_seconds() / 60
    now = datetime.now(ET)

    trader.cancel_stale_entries()
    if mins_left <= cfg.flatten_minutes:
        trader.flatten()
        trader.write_summary()
        return 0

    acct = tc.get_account()
    if acct.trading_blocked or acct.account_blocked:
        trader.note("Account is blocked from trading - check the Alpaca dashboard.")
        trader.write_summary()
        return 1
    if trader.daily_loss_hit(acct):
        trader.note("Daily loss limit reached - no new trades today.")
        trader.write_summary()
        return 0
    if float(acct.equity) < 25_000 and int(acct.daytrade_count or 0) >= 3:
        trader.note("PDT limit: 3 day trades used on a <$25k account - skipping new entries.")
        trader.write_summary()
        return 0

    bought_today = trader.todays_buys()
    held = {p.symbol for p in tc.get_all_positions()}
    open_syms = {o.symbol for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    busy = held | open_syms
    slots = cfg.max_open_positions - len(busy)
    room = cfg.max_trades_per_day - len(bought_today)
    trader.note(f"Positions/open orders: {len(busy)} | buys today: {len(bought_today)} | slots: {max(slots, 0)}")
    if slots <= 0 or room <= 0:
        trader.write_summary()
        return 0

    seeds = sorted(set(SEEDS) | {t.strip().upper() for t in os.getenv("EXTRA_TICKERS", "").split(",") if t.strip()})
    data = Data(key, secret, cfg)
    syms, scanned = data.universe(seeds)
    log.info("Universe: %d symbols (%d from scanner)", len(syms), len(scanned))

    signals: list[Signal] = []
    for ctx in data.contexts(syms, now, scanned):
        if ctx.symbol in busy or ctx.symbol in bought_today:
            continue
        if ctx.screened and not ctx.rvol >= cfg.screen_min_rvol:
            continue
        sig = evaluate(ctx, cfg)
        if sig:
            signals.append(sig)

    trader.note(f"Signals found: {len(signals)}")
    signals.sort(key=lambda s: -(s.rvol if s.rvol == s.rvol else 0))
    for sig in signals[: min(slots, room)]:
        trader.submit(sig, acct)
    trader.write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
