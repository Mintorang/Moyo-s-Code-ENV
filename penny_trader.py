#!/usr/bin/env python3
"""
penny_trader.py - ONE-SHOT penny-stock bot for GitHub Actions.

Each run:
  check market clock -> cancel stale BOT-OWNED entries -> (flatten bot-owned
  positions near the close) -> scan seeds + live movers -> look back over the
  last N minutes of 1-min bars -> place Alpaca BRACKET orders.

Bot orders are tagged with a `pt_<strategy>_<symbol>_<YYYYMMDDHH>` client_order_id
prefix. This gives us (a) deterministic duplicate rejection, (b) the ability to
flatten ONLY positions the bot opened, and (c) a join key for monitor.py.

Key changes vs. the prior version (see review):
  - strict env_bool (fail loudly on bad values)
  - 1-minute grid reindex + bar-density gate before any indicator math
  - prior close from daily bars (not the last IEX 1-min print)
  - RVOL uses today's RTH volume (not premarket) and skips thin sessions
  - bot-owned order tagging, scoped flatten
  - capital-aware sizing with a running budget (no BP double-spend)
  - deterministic coid; log only after acceptance
  - per-symbol cooldown after cancel/reject
  - float cache (JSON, 7-day TTL) instead of live yfinance per run
  - per-strategy kill switch driven by state/ledger.csv (from monitor.py)
  - removed the obsolete daytrade_count PDT guard
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed, MostActivesBy
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (
    MarketMoversRequest,
    MostActivesRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
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
RTH_START_MOD = 9 * 60 + 30
RTH_END_MOD = 15 * 60 + 59
BOT_PREFIX = "pt_"
LEDGER_PATH = "state/ledger.csv"
FLOAT_CACHE_PATH = "state/float_cache.json"
log = logging.getLogger("penny")

SEED_HIGH_LIQUIDITY = ["PLUG", "OPEN", "NIO", "RIG"]
SEED_LOW_FLOAT_MOMO = ["SOUN", "BBAI", "RGTI", "SDA", "SNDL"]
SEED_DIP_BUY = ["ACRS", "STTK", "DBGI"]
SEEDS = SEED_HIGH_LIQUIDITY + SEED_LOW_FLOAT_MOMO + SEED_DIP_BUY


# --------------------------------------------------------------------------- env / config
def env(name: str, default):
    v = os.getenv(name)
    return default if v in (None, "") else type(default)(v)


def env_bool(name: str, default: bool) -> bool:
    """Strict boolean parsing: never silently treat 'false' as truthy."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name}={raw!r} is not a valid boolean")


def coid(strategy: str, symbol: str, ts: datetime) -> str:
    """Deterministic client_order_id; hour-precision bucket prevents same-run
    double-submission but still allows an hourly retry after a cancel/reject."""
    s = f"{BOT_PREFIX}{strategy}_{symbol}_{ts:%Y%m%d%H}"
    assert len(s) <= 48, f"client_order_id too long: {s} ({len(s)} > 48)"
    return s


def is_bot_order(o) -> bool:
    return bool(o.client_order_id and o.client_order_id.startswith(BOT_PREFIX))


def strategy_from_coid(c: Optional[str]) -> Optional[str]:
    if not c or not c.startswith(BOT_PREFIX):
        return None
    parts = c.split("_")
    if len(parts) < 4:
        return None
    return "_".join(parts[1:-2])


@dataclass
class Config:
    # universe
    min_price: float = 0.50
    max_price: float = 5.00
    max_scanned: int = 30
    screen_min_volume: int = 1_000_000
    screen_min_rvol: float = 2.0
    # gap & go
    gap_min_pct: float = 4.0
    rvol_min: float = 3.0
    max_float: float = 20_000_000
    allow_unknown_float: bool = False      # B: fail closed on unknown float
    gap_orb_minutes: int = 5
    # ORB
    orb_minutes: int = 15
    orb_vol_mult: float = 1.5
    # VWAP momentum
    vwap_vol_mult: float = 2.0
    # panic dip
    rsi_oversold: float = 25.0
    dip_min_drop_pct: float = 8.0
    dip_vol_mult: float = 2.0
    # hourly-run behaviour
    lookback: int = 30
    max_chase_pct: float = 3.0
    entry_slip_pct: float = 0.75
    stale_entry_minutes: int = 30
    retry_cooldown_minutes: int = 20       # B: cooldown after cancel/reject
    flatten_minutes: int = 35
    # data quality
    min_bar_density: float = 0.60
    min_volume_for_rvol: int = 10_000      # IEX-only threshold
    max_spread_pct: float = 3.0            # entry spread gate; None to disable
    # risk
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 10.0
    max_open_positions: int = 3
    max_trades_per_day: int = 6
    daily_loss_limit_pct: float = 3.0
    reward_risk: float = 1.5
    max_stop_pct: float = 8.0
    # kill switch
    kill_min_trades: int = 10
    kill_min_net_r_sum: float = -4.0
    kill_max_dd_r: float = -5.0

    def __post_init__(self):
        self.risk_per_trade_pct = env("RISK_PER_TRADE_PCT", self.risk_per_trade_pct)
        self.max_open_positions = env("MAX_OPEN_POSITIONS", self.max_open_positions)
        self.reward_risk = env("REWARD_RISK", self.reward_risk)
        self.min_bar_density = env("MIN_BAR_DENSITY", self.min_bar_density)
        self.min_volume_for_rvol = env("MIN_VOLUME_FOR_RVOL", self.min_volume_for_rvol)
        iv = env("RUN_INTERVAL_MIN", 60)
        # Cap the acceptance window so an hourly schedule doesn't resurrect
        # signals from 90 minutes ago on every run.
        self.lookback = min(int(iv * 1.5 + 5), 30)
        self.stale_entry_minutes = max(10, int(iv * 1.5))
        self.flatten_minutes = 35 if iv >= 30 else 20
        self.retry_cooldown_minutes = max(10, iv // 3)

        # G: strict config validation
        if not self.min_price < self.max_price:
            raise ValueError(f"min_price {self.min_price} >= max_price {self.max_price}")
        if not (0 < self.risk_per_trade_pct <= self.max_position_pct <= 100):
            raise ValueError("need 0 < risk_per_trade_pct <= max_position_pct <= 100")
        if self.reward_risk < 1.0:
            raise ValueError("reward_risk must be >= 1.0")
        if self.daily_loss_limit_pct <= 0:
            raise ValueError("daily_loss_limit_pct must be positive")
        if not (0 < self.min_bar_density <= 1.0):
            raise ValueError("min_bar_density must be in (0, 1]")


@dataclass
class Signal:
    symbol: str
    strategy: str
    entry: float
    stop: float
    target: float
    rvol: float
    reason: str
    coid: Optional[str] = None


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


def to_minute_grid(rth: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Reindex to a clean 1-minute grid so rolling() actually rolls over
    minutes, not over sparse bars. Returns (grid, density)."""
    if rth is None or rth.empty:
        return rth, 0.0
    start = rth.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
    end = rth.index[-1]
    if end <= start:
        return rth, 0.0
    grid = pd.date_range(start, end, freq="min", tz=rth.index.tz)
    g = rth.reindex(grid)
    density = float(g.close.notna().mean())
    g["close"] = g.close.ffill()
    for c in ("open", "high", "low"):
        g[c] = g[c].fillna(g.close)
    g["volume"] = g.volume.fillna(0)
    g = g.dropna(subset=["close"])
    g["mod"] = g.index.hour * 60 + g.index.minute
    return g, density


def last_true(mask: pd.Series, lookback: int) -> Optional[int]:
    m = mask.fillna(False).astype(bool).iloc[-lookback:]
    idx = np.flatnonzero(m.values)
    return len(mask) - len(m) + int(idx[-1]) if len(idx) else None


# --------------------------------------------------------------------------- strategies
def gap_and_go(c: Ctx, cfg: Config):
    r = c.rth
    if c.gap_pct < cfg.gap_min_pct or not c.rvol >= cfg.rvol_min:
        return None
    if c.flt is None:
        if not cfg.allow_unknown_float:
            return None  # fail closed
    elif c.flt > cfg.max_float:
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
    return ("PanicDip", i, stop, float(r.vwap.iloc[-1]),
            f"drop {drop.iloc[i]:.1f}% RSI<25 reversal, target VWAP")


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
            log.info("%s %s trigger found but price already ran > %.1f%%",
                     c.symbol, name, cfg.max_chase_pct)
            continue
        stop = max(float(stop), entry * (1 - cfg.max_stop_pct / 100))
        stop = min(stop, entry * 0.99)
        if not stop < entry:
            continue
        risk = entry - stop
        if target is None:
            target = entry + cfg.reward_risk * risk
        elif target - entry < risk:
            continue
        return Signal(c.symbol, name, entry, stop, float(target), c.rvol, reason)
    return None


# --------------------------------------------------------------------------- state helpers
class FloatCache:
    def __init__(self, path: str = FLOAT_CACHE_PATH, ttl_days: int = 7):
        self.path = path
        self.ttl_days = ttl_days
        self.data: dict[str, dict] = {}
        self.dirty = False
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self.data = json.load(f)
        except Exception as e:
            log.warning("float cache load failed: %s", e)

    def get(self, sym: str) -> Optional[float]:
        entry = self.data.get(sym)
        if not entry:
            return None
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(entry["ts"])).days
        except Exception:
            return None
        if age > self.ttl_days:
            return None
        return entry.get("float")

    def put(self, sym: str, val: Optional[float]):
        self.data[sym] = {"float": val, "ts": datetime.now(timezone.utc).isoformat()}
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            log.warning("float cache save failed: %s", e)


@dataclass
class Budget:
    cash: float

    @classmethod
    def from_account(cls, acct, open_buy_orders) -> "Budget":
        committed = 0.0
        for o in open_buy_orders:
            try:
                committed += float(o.qty or 0) * float(o.limit_price or 0)
            except Exception:
                pass
        base = min(float(acct.buying_power or 0), float(acct.equity or 0))
        return cls(cash=max(base - committed, 0.0))

    def reserve(self, notional: float) -> bool:
        if notional > self.cash:
            return False
        self.cash -= notional
        return True


# --------------------------------------------------------------------------- data layer
class Data:
    def __init__(self, key: str, secret: str, cfg: Config):
        self.cfg = cfg
        self.hist = StockHistoricalDataClient(key, secret)
        self.screener = ScreenerClient(key, secret)
        self.profiles: dict[str, Optional[dict]] = {}
        self.float_cache = FloatCache()

    def bars(self, symbols, start, end) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        syms = list(symbols)
        for k in range(0, len(syms), 40):
            chunk = syms[k:k + 40]
            try:
                req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                       start=start, end=end, feed=DataFeed.IEX)
                df = self.hist.get_stock_bars(req).df
            except Exception as e:
                log.warning("bars fetch failed for %s: %s", chunk[:3], e)
                continue
            if df.empty:
                continue
            present = set(df.index.get_level_values(0))
            for s in chunk:
                if s in present:
                    d = df.xs(s, level=0).copy()
                    d.index = d.index.tz_convert(ET)
                    out[s] = d
        return out

    def daily_bars(self, symbols, start, end) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        syms = list(symbols)
        for k in range(0, len(syms), 40):
            chunk = syms[k:k + 40]
            try:
                req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                       start=start, end=end, feed=DataFeed.IEX)
                df = self.hist.get_stock_bars(req).df
            except Exception as e:
                log.warning("daily bars failed for %s: %s", chunk[:3], e)
                continue
            if df.empty:
                continue
            present = set(df.index.get_level_values(0))
            for s in chunk:
                if s in present:
                    d = df.xs(s, level=0).copy()
                    if not d.empty:
                        try:
                            d.index = d.index.tz_convert(ET)
                        except TypeError:
                            d.index = d.index.tz_localize("UTC").tz_convert(ET)
                        out[s] = d
        return out

    def universe(self, seeds: list[str]) -> tuple[list[str], set[str]]:
        cfg, scanned = self.cfg, set()
        try:
            for g in self.screener.get_market_movers(MarketMoversRequest(top=50)).gainers:
                if cfg.min_price <= g.price <= cfg.max_price:
                    scanned.add(g.symbol)
        except Exception as e:
            log.warning("movers screener failed (free tier?): %s", e)
        try:
            acts = self.screener.get_most_actives(MostActivesRequest(by=MostActivesBy.VOLUME, top=50))
            scanned |= {a.symbol for a in acts.most_actives if a.volume >= cfg.screen_min_volume}
        except Exception as e:
            log.warning("most-actives screener failed: %s", e)
        extra = sorted(scanned - set(seeds))[: cfg.max_scanned]
        return list(seeds) + extra, set(extra)

    def _float(self, sym: str) -> Optional[float]:
        cached = self.float_cache.get(sym)
        if cached is not None:
            return cached
        # A previous lookup failure is also cached (as `None`) — avoid hammering
        if sym in self.float_cache.data:
            return None
        try:
            import yfinance as yf
            f = yf.Ticker(sym).info.get("floatShares")
            val = float(f) if f else None
        except Exception as e:
            log.warning("float lookup failed for %s: %s", sym, e)
            val = None
        self.float_cache.put(sym, val)
        return val

    def _profiles(self, syms, now):
        need = [s for s in syms if s not in self.profiles]
        if not need:
            return
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        intraday = self.bars(need, midnight - timedelta(days=10), midnight)
        daily = self.daily_bars(need, midnight - timedelta(days=10), midnight)

        for s in need:
            d = intraday.get(s)
            dl = daily.get(s)
            if d is None or d.empty or dl is None or dl.empty:
                self.profiles[s] = None
                continue
            d = d.assign(date=d.index.date, mod=d.index.hour * 60 + d.index.minute)
            # B: exclude likely half-days from the RVOL baseline
            curves = []
            for _, g in d.groupby("date"):
                gr = g[(g["mod"] >= RTH_START_MOD) & (g["mod"] <= RTH_END_MOD)]
                if gr.empty:
                    continue
                if gr["mod"].iloc[-1] < 15 * 60 + 55:  # didn't reach 15:55
                    continue
                curve = gr.groupby("mod")["volume"].sum().reindex(range(240, 960), fill_value=0).cumsum()
                curves.append(curve)
            if not curves:
                self.profiles[s] = None
                continue
            prior_close = float(dl.close.iloc[-1])
            self.profiles[s] = {
                "avg_curve": pd.concat(curves, axis=1).mean(axis=1),
                "prior_close": prior_close,
            }

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

            # B: reindex to minute grid and gate on density
            rth, density = to_minute_grid(rth)
            if density < cfg.min_bar_density:
                skipped[s] = f"bar density {density:.0%} < {cfg.min_bar_density:.0%}"
                continue
            if rth.empty or len(rth) < 20:
                skipped[s] = "grid too short"
                continue

            rth = add_indicators(rth)
            gap = (float(rth.open.iloc[0]) - p["prior_close"]) / p["prior_close"] * 100
            mod_now = int(rth.index[-1].hour * 60 + rth.index[-1].minute)

            # B: RVOL uses today's RTH-only volume, not premarket
            today_vol = float(rth.volume.sum())
            if today_vol < cfg.min_volume_for_rvol:
                skipped[s] = f"rth volume {today_vol:.0f} < {cfg.min_volume_for_rvol}"
                continue
            exp = p["avg_curve"].get(mod_now, np.nan)
            rvol = float(today_vol / exp) if exp and exp > 0 else float("nan")

            flt = self._float(s) if gap >= cfg.gap_min_pct else None
            ctxs.append(Ctx(s, rth, gap, rvol, flt, s in scanned))
            log.info("  %-6s $%.2f gap %+5.1f%% rvol %4.1f bars %d dens %.0f%%",
                     s, price, gap, rvol, len(rth), density * 100)

        if skipped:
            log.info("Skipped: %s", "; ".join(f"{k} ({v})" for k, v in skipped.items()))
        return ctxs


# --------------------------------------------------------------------------- execution
def px(p: float) -> float:
    return round(p, 2) if p >= 1 else round(p, 4)


def strategy_enabled(name: str, cfg: Config) -> bool:
    """Kill switch driven by monitor.py's ledger."""
    if not os.path.exists(LEDGER_PATH):
        return True
    try:
        df = pd.read_csv(LEDGER_PATH)
    except Exception:
        return True
    if "strategy" not in df.columns or "net_r" not in df.columns:
        return True
    r = df[df.strategy == name].dropna(subset=["net_r"]).net_r.tail(20)
    if len(r) < cfg.kill_min_trades:
        return True
    if float(r.sum()) <= cfg.kill_min_net_r_sum:
        log.warning("KILL: %s disabled — sum(net_r over %d) = %.2f", name, len(r), r.sum())
        return False
    running = r.cumsum() - r.cumsum().cummax()
    if float(running.min()) <= cfg.kill_max_dd_r:
        log.warning("KILL: %s disabled — max drawdown %.2fR over %d trades",
                    name, running.min(), len(r))
        return False
    return True


class Trader:
    def __init__(self, cfg: Config, tc: TradingClient, hist: StockHistoricalDataClient, dry: bool):
        self.cfg, self.tc, self.hist, self.dry = cfg, tc, hist, dry
        self.events: list[str] = []

    def note(self, msg: str):
        log.info(msg)
        self.events.append(msg)

    def daily_loss_hit(self, acct) -> bool:
        last = float(acct.last_equity or 0)
        pnl = (float(acct.equity) - last) / last * 100 if last else 0
        self.note(f"Equity ${float(acct.equity):,.2f} (day {pnl:+.2f}%)")
        return pnl <= -self.cfg.daily_loss_limit_pct

    def compute_state(self) -> tuple[set[str], set[str]]:
        """Return (filled_buy_symbols_today, blocked_symbols)."""
        midnight = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
        now = datetime.now(timezone.utc)
        try:
            orders = self.tc.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.ALL, after=midnight, limit=500))
        except Exception as e:
            log.warning("order history fetch failed: %s", e)
            return set(), set()
        filled = {o.symbol for o in orders
                  if o.side == OrderSide.BUY and float(o.filled_qty or 0) > 0}
        blocked = set(filled)
        for o in orders:
            if o.side != OrderSide.BUY:
                continue
            if float(o.filled_qty or 0) > 0:
                continue
            if o.status in ("rejected", "canceled") and o.submitted_at:
                age = (now - o.submitted_at).total_seconds() / 60
                if age < self.cfg.retry_cooldown_minutes:
                    blocked.add(o.symbol)
        return filled, blocked

    def bot_symbols(self, days: int = 5) -> set[str]:
        try:
            orders = self.tc.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=datetime.now(ET) - timedelta(days=days), limit=500))
        except Exception as e:
            log.warning("bot_symbols fetch failed: %s", e)
            return set()
        return {o.symbol for o in orders
                if is_bot_order(o) and o.side == OrderSide.BUY
                and float(o.filled_qty or 0) > 0}

    def cancel_stale_entries(self):
        now = datetime.now(timezone.utc)
        try:
            orders = self.tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception as e:
            log.warning("open orders fetch failed: %s", e)
            return
        for o in orders:
            if not is_bot_order(o):
                continue
            age = (now - o.submitted_at).total_seconds() / 60 if o.submitted_at else 0
            if o.side == OrderSide.BUY and float(o.filled_qty or 0) == 0 \
                    and age > self.cfg.stale_entry_minutes:
                self.note(f"Cancelling stale bot entry {o.symbol} ({age:.0f} min old)")
                if not self.dry:
                    try:
                        self.tc.cancel_order_by_id(o.id)
                    except Exception as e:
                        log.warning("cancel failed: %s", e)

    def flatten(self):
        # C: only flatten positions the bot opened
        held = {p.symbol for p in self.tc.get_all_positions()}
        mine = self.bot_symbols() & held
        self.note(f"Near close: flattening {len(mine)} bot-owned position(s): {sorted(mine)}")
        if self.dry or not mine:
            return
        for sym in mine:
            try:
                self.tc.close_position(sym, cancel_orders=True)
            except Exception as e:
                log.error("flatten %s failed: %s", sym, e)

    def _spread_pct(self, symbol: str) -> Optional[float]:
        try:
            q = self.hist.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            if bid > 0 and ask > bid:
                return (ask - bid) / ((ask + bid) / 2) * 100
        except Exception:
            pass
        return None

    def submit(self, s: Signal, acct, budget: Budget) -> bool:
        cfg = self.cfg
        equity = float(acct.equity)

        # D: log the spread we actually face at entry
        spread = self._spread_pct(s.symbol)
        if spread is not None and spread > cfg.max_spread_pct:
            self.note(f"{s.symbol}: {s.strategy} skipped — spread {spread:.2f}% > {cfg.max_spread_pct}%")
            return False

        limit = s.entry * (1 + cfg.entry_slip_pct / 100)
        stop = min(s.stop, limit - 0.01)
        target = max(s.target, limit + 0.01)

        # B: capital-aware sizing with running budget
        qty_by_risk = int(equity * cfg.risk_per_trade_pct / 100 / (limit - stop))
        qty_by_pct = int(equity * cfg.max_position_pct / 100 / limit)
        qty_by_cash = int(budget.cash * 0.95 / limit)
        qty = min(qty_by_risk, qty_by_pct, qty_by_cash)
        if qty < 1:
            self.note(f"{s.symbol}: {s.strategy} sizing floors to 0 "
                      f"(cash ${budget.cash:.2f}, limit ${limit:.4f})")
            return False

        notional = qty * limit
        if not budget.reserve(notional):
            self.note(f"{s.symbol}: budget reserve failed (need ${notional:.2f})")
            return False

        c = coid(s.strategy, s.symbol, datetime.now(ET))
        s.coid = c
        self.note(f"BUY {s.symbol} x{qty} limit {px(limit)} stop {px(stop)} "
                  f"target {px(target)} [{s.strategy}] {s.reason} "
                  f"spread={f'{spread:.2f}%' if spread is not None else 'n/a'}")

        if self.dry:
            self._csv(s, qty, limit, stop, target, spread)
            return True

        try:
            o = self.tc.submit_order(LimitOrderRequest(
                symbol=s.symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                limit_price=px(limit), order_class=OrderClass.BRACKET,
                client_order_id=c,
                take_profit=TakeProfitRequest(limit_price=px(target)),
                stop_loss=StopLossRequest(stop_price=px(stop))))
            # B: log only after acceptance
            self.note(f"  -> order accepted id={o.id} coid={c} status={o.status}")
            self._csv(s, qty, limit, stop, target, spread)
            return True
        except Exception as e:
            self.note(f"  -> ORDER REJECTED for {s.symbol}: {e}")
            budget.reserve(-notional)  # release the reservation
            return False

    @staticmethod
    def _csv(s: Signal, qty, limit, stop, target, spread):
        new = not os.path.exists("signals.csv")
        with open("signals.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "strategy", "qty", "limit", "stop",
                            "target", "reason", "client_order_id", "spread_pct"])
            w.writerow([datetime.now(ET).isoformat(timespec="seconds"), s.symbol,
                        s.strategy, qty, px(limit), px(stop), px(target), s.reason,
                        s.coid or "", "" if spread is None else f"{spread:.3f}"])

    def write_summary(self):
        path = os.getenv("GITHUB_STEP_SUMMARY")
        if not path:
            return
        with open(path, "a") as f:
            f.write("### Penny trader run\n")
            for e in self.events:
                f.write(f"- {e}\n")
            if os.path.exists(LEDGER_PATH):
                try:
                    df = pd.read_csv(LEDGER_PATH).tail(5)
                    if not df.empty:
                        f.write("\n#### Last 5 closed trades\n")
                        f.write("| Strategy | Symbol | Entry | Exit | Reason | PnL% |\n")
                        f.write("|---|---|---|---|---|---|\n")
                        for _, row in df.iterrows():
                            ep = float(row.get("entry_px", 0) or 0)
                            xp = float(row.get("exit_px", 0) or 0)
                            pp = float(row.get("pnl_pct", 0) or 0)
                            f.write(f"| {row.get('strategy','?')} | {row.get('symbol','?')} "
                                    f"| {ep:.4f} | {xp:.4f} | {row.get('exit_reason','?')} "
                                    f"| {pp:+.2f} |\n")
                except Exception:
                    pass


# --------------------------------------------------------------------------- one run
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY missing.")
        return 1
    paper = env_bool("ALPACA_PAPER", True)
    dry = env_bool("DRY_RUN", False)
    cfg = Config()
    tc = TradingClient(key, secret, paper=paper)
    data = Data(key, secret, cfg)
    trader = Trader(cfg, tc, data.hist, dry)
    trader.note(f"Mode: {'PAPER' if paper else 'LIVE'}"
                f"{' (dry-run: no orders sent)' if dry else ''}")

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
    # B: removed obsolete daytrade_count PDT guard (Alpaca discontinued it)

    filled_today, blocked = trader.compute_state()
    held = {p.symbol for p in tc.get_all_positions()}
    open_syms = {o.symbol for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    occupied = held | open_syms
    busy = occupied | blocked
    slots = cfg.max_open_positions - len(occupied)
    room = cfg.max_trades_per_day - len(filled_today)
    trader.note(f"Positions/open orders: {len(occupied)} | buys filled today: "
                f"{len(filled_today)} | slots: {max(slots, 0)} | room: {max(room, 0)}")
    if slots <= 0 or room <= 0:
        trader.write_summary()
        return 0

    seeds = sorted(set(SEEDS) | {t.strip().upper() for t in os.getenv("EXTRA_TICKERS", "").split(",") if t.strip()})
    syms, scanned = data.universe(seeds)
    log.info("Universe: %d symbols (%d from scanner)", len(syms), len(scanned))

    signals: list[Signal] = []
    for ctx in data.contexts(syms, now, scanned):
        if ctx.symbol in busy or ctx.symbol in filled_today:
            continue
        if ctx.screened and not ctx.rvol >= cfg.screen_min_rvol:
            continue
        sig = evaluate(ctx, cfg)
        if sig and strategy_enabled(sig.strategy, cfg):
            signals.append(sig)

    # dedupe by symbol (evaluate returns the first matching strategy per symbol,
    # but a future change could return more than one)
    seen = set()
    unique: list[Signal] = []
    for s in signals:
        if s.symbol in seen:
            continue
        seen.add(s.symbol)
        unique.append(s)
    signals = unique

    trader.note(f"Signals found: {len(signals)}")
    signals.sort(key=lambda s: -(s.rvol if s.rvol == s.rvol else 0))

    open_buy_orders = [o for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
                       if o.side == OrderSide.BUY]
    budget = Budget.from_account(acct, open_buy_orders)

    opened = 0
    for sig in signals:
        if opened >= min(slots, room):
            break
        if trader.submit(sig, acct, budget):
            opened += 1

    data.float_cache.save()
    trader.write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
