#!/usr/bin/env python3
"""
monitor.py - Fills-based P&L ledger for penny_trader.

Runs after penny_trader.py each session. Pulls recent Alpaca orders, matches
each bot-parented BUY bracket (client_order_id starts with `pt_`) to its exit
leg (target, stop, or a later flatten SELL on the same symbol), appends deduped
rows to state/ledger.csv, writes strategy_stats.json, and appends a daily
summary to $GITHUB_STEP_SUMMARY.

Idempotent: deduped on the parent order id. Safe to run on every cycle.

Usage:
    python monitor.py
    python monitor.py --days 14 --verbose

Env: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

LEDGER_PATH = "state/ledger.csv"
STATS_PATH = "state/strategy_stats.json"
BOT_PREFIX = "pt_"

LEDGER_COLUMNS = [
    "order_id", "symbol", "strategy", "qty",
    "opened_at", "closed_at",
    "entry_px", "exit_px", "exit_reason",
    "pnl_pct", "risk_pct", "net_r", "holding_min",
    "client_order_id",
]

log = logging.getLogger("monitor")


# --------------------------------------------------------------------------- env
def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name}={raw!r} is not a valid boolean")


# --------------------------------------------------------------------------- order matching
def _filled_qty(o) -> float:
    try:
        return float(o.filled_qty or 0)
    except Exception:
        return 0.0


def _filled_px(o) -> float:
    try:
        return float(o.filled_avg_price or 0)
    except Exception:
        return 0.0


def _parse_strategy(coid: Optional[str]) -> Optional[str]:
    if not coid or not coid.startswith(BOT_PREFIX):
        return None
    parts = coid.split("_")
    if len(parts) < 4:
        return None
    return "_".join(parts[1:-2])


def _leg_reason(leg) -> str:
    """Best-effort tag for which bracket leg fired."""
    raw = str(getattr(leg, "order_type", "") or "").lower()
    if "stop" in raw:
        return "stop"
    if "limit" in raw or "take_profit" in raw:
        return "target"
    return "leg"


def match_exits(parents, all_orders):
    """For each filled parent, find the exit order + reason.

    Order of preference:
      1. A filled child leg (bracket target or stop).
      2. The earliest unmatched filled SELL on the same symbol after the
         parent's fill time (i.e. a flatten sell).
    Parents without an exit yet are skipped (still open).
    """
    sells = [o for o in all_orders
             if o.side == OrderSide.SELL and _filled_qty(o) > 0 and o.filled_at]
    sells.sort(key=lambda o: o.filled_at)
    used_sell_ids: set[str] = set()

    results = []
    for p in parents:
        if _filled_qty(p) == 0:
            continue
        exit_order = None
        reason = None
        for leg in (getattr(p, "legs", None) or []):
            if _filled_qty(leg) > 0:
                exit_order = leg
                reason = _leg_reason(leg)
                break
        if exit_order is None and p.filled_at:
            for sell in sells:
                if sell.id in used_sell_ids:
                    continue
                if sell.symbol != p.symbol:
                    continue
                if sell.filled_at and sell.filled_at > p.filled_at:
                    exit_order = sell
                    reason = "flatten"
                    used_sell_ids.add(sell.id)
                    break
        if exit_order is None:
            continue  # still open — skip
        results.append((p, exit_order, reason))
    return results


def build_rows(pairs) -> list[dict]:
    rows = []
    for parent, exit_order, reason in pairs:
        entry_px = _filled_px(parent)
        exit_px = _filled_px(exit_order)
        if entry_px <= 0 or exit_px <= 0:
            continue
        pnl_pct = (exit_px - entry_px) / entry_px * 100

        # Risk from the child stop, if we can find it
        risk_pct = None
        for leg in (getattr(parent, "legs", None) or []):
            if _leg_reason(leg) == "stop":
                stop_px = float(getattr(leg, "stop_price", 0) or 0)
                if stop_px > 0 and stop_px < entry_px:
                    risk_pct = (entry_px - stop_px) / entry_px * 100
                break
        if risk_pct is None or risk_pct <= 0:
            # Fallback: assume the max_stop_pct the bot could have used (8%)
            risk_pct = 8.0

        net_r = pnl_pct / risk_pct if risk_pct else None
        holding_min = None
        if parent.filled_at and exit_order.filled_at:
            holding_min = (exit_order.filled_at - parent.filled_at).total_seconds() / 60

        rows.append({
            "order_id": str(parent.id),
            "symbol": parent.symbol,
            "strategy": _parse_strategy(parent.client_order_id) or "unknown",
            "qty": _filled_qty(parent),
            "opened_at": parent.filled_at.isoformat() if parent.filled_at else "",
            "closed_at": exit_order.filled_at.isoformat() if exit_order.filled_at else "",
            "entry_px": round(entry_px, 6),
            "exit_px": round(exit_px, 6),
            "exit_reason": reason or "unknown",
            "pnl_pct": round(pnl_pct, 4),
            "risk_pct": round(risk_pct, 4),
            "net_r": round(net_r, 4) if net_r is not None else None,
            "holding_min": round(holding_min, 1) if holding_min is not None else None,
            "client_order_id": parent.client_order_id or "",
        })
    return rows


# --------------------------------------------------------------------------- ledger I/O
def load_ledger(path: str = LEDGER_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning("ledger load failed: %s", e)
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return df


def append_ledger(rows: list[dict], path: str = LEDGER_PATH) -> int:
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = load_ledger(path)
    existing = set(df["order_id"].astype(str)) if not df.empty else set()
    new_rows = [r for r in rows if str(r["order_id"]) not in existing]
    if not new_rows:
        return 0
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return len(new_rows)


# --------------------------------------------------------------------------- stats + summary
def compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    stats: dict[str, dict] = {}
    for name, g in df.groupby("strategy"):
        r = g.dropna(subset=["net_r"]).net_r.astype(float)
        if r.empty:
            continue
        wins = r[r > 0]
        losses = r[r < 0]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        win_rate = float((r > 0).mean())
        # Expectancy in R
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        stats[name] = {
            "trades": int(len(r)),
            "win_rate": round(win_rate, 4),
            "avg_win_r": round(avg_win, 4),
            "avg_loss_r": round(avg_loss, 4),
            "expectancy_r": round(expectancy, 4),
            "total_r": round(float(r.sum()), 4),
            "max_dd_r": round(float((r.cumsum() - r.cumsum().cummax()).min()), 4),
        }
    return stats


def save_stats(stats: dict, path: str = STATS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def write_daily_summary(df: pd.DataFrame, new_count: int):
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    today = datetime.now(timezone.utc).date()
    with open(path, "a") as f:
        f.write("### P&L monitor\n")
        f.write(f"- Rows in ledger: {len(df)} (added {new_count} this run)\n")
        if df.empty:
            return
        # Today's trades (closed_at starts with today's date, UTC)
        try:
            df = df.copy()
            df["closed_date"] = pd.to_datetime(df.closed_at, errors="coerce", utc=True).dt.date
            today_df = df[df.closed_date == today]
        except Exception:
            today_df = pd.DataFrame()

        if not today_df.empty:
            f.write("\n#### Today\n")
            f.write(f"- Trades: {len(today_df)}\n")
            f.write(f"- Total R: {today_df.net_r.fillna(0).sum():+.2f}\n")
            f.write(f"- Win rate: {(today_df.net_r.fillna(0) > 0).mean():.0%}\n")

        stats = compute_stats(df)
        if stats:
            f.write("\n#### Per-strategy (all time)\n")
            f.write("| Strategy | Trades | Win% | Exp (R) | Total R | Max DD (R) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for name, s in sorted(stats.items(), key=lambda x: -x[1]["total_r"]):
                f.write(f"| {name} | {s['trades']} | {s['win_rate']:.0%} | "
                        f"{s['expectancy_r']:+.3f} | {s['total_r']:+.2f} | "
                        f"{s['max_dd_r']:+.2f} |\n")


# --------------------------------------------------------------------------- main
def update_ledger(tc: TradingClient, days_back: int = 7) -> int:
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        orders = tc.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL, after=since, limit=500, nested=True))
    except Exception as e:
        log.error("order fetch failed: %s", e)
        return 0

    parents = [o for o in orders
               if o.side == OrderSide.BUY
               and o.client_order_id
               and o.client_order_id.startswith(BOT_PREFIX)]
    pairs = match_exits(parents, orders)
    rows = build_rows(pairs)
    new_count = append_ledger(rows)
    log.info("Ledger: %d parents, %d matched exits, %d new rows",
             len(parents), len(pairs), new_count)

    df = load_ledger()
    stats = compute_stats(df)
    save_stats(stats)
    write_daily_summary(df, new_count)
    return new_count


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="lookback days (default 7)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY missing.")
        return 1
    paper = env_bool("ALPACA_PAPER", True)
    tc = TradingClient(key, secret, paper=paper)
    update_ledger(tc, days_back=args.days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
