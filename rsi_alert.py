#!/usr/bin/env python3
"""
RSI alert checker.

Reads tickers from a JSON watchlist, computes Wilder's RSI(14) on:
  - the 1Y chart  -> daily candles
  - the 5Y chart  -> weekly candles
and sends you a Telegram message when a stock's status changes between:
  oversold (RSI <= 30) | close to oversold (30-35) | normal (35-65)
  | close to overbought (65-70) | overbought (RSI >= 70)

To avoid a message every hour while a stock sits in a zone, the last known
zone per ticker/timeframe is stored in a state file; you are only alerted
when the zone changes.

Setup:
    pip install yfinance pandas

Environment variables (for Telegram):
    TELEGRAM_BOT_TOKEN  token from @BotFather
    TELEGRAM_CHAT_ID    your chat id (message your bot first, then see README)

Usage:
    python rsi_alert.py --watchlist watchlist.json
    python rsi_alert.py --watchlist watchlist.json --dry-run
"""

import argparse
import html
import json
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

# yfinance still uses pandas' deprecated `inplace=` internally, which newer
# pandas versions warn about on every call. They are harmless, so hide them.


import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=r".*inplace.*")
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")

load_dotenv(Path(__file__).resolve().parent / ".env")

RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
NEAR_OVERSOLD = 35     # 30 < RSI <= 35  -> "close to oversold"
NEAR_OVERBOUGHT = 65   # 65 <= RSI < 70  -> "close to overbought"

# Human-readable names for each zone (used in messages)
ZONE_LABEL = {
    "oversold": "oversold",
    "near_oversold": "close to oversold",
    "neutral": "normal",
    "near_overbought": "close to overbought",
    "overbought": "overbought",
}

# label -> (yfinance period, yfinance interval)
TIMEFRAMES = {
    "1Y": ("1y", "1d"),   # 1-year chart, daily candles
    "5Y": ("5y", "1wk"),  # 5-year chart, weekly candles
}

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "rsi_state.json"

log = logging.getLogger("rsi_alert")


def load_watchlist(path: Path) -> list[str]:
    """Accepts either ["AAPL", "NVDA"] or {"tickers": ["AAPL", "NVDA"]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tickers = data.get("tickers", []) if isinstance(data, dict) else data
    cleaned = []
    for t in tickers:
        if isinstance(t, str) and t.strip():
            cleaned.append(t.strip().upper())
    return list(dict.fromkeys(cleaned))  # dedupe, keep order


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    return rsi


def zone_for(rsi: float) -> str:
    if rsi <= OVERSOLD:
        return "oversold"
    if rsi >= OVERBOUGHT:
        return "overbought"
    if rsi <= NEAR_OVERSOLD:
        return "near_oversold"
    if rsi >= NEAR_OVERBOUGHT:
        return "near_overbought"
    return "neutral"


def fetch_latest_rsi(ticker: str, period: str, interval: str):
    """Returns (rsi, last_close) or None if data is unavailable."""
    hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    close = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
    if len(close) < RSI_PERIOD * 2:
        return None
    rsi = compute_rsi(close).iloc[-1]
    if pd.isna(rsi):
        return None
    return float(rsi), float(close.iloc[-1])


def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_FILE)


def _chunks(text: str, limit: int = 4000):
    chunk = ""
    for line in text.split("\n"):
        if chunk and len(chunk) + len(line) + 1 > limit:
            yield chunk
            chunk = line
        else:
            chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        yield chunk


def plain(text: str) -> str:
    """Strip the HTML tags used for Telegram formatting (for --dry-run)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram messages max out at 4096 characters. Split on line boundaries
    # (never mid-tag) to be safe.
    for chunk in _chunks(text):
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",  # lets us use <b>bold</b>
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(url, data=data)
        # urlopen raises on non-2xx responses. The error text does not
        # include the URL, so the bot token never ends up in the logs.
        with urllib.request.urlopen(req, timeout=30):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram alerts when RSI hits 30/70.")
    parser.add_argument("--watchlist", default=str(BASE_DIR / "watchlist.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print alerts instead of sending, and don't save state")
    parser.add_argument("--report", action="store_true",
                        help="manual mode: send the current RSI of every ticker, "
                             "whether or not it is in a zone. Ignores and never "
                             "changes the saved alert state")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        tickers = load_watchlist(Path(args.watchlist))
    except (OSError, json.JSONDecodeError) as e:
        log.error("Could not read watchlist: %s", e)
        return 1
    if not tickers:
        log.error("Watchlist is empty")
        return 1

    state = load_state()
    new_state = dict(state)
    alerts = []  # (key, ticker, label, rsi, price, zone)
    readings = []  # (ticker, label, rsi, price, zone), used by --report

    for ticker in tickers:
        for label, (period, interval) in TIMEFRAMES.items():
            key = f"{ticker}|{label}"
            try:
                result = fetch_latest_rsi(ticker, period, interval)
            except Exception as e:  # network / yfinance hiccups
                log.warning("%s %s: fetch failed (%s)", ticker, label, e)
                continue
            if result is None:
                log.warning("%s %s: not enough data", ticker, label)
                continue

            rsi, price = result
            zone = zone_for(rsi)
            prev = state.get(key, "neutral")
            log.info("%s %s: RSI=%.1f (%s)", ticker, label, rsi, zone)
            readings.append((ticker, label, rsi, price, zone))

            if zone != prev:
                # status changed: became oversold / overbought / back to normal
                alerts.append((key, ticker, label, rsi, price, zone, prev))
            else:
                new_state[key] = zone  # unchanged, nothing to send

    if args.report:
        if not readings:
            log.error("No RSI data could be fetched")
            return 1
        icons = {"oversold": "📉 <b>oversold</b>",
                 "near_oversold": "🔻 <b>close to oversold</b>",
                 "overbought": "📈 <b>overbought</b>",
                 "near_overbought": "🔺 <b>close to overbought</b>",
                 "neutral": "<b>normal</b>"}
        rows = [f"<b>{html.escape(t)}</b> {lbl}: RSI {r:.1f} ({icons[z]}), price {p:,.2f}"
                for t, lbl, r, p, z in readings]
        message = "<b>RSI report (manual run)</b>\n\n" + "\n".join(rows)
        if args.dry_run:
            print(plain(message))
        else:
            try:
                send_telegram(message)
                log.info("Sent Telegram report with %d reading(s)", len(readings))
            except Exception as e:
                log.error("Telegram send failed: %s", e)
                return 1
        return 0  # report mode never touches the saved alert state

    if alerts:
        lines = []
        # "<" and ">" must be written as &lt; and &gt; in Telegram HTML
        status_text = {
            "oversold": "🟢 now <b>OVERSOLD</b> (RSI &lt;= 30)",
            "near_oversold": "🟢 now <b>CLOSE TO OVERSOLD</b> (RSI 30-35)",
            "overbought": "🔴 now <b>OVERBOUGHT</b> (RSI &gt;= 70)",
            "near_overbought": "🔴 now <b>CLOSE TO OVERBOUGHT</b> (RSI 65-70)",
            "neutral": "✅ back to <b>NORMAL</b> (RSI between 35 and 65)",
        }
        for _, ticker, label, rsi, price, zone, prev in alerts:
            tf = "daily" if label == "1Y" else "weekly"
            lines.append(f"<b>{html.escape(ticker)}</b> - {label} chart ({tf})\n"
                         f"{status_text[zone]}, was {ZONE_LABEL.get(prev, prev)}\n"
                         f"RSI {rsi:.1f}, last price {price:,.2f}")
        message = "<b>RSI status change</b>\n\n" + "\n\n".join(lines) + \
                  "\n\nAutomated signal, not financial advice."

        if args.dry_run:
            print(plain(message))
        else:
            try:
                send_telegram(message)
                log.info("Sent Telegram message with %d alert(s)", len(alerts))
                for key, _t, _l, _r, _p, zone, _prev in alerts:
                    new_state[key] = zone
            except Exception as e:
                # leave state unchanged so the alert is retried next run
                log.error("Telegram send failed: %s", e)

    if not args.dry_run:
        save_state(new_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())