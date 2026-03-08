#!/usr/bin/env python3
"""
v15 Live Dashboard — Real-time performance monitoring + controls
================================================================
A lightweight HTTP server that runs alongside the trader.
Open http://localhost:8080 in your browser.

Reads trades.json and state.json to display:
  - Portfolio summary (cumulative PnL, win rate, drawdown)
  - Equity curve chart
  - Current position details
  - Recent trade history
  - Long vs Short breakdown
  - Account & exposure panel
  - Engine control buttons (stop, pause, resume, flatten)

Also supports Backtest mode (reads backtest_results.json):
  - Regime periods table
  - Performance by regime cards
  - Regime-colored equity curve
  - Full trade history with Regime column

No external dependencies beyond Python stdlib.

Usage:
  python dashboard.py              # Start on port 8080
  python dashboard.py --port 9090  # Custom port
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add project to path — config.py lives inside btc_trader_v15/
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
_DASH_V15 = os.path.join(_DASH_DIR, "btc_trader_v15")
if _DASH_V15 not in sys.path:
    sys.path.insert(0, _DASH_V15)
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)
import config as cfg

DASHBOARD_PORT = 8080


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and JSON API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/state":
            self._serve_json(self._get_state())
        elif path == "/api/trades":
            self._serve_json(self._get_trades())
        elif path == "/api/metrics":
            self._serve_json(self._compute_metrics())
        elif path == "/api/all":
            # Single endpoint for everything (reduces polling requests)
            state = self._get_state()
            mode = state.get("mode", "live")

            base = {
                "state": state,
                "trades": self._get_trades(),
                "metrics": self._compute_metrics(),
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                # Account / exposure fields surfaced at top level for convenience
                "paper_balance": state.get("paper_balance", cfg.PAPER_BALANCE),
                "max_exposure": state.get("max_exposure", cfg.MAX_EXPOSURE_USD),
                "current_exposure": state.get("current_exposure", 0),
                "current_contracts": state.get("current_contracts", 0),
                "max_contracts": state.get("max_contracts", cfg.MAX_CONTRACTS),
                "contract_symbol": state.get("contract_symbol", ""),
                "days_to_expiry": state.get("days_to_expiry", None),
                "cooldown_hours": state.get("cooldown_hours", cfg.CHOPPY.get("cooldown_hours", 3)),
                "last_recal_time": state.get("last_recal_time"),
                "recalibrations": state.get("recalibrations", 0),
            }

            if mode == "backtest":
                base["backtest_results"] = self._get_backtest_results()

            self._serve_json(base)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/control":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                allowed = {"stop", "flatten_stop", "pause", "resume", "flatten"}
                cmd = data.get("command", "")
                if cmd not in allowed:
                    self._serve_json({"ok": False, "error": f"Unknown command: {cmd}"})
                    return
                control_path = Path(cfg.CONTROL_FILE)
                control_path.write_text(json.dumps(data))
                self._serve_json({"ok": True, "command": cmd})
            except Exception as e:
                self._serve_json({"ok": False, "error": str(e)})
        elif path == "/api/config":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                # Write config update file for engine to consume
                config_path = Path("config_update.json")
                config_path.write_text(json.dumps(data))
                self._serve_json({"ok": True, "updated": data})
            except Exception as e:
                self._serve_json({"ok": False, "error": str(e)})
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default access logging (too noisy)."""
        pass

    def _serve_json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(html)

    def _get_state(self):
        state_path = Path(cfg.STATE_FILE)
        if state_path.exists():
            try:
                return json.loads(state_path.read_text())
            except:
                pass
        return {}

    def _get_trades(self):
        trade_path = Path(cfg.TRADE_LOG)
        if trade_path.exists():
            try:
                return json.loads(trade_path.read_text())
            except:
                pass
        return []

    def _get_backtest_results(self):
        """Read backtest_results.json if it exists."""
        bt_path = Path("backtest_results.json")
        if bt_path.exists():
            try:
                return json.loads(bt_path.read_text())
            except:
                pass
        return {}

    def _compute_metrics(self):
        trades = self._get_trades()
        state = self._get_state()

        # Filter to closed trades (SELL or COVER)
        closed = [t for t in trades if t.get("action") in ("SELL", "COVER")]
        entries = [t for t in trades if t.get("action") in ("BUY", "SHORT")]

        if not closed:
            return {
                "total_trades": 0,
                "cumulative_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "max_drawdown": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "long_trades": 0,
                "long_pnl": 0,
                "long_wins": 0,
                "short_trades": 0,
                "short_pnl": 0,
                "short_wins": 0,
                "equity_curve": [],
                "profit_factor": 0,
                "avg_hold_hours": 0,
            }

        # Build equity curve
        equity = []
        running_pnl = 0
        peak = 0
        max_dd = 0
        wins = 0
        losses = 0
        gross_profit = 0
        gross_loss = 0

        long_trades = 0
        long_pnl = 0
        long_wins = 0
        short_trades = 0
        short_pnl = 0
        short_wins = 0

        for t in closed:
            pnl = t.get("net_pnl", t.get("pnl", 0))
            running_pnl += pnl
            # Derive side from action if not stored
            side = t.get("side", "")
            if not side:
                action = t.get("action", "")
                side = "long" if action == "SELL" else "short" if action == "COVER" else "unknown"

            if pnl >= 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

            if side == "long":
                long_trades += 1
                long_pnl += pnl
                if pnl >= 0:
                    long_wins += 1
            elif side == "short":
                short_trades += 1
                short_pnl += pnl
                if pnl >= 0:
                    short_wins += 1

            peak = max(peak, running_pnl)
            dd = peak - running_pnl
            max_dd = max(max_dd, dd)

            equity.append({
                "time": t.get("time", ""),
                "pnl": round(running_pnl, 2),
                "trade_pnl": round(pnl, 2),
                "side": side,
                "action": t.get("action", ""),
            })

        pnl_list = [t.get("net_pnl", t.get("pnl", 0)) for t in closed]
        total = len(closed)

        return {
            "total_trades": total,
            "cumulative_pnl": round(running_pnl, 2),
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "avg_pnl": round(running_pnl / total, 2) if total > 0 else 0,
            "max_drawdown": round(max_dd, 2),
            "best_trade": round(max(pnl_list), 2) if pnl_list else 0,
            "worst_trade": round(min(pnl_list), 2) if pnl_list else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "long_trades": long_trades,
            "long_pnl": round(long_pnl, 2),
            "long_wins": long_wins,
            "short_trades": short_trades,
            "short_pnl": round(short_pnl, 2),
            "short_wins": short_wins,
            "equity_curve": equity,
            "avg_hold_hours": 0,  # computed below
        }


# ══════════════════════════════════════════════════════
# DASHBOARD HTML (single-page, self-contained)
# ══════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Trader v15 — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232733;
    --border: #2d3141;
    --text: #e4e6ed;
    --text-dim: #8b8fa3;
    --accent: #3b82f6;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --bull: #22c55e;
    --bear: #ef4444;
    --choppy: #3b82f6;
    --neg-momentum: #f97316;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ─────────────────────────────────────── */
  .header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }

  .header-top {
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .header h1 {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.02em;
    white-space: nowrap;
  }

  .header h1 span { color: var(--accent); }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(34, 197, 94, 0.12);
    color: var(--green);
    white-space: nowrap;
  }

  .status-badge.offline {
    background: rgba(239, 68, 68, 0.12);
    color: var(--red);
  }

  .status-badge.paused {
    background: rgba(245, 158, 11, 0.12);
    color: var(--amber);
  }

  .status-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Mode Badge ──────────────────────────────────── */
  .mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .mode-badge.live {
    background: rgba(34, 197, 94, 0.1);
    color: var(--green);
    border: 1px solid rgba(34, 197, 94, 0.25);
  }

  .mode-badge.backtest {
    background: rgba(245, 158, 11, 0.1);
    color: var(--amber);
    border: 1px solid rgba(245, 158, 11, 0.25);
  }

  /* ── Regime Indicator (live mode) ───────────────── */
  .regime-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    background: var(--surface2);
    border: 1px solid var(--border);
    white-space: nowrap;
  }

  .regime-indicator.bull  { color: var(--bull);   border-color: rgba(34, 197, 94, 0.3); }
  .regime-indicator.bear  { color: var(--bear);   border-color: rgba(239, 68, 68, 0.3); }
  .regime-indicator.choppy { color: var(--choppy); border-color: rgba(59, 130, 246, 0.3); }
  .regime-indicator.neg_momentum { color: var(--neg-momentum); border-color: rgba(249, 115, 22, 0.3); }

  .regime-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
  }

  /* ── Control Bar ─────────────────────────────────── */
  .control-bar {
    padding: 10px 24px 12px;
    border-top: 1px solid rgba(45, 49, 65, 0.6);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .ctrl-label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-right: 4px;
    white-space: nowrap;
  }

  .ctrl-divider {
    width: 1px;
    height: 20px;
    background: var(--border);
    margin: 0 4px;
  }

  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: pointer;
    border: none;
    transition: opacity 0.15s, transform 0.1s, filter 0.15s;
    white-space: nowrap;
    font-family: var(--font);
  }

  .btn:hover { filter: brightness(1.15); }
  .btn:active { transform: scale(0.97); }
  .btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
    filter: none;
    transform: none;
  }

  .btn-start {
    background: rgba(34, 197, 94, 0.15);
    color: var(--green);
    border: 1px solid rgba(34, 197, 94, 0.3);
  }

  .btn-stop {
    background: #ef4444;
    color: #fff;
  }

  .btn-flatten-stop {
    background: #f59e0b;
    color: #0f1117;
  }

  .btn-pause {
    background: rgba(245, 158, 11, 0.15);
    color: var(--amber);
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .btn-resume {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent);
    border: 1px solid rgba(59, 130, 246, 0.3);
  }

  .btn-flatten {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
  }

  .btn-flatten:hover {
    color: var(--text);
    border-color: var(--text-dim);
    filter: none;
  }

  /* Pause/Resume toggle wrapper */
  .btn-toggle-wrap { display: contents; }

  /* ── Main Container ──────────────────────────────── */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 24px;
  }

  /* ── KPI Cards ───────────────────────────────────── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }

  .kpi-label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 4px;
  }

  .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }

  .kpi-sub {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 2px;
  }

  .pos { color: var(--green); }
  .neg { color: var(--red); }

  /* ── Account & Exposure Panel ────────────────────── */
  .account-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px 24px;
    align-items: start;
  }

  .ap-item {}

  .ap-label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 4px;
  }

  .ap-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  .ap-sub {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 1px;
  }

  /* Editable config fields */
  .ap-editable {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    position: relative;
  }
  .ap-editable:hover { opacity: 0.8; }
  .ap-editable .edit-icon {
    font-size: 10px;
    color: var(--text-dim);
    opacity: 0;
    transition: opacity 0.15s;
  }
  .ap-editable:hover .edit-icon { opacity: 1; }
  .config-input {
    background: var(--surface2);
    border: 1px solid var(--accent);
    border-radius: 4px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 2px 6px;
    width: 110px;
    outline: none;
  }
  .config-save {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 3px;
    font-size: 11px;
    padding: 2px 8px;
    cursor: pointer;
    font-weight: 600;
  }
  .config-save:hover { opacity: 0.85; }
  .config-cancel {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 11px;
    padding: 2px 6px;
    cursor: pointer;
  }

  /* Exposure gauge column — wider */
  .ap-gauge-col {
    grid-column: span 2;
  }

  @media (max-width: 640px) {
    .ap-gauge-col { grid-column: span 1; }
  }

  .gauge-wrap {
    margin-top: 6px;
  }

  .gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 5px;
  }

  .gauge-track {
    height: 8px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
  }

  .gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease, background-color 0.4s ease;
    background: var(--green);
  }

  .gauge-fill.amber { background: var(--amber); }
  .gauge-fill.red   { background: var(--red); }

  .gauge-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    margin-top: 4px;
    color: var(--green);
    transition: color 0.4s ease;
  }

  .gauge-pct.amber { color: var(--amber); }
  .gauge-pct.red   { color: var(--red); }

  /* ── Entry Parameters Panel ────────────────────────────────────── */
  .params-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 768px) {
    .params-grid { grid-template-columns: 1fr; }
  }
  .params-col {
    padding: 8px;
  }
  .params-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
    color: var(--text-dim);
  }
  .long-params { background: rgba(34,197,94,0.05); border-radius: 6px; }
  .short-params { background: rgba(239,68,68,0.05); border-radius: 6px; }
  .param-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .param-row:last-child { border-bottom: none; }
  .param-label { color: var(--text-dim); }
  .param-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }
  .param-check {
    width: 20px;
    text-align: center;
    font-size: 13px;
  }
  .param-check.pass { color: var(--green); }
  .param-check.fail { color: var(--red); }

  /* Expiry coloring */
  .expiry-green { color: var(--green); }
  .expiry-amber { color: var(--amber); }
  .expiry-red   { color: var(--red); }

  /* ── Two-column layout ───────────────────────────── */
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 768px) {
    .grid-2 { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .control-bar { gap: 6px; }
    .ctrl-divider { display: none; }
  }

  /* ── Card ────────────────────────────────────────── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .card-body { padding: 16px; }

  /* ── Position card ───────────────────────────────── */
  .position-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(45, 49, 65, 0.5);
    font-size: 13px;
  }

  .position-row:last-child { border-bottom: none; }
  .position-row .label { color: var(--text-dim); }
  .position-row .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }

  .side-long {
    background: rgba(34, 197, 94, 0.12);
    color: var(--green);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .side-short {
    background: rgba(239, 68, 68, 0.12);
    color: var(--red);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .side-flat {
    background: rgba(139, 143, 163, 0.12);
    color: var(--text-dim);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  /* ── Chart ───────────────────────────────────────── */
  .chart-container {
    position: relative;
    height: 220px;
    padding: 12px 16px 16px;
  }

  /* ── Trade table ─────────────────────────────────── */
  .trade-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  .trade-table th {
    text-align: left;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
  }

  .trade-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(45, 49, 65, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .trade-table tr:last-child td { border-bottom: none; }

  .trade-table .action-buy   { color: var(--green); font-weight: 600; }
  .trade-table .action-sell  { color: var(--amber); font-weight: 600; }
  .trade-table .action-short { color: var(--red); font-weight: 600; }
  .trade-table .action-cover { color: var(--accent); font-weight: 600; }

  /* ── Long/Short breakdown ────────────────────────── */
  .breakdown {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .breakdown-box {
    padding: 12px;
    border-radius: 6px;
    text-align: center;
  }

  .breakdown-box.long-box {
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.15);
  }

  .breakdown-box.short-box {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.15);
  }

  .breakdown-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }

  .breakdown-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
  }

  .breakdown-detail {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 2px;
  }

  /* ── Range bar ───────────────────────────────────── */
  .range-bar {
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    margin-top: 8px;
    position: relative;
    overflow: visible;
  }

  .range-bar .fill {
    position: absolute;
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  .range-bar .marker {
    position: absolute;
    top: -4px;
    width: 3px;
    height: 14px;
    background: var(--text);
    border-radius: 2px;
    transform: translateX(-50%);
    transition: left 0.5s ease;
  }

  /* ── Footer ──────────────────────────────────────── */
  .footer {
    text-align: center;
    padding: 16px;
    font-size: 11px;
    color: var(--text-dim);
  }

  .empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-dim);
    font-size: 13px;
  }

  /* ── Toast notification ──────────────────────────── */
  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #fff;
    background: #1a1d27;
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    z-index: 9999;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.2s, transform 0.2s;
    pointer-events: none;
  }

  #toast.show {
    opacity: 1;
    transform: translateY(0);
  }

  #toast.ok   { border-color: var(--green); color: var(--green); }
  #toast.err  { border-color: var(--red);   color: var(--red); }

  /* ── Regime tag chip ─────────────────────────────── */
  .regime-chip {
    display: inline-flex;
    align-items: center;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: var(--font);
  }
  .regime-chip.bull   { background: rgba(34, 197, 94, 0.12); color: var(--bull); }
  .regime-chip.bear   { background: rgba(239, 68, 68, 0.12); color: var(--bear); }
  .regime-chip.choppy { background: rgba(59, 130, 246, 0.12); color: var(--choppy); }
  .regime-chip.neg_momentum { background: rgba(249, 115, 22, 0.12); color: var(--neg-momentum); }

  /* ── Backtest view ───────────────────────────────── */
  #bt-view { display: none; }
  #live-view { display: block; }

  .bt-section-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
    margin-top: 4px;
  }

  /* Regime periods table row coloring */
  .trade-table tr.regime-bull td   { background: rgba(34, 197, 94, 0.04); }
  .trade-table tr.regime-bear td   { background: rgba(239, 68, 68, 0.04); }
  .trade-table tr.regime-choppy td { background: rgba(59, 130, 246, 0.04); }

  /* Per-regime performance cards */
  .regime-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .regime-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }

  .regime-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .regime-card-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .regime-card-title.bull   { color: var(--bull); }
  .regime-card-title.bear   { color: var(--bear); }
  .regime-card-title.choppy { color: var(--choppy); }
  .regime-card-title.neg_momentum { color: var(--neg-momentum); }

  .regime-card-stat {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(45, 49, 65, 0.4);
  }
  .regime-card-stat:last-child { border-bottom: none; }
  .regime-card-stat .rcs-label { color: var(--text-dim); }
  .regime-card-stat .rcs-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }

  /* ── Exposure & ROI Panel (backtest) ────────────── */
  .exposure-roi-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .exp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }

  .exp-card-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 10px;
  }

  .exp-stat {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(45, 49, 65, 0.4);
  }

  .exp-stat:last-child { border-bottom: none; }
  .exp-stat .es-label { color: var(--text-dim); }
  .exp-stat .es-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }

  .roi-highlight {
    font-size: 18px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
  }

  .roi-ann {
    font-size: 11px;
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
  }

  /* Backtest banner */
  .bt-banner {
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: var(--amber);
  }

  .bt-banner-icon { font-size: 16px; }
  .bt-banner-text { flex: 1; }
  .bt-banner-text strong { font-weight: 600; }
  .bt-banner-text small { font-size: 11px; color: var(--text-dim); display: block; margin-top: 1px; }

  /* Controls disabled overlay message */
  .controls-disabled-note {
    font-size: 11px;
    color: var(--text-dim);
    font-style: italic;
    padding: 10px 24px 12px;
    border-top: 1px solid rgba(45, 49, 65, 0.6);
  }
</style>
</head>
<body>

<!-- ── Header + Control Bar ───────────────────────── -->
<div class="header">
  <div class="header-top">
    <h1>BTC Trader <span>v15</span> — Config I</h1>
    <div class="header-right">
      <!-- Mode badge -->
      <span id="mode-badge" class="mode-badge live">LIVE PAPER TRADING</span>
      <!-- Live regime indicator (shown only in live mode) -->
      <span id="regime-indicator" class="regime-indicator" style="display:none;"></span>
      <!-- Connection status -->
      <div id="status-badge" class="status-badge offline">
        <div class="dot"></div>
        <span id="status-text">Connecting...</span>
      </div>
    </div>
  </div>

  <!-- Control buttons (hidden/disabled in backtest mode) -->
  <div class="control-bar" id="control-bar">
    <span class="ctrl-label">Controls</span>

    <!-- START (visual only — engine starts from CLI) -->
    <button class="btn btn-start" id="btn-start" disabled title="Start the engine from the command line">
      ▶ START
    </button>

    <div class="ctrl-divider"></div>

    <!-- STOP -->
    <button class="btn btn-stop" id="btn-stop" onclick="sendCommand('stop', true)">
      ■ STOP
    </button>

    <!-- FLATTEN & STOP -->
    <button class="btn btn-flatten-stop" id="btn-flatten-stop" onclick="sendCommand('flatten_stop', true)">
      ⬛ FLATTEN &amp; STOP
    </button>

    <div class="ctrl-divider"></div>

    <!-- PAUSE / RESUME toggle -->
    <button class="btn btn-pause" id="btn-pause" onclick="sendCommand('pause', false)">
      ⏸ PAUSE
    </button>
    <button class="btn btn-resume" id="btn-resume" onclick="sendCommand('resume', false)" style="display:none;">
      ▶ RESUME
    </button>

    <div class="ctrl-divider"></div>

    <!-- FLATTEN (keep running) -->
    <button class="btn btn-flatten" id="btn-flatten" onclick="sendCommand('flatten', false)">
      ↩ FLATTEN
    </button>
  </div>

  <!-- Shown in backtest mode instead of controls -->
  <div class="controls-disabled-note" id="bt-controls-note" style="display:none;">
    Controls are disabled in backtest mode — this is a completed simulation.
  </div>
</div>

<!-- ── Main Content ────────────────────────────────── -->
<div class="container">

  <!-- ════════════════════════════════════════════════
       LIVE VIEW (shown when mode = "live" or missing)
       ════════════════════════════════════════════════ -->
  <div id="live-view">

    <!-- KPI Row -->
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Cumulative PnL</div>
        <div class="kpi-value" id="kpi-pnl">$0.00</div>
        <div class="kpi-sub" id="kpi-pnl-sub">0 trades</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value" id="kpi-winrate">0%</div>
        <div class="kpi-sub" id="kpi-winrate-sub">0W / 0L</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value neg" id="kpi-dd">$0.00</div>
        <div class="kpi-sub">Peak-to-trough</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Avg Trade</div>
        <div class="kpi-value" id="kpi-avg">$0.00</div>
        <div class="kpi-sub" id="kpi-pf">PF: —</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Best Trade</div>
        <div class="kpi-value pos" id="kpi-best">$0.00</div>
        <div class="kpi-sub">Single trade</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Worst Trade</div>
        <div class="kpi-value neg" id="kpi-worst">$0.00</div>
        <div class="kpi-sub">Single trade</div>
      </div>
    </div>

    <!-- Account & Exposure Panel -->
    <div class="account-panel" id="account-panel">

      <div class="ap-item">
        <div class="ap-label">Paper Balance</div>
        <div class="ap-value" id="ap-balance-wrap">
          <span class="ap-editable" id="ap-balance" onclick="editConfig('paper_balance', this)" title="Click to edit">—</span>
        </div>
      </div>

      <div class="ap-item">
        <div class="ap-label">Max Exposure</div>
        <div class="ap-value" id="ap-max-exp-wrap">
          <span class="ap-editable" id="ap-max-exp" onclick="editConfig('max_exposure', this)" title="Click to edit">—</span>
        </div>
      </div>

      <div class="ap-item">
        <div class="ap-label">Cooldown</div>
        <div class="ap-value" id="ap-cooldown-wrap">
          <span class="ap-editable" id="ap-cooldown" onclick="editConfig('cooldown_hours', this)" title="Click to edit">—</span>
        </div>
      </div>

      <div class="ap-item">
        <div class="ap-label">Contracts</div>
        <div class="ap-value"><span id="ap-cur-ct">0</span> / <span id="ap-max-ct-display">3</span></div>
        <div class="ap-sub"><span id="ap-symbol">—</span> · <span class="ap-editable" id="ap-max-ct" onclick="editConfig('max_contracts', this)" title="Click to edit max" style="font-size:11px;">max: 3 ✎</span></div>
      </div>

      <div class="ap-item">
        <div class="ap-label">Expiry</div>
        <div class="ap-value" id="ap-expiry">—</div>
        <div class="ap-sub" id="ap-expiry-sub"></div>
      </div>

      <div class="ap-item">
        <div class="ap-label">Recalibration</div>
        <div class="ap-value" id="ap-recal-count">—</div>
        <div class="ap-sub" id="ap-recal-time">—</div>
      </div>

      <!-- Exposure gauge spans 2 columns -->
      <div class="ap-item ap-gauge-col">
        <div class="ap-label">Current Exposure</div>
        <div class="gauge-wrap">
          <div class="gauge-labels">
            <span id="ap-cur-exp">$0</span>
            <span id="ap-exp-pct-label">0% used</span>
          </div>
          <div class="gauge-track">
            <div class="gauge-fill" id="gauge-fill"></div>
          </div>
          <div class="gauge-pct" id="gauge-pct-text">0%</div>
        </div>
      </div>

    </div>

    <!-- Entry Parameters Panel -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Entry Parameters</span>
        <span id="signal-status" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="card-body">
        <div class="params-grid">
          <!-- Current Values -->
          <div class="params-col">
            <div class="params-title">Current Values</div>
            <div class="param-row">
              <span class="param-label">Price</span>
              <span class="param-val" id="param-price">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">Range Position</span>
              <span class="param-val" id="param-rng-pos">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">RSI (14)</span>
              <span class="param-val" id="param-rsi">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">ADX (14)</span>
              <span class="param-val" id="param-adx">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">Last Signal</span>
              <span class="param-val" id="param-signal" style="font-size:11px;">—</span>
            </div>
          </div>
          <!-- Long Entry Thresholds -->
          <div class="params-col long-params">
            <div class="params-title" style="color:var(--green);">Long Entry Requires</div>
            <div class="param-row">
              <span class="param-label">Range Position</span>
              <span class="param-val">&le; 30%</span>
              <span class="param-check" id="chk-long-rng">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">RSI</span>
              <span class="param-val">&lt; 45</span>
              <span class="param-check" id="chk-long-rsi">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">No Cooldown</span>
              <span class="param-val">—</span>
              <span class="param-check" id="chk-long-cd">—</span>
            </div>
          </div>
          <!-- Short Entry Thresholds -->
          <div class="params-col short-params">
            <div class="params-title" style="color:var(--red);">Short Entry Requires</div>
            <div class="param-row">
              <span class="param-label">Range Position</span>
              <span class="param-val">&ge; 70%</span>
              <span class="param-check" id="chk-short-rng">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">RSI</span>
              <span class="param-val">&gt; 55</span>
              <span class="param-check" id="chk-short-rsi">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">ADX</span>
              <span class="param-val">&lt; 25</span>
              <span class="param-check" id="chk-short-adx">—</span>
            </div>
            <div class="param-row">
              <span class="param-label">No Cooldown</span>
              <span class="param-val">—</span>
              <span class="param-check" id="chk-short-cd">—</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Equity Curve -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Equity Curve</span>
        <span id="chart-label" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="chart-container">
        <canvas id="equityChart"></canvas>
      </div>
    </div>

    <!-- Two-column: Position + Breakdown -->
    <div class="grid-2">

      <!-- Current Position -->
      <div class="card">
        <div class="card-header">
          <span>Current Position</span>
          <span id="pos-side" class="side-flat">FLAT</span>
        </div>
        <div class="card-body" id="position-body">
          <div class="empty-state">No active position</div>
        </div>
      </div>

      <!-- Long vs Short -->
      <div class="card">
        <div class="card-header">Long vs Short Breakdown</div>
        <div class="card-body">
          <div class="breakdown">
            <div class="breakdown-box long-box">
              <div class="breakdown-title" style="color:var(--green);">Longs</div>
              <div class="breakdown-pnl pos" id="long-pnl">$0</div>
              <div class="breakdown-detail" id="long-detail">0 trades, 0 wins</div>
            </div>
            <div class="breakdown-box short-box">
              <div class="breakdown-title" style="color:var(--red);">Shorts</div>
              <div class="breakdown-pnl neg" id="short-pnl">$0</div>
              <div class="breakdown-detail" id="short-detail">0 trades, 0 wins</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Trades (Live) -->
    <div class="card">
      <div class="card-header">
        <span>Trade History</span>
        <span id="trade-count" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="card-body" style="padding:0; overflow-x:auto;">
        <table class="trade-table" id="trade-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Action</th>
              <th>Side</th>
              <th>Price</th>
              <th>Entry</th>
              <th>PnL</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody id="trade-tbody">
            <tr><td colspan="7" class="empty-state">No trades yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div><!-- /#live-view -->


  <!-- ════════════════════════════════════════════════
       BACKTEST VIEW (shown when mode = "backtest")
       ════════════════════════════════════════════════ -->
  <div id="bt-view">

    <!-- Backtest info banner -->
    <div class="bt-banner" id="bt-banner">
      <span class="bt-banner-icon">📊</span>
      <div class="bt-banner-text">
        <strong>Backtest Results</strong> <span style="font-size:10px;color:#888;">(v3.1)</span>
        <small id="bt-banner-sub">Loading backtest data...</small>
      </div>
    </div>

    <!-- KPI Row (reuse same IDs — populated by backtest metrics) -->
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Total PnL</div>
        <div class="kpi-value" id="bt-kpi-pnl">$0.00</div>
        <div class="kpi-sub" id="bt-kpi-pnl-sub">0 trades</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value" id="bt-kpi-winrate">0%</div>
        <div class="kpi-sub" id="bt-kpi-winrate-sub">0W / 0L</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value neg" id="bt-kpi-dd">$0.00</div>
        <div class="kpi-sub">Peak-to-trough</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Avg Trade</div>
        <div class="kpi-value" id="bt-kpi-avg">$0.00</div>
        <div class="kpi-sub" id="bt-kpi-pf">PF: —</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Best Trade</div>
        <div class="kpi-value pos" id="bt-kpi-best">$0.00</div>
        <div class="kpi-sub">Single trade</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Worst Trade</div>
        <div class="kpi-value neg" id="bt-kpi-worst">$0.00</div>
        <div class="kpi-sub">Single trade</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Trade Duration</div>
        <div class="kpi-value" id="bt-kpi-duration" style="font-size:20px;">—</div>
        <div class="kpi-sub" id="bt-kpi-duration-sub">avg · min · max</div>
      </div>
    </div>

    <!-- Equity Curve (Backtest, regime-colored) -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Equity Curve</span>
        <span id="bt-chart-label" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="chart-container">
        <canvas id="btEquityChart"></canvas>
      </div>
    </div>

    <!-- Notional Exposure Chart (Backtest) -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Investment Amount Over Time</span>
        <span id="bt-notional-label" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="chart-container">
        <canvas id="btNotionalChart"></canvas>
      </div>
    </div>

    <!-- Margin Over Time Chart (Backtest) -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Broker Margin Over Time</span>
        <span id="bt-blocked-label" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="chart-container">
        <canvas id="btBlockedChart"></canvas>
      </div>
    </div>

    <!-- Exposure & ROI Summary -->
    <div style="margin-bottom: 16px;">
      <div class="bt-section-title">Exposure, Capital Utilization & ROI</div>
      <div class="exposure-roi-panel" id="bt-exposure-roi">
        <!-- Populated by JS -->
        <div class="empty-state" style="padding:20px;">No exposure data available</div>
      </div>
    </div>

    <!-- Performance by Regime -->
    <div style="margin-bottom: 16px;">
      <div class="bt-section-title">Performance by Cluster</div>
      <div class="regime-cards" id="bt-regime-cards">
        <!-- Populated by JS -->
        <div class="empty-state" style="padding:20px;">No regime data available</div>
      </div>
    </div>

    <!-- Regime Periods Table -->
    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <span>Regime Periods</span>
        <span id="bt-regime-count" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="card-body" style="padding:0; overflow-x:auto;">
        <table class="trade-table" id="bt-regime-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Regime</th>
              <th>Start</th>
              <th>End</th>
              <th>Duration</th>
              <th>Start Price</th>
              <th>End Price</th>
              <th>Return</th>
            </tr>
          </thead>
          <tbody id="bt-regime-tbody">
            <tr><td colspan="8" class="empty-state">No regime periods</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Trade History (Backtest) -->
    <div class="card">
      <div class="card-header">
        <span>Trade History</span>
        <span id="bt-trade-count" style="font-size:11px; color:var(--text-dim);"></span>
      </div>
      <div class="card-body" style="padding:0; overflow-x:auto;">
        <table class="trade-table" id="bt-trade-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Action</th>
              <th>Side</th>
              <th>Price</th>
              <th>Entry</th>
              <th>Investment</th>
              <th>PnL</th>
              <th>Regime</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody id="bt-trade-tbody">
            <tr><td colspan="9" class="empty-state">No trades yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div><!-- /#bt-view -->

</div>

<div class="footer" id="footer">Last updated: —</div>

<!-- Toast -->
<div id="toast"></div>

<script>
// ── Chart setup (Live) ───────────────────────────────
const ctx = document.getElementById('equityChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Cumulative PnL',
      data: [],
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 4,
      pointBackgroundColor: [],
      pointBorderColor: [],
      pointBorderWidth: 2,
      borderWidth: 2,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1d27',
        borderColor: '#2d3141',
        borderWidth: 1,
        titleColor: '#e4e6ed',
        bodyColor: '#e4e6ed',
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        padding: 10,
        callbacks: {
          label: function(ctx) {
            return 'PnL: $' + ctx.parsed.y.toFixed(2);
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8b8fa3', font: { size: 10 } },
        grid: { color: 'rgba(45,49,65,0.4)' }
      },
      y: {
        ticks: {
          color: '#8b8fa3',
          font: { family: 'JetBrains Mono', size: 11 },
          callback: v => '$' + v.toFixed(0)
        },
        grid: { color: 'rgba(45,49,65,0.4)' }
      }
    }
  }
});

// ── Chart setup (Backtest) ───────────────────────────
const btCtx = document.getElementById('btEquityChart').getContext('2d');
const btChart = new Chart(btCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Backtest PnL',
      data: [],
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 4,
      pointBackgroundColor: [],
      pointBorderColor: [],
      pointBorderWidth: 2,
      borderWidth: 2,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1d27',
        borderColor: '#2d3141',
        borderWidth: 1,
        titleColor: '#e4e6ed',
        bodyColor: '#e4e6ed',
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        padding: 10,
        callbacks: {
          label: function(c) {
            const extra = c.dataset.regimes ? (' [' + (c.dataset.regimes[c.dataIndex] || '') + ']') : '';
            return 'PnL: $' + c.parsed.y.toFixed(2) + extra;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8b8fa3', font: { size: 10 } },
        grid: { color: 'rgba(45,49,65,0.4)' }
      },
      y: {
        ticks: {
          color: '#8b8fa3',
          font: { family: 'JetBrains Mono', size: 11 },
          callback: v => '$' + v.toFixed(0)
        },
        grid: { color: 'rgba(45,49,65,0.4)' }
      }
    }
  }
});

// ── Chart setup (Backtest Notional Exposure) ───────────
const ntCtx = document.getElementById('btNotionalChart').getContext('2d');
const ntChart = new Chart(ntCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Investment Amount ($)',
      data: [],
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.10)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 1.5,
      stepped: 'before',
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1d27',
        borderColor: '#2d3141',
        borderWidth: 1,
        titleColor: '#e4e6ed',
        bodyColor: '#e4e6ed',
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        padding: 10,
        callbacks: {
          label: function(c) {
            return 'Investment: $' + c.parsed.y.toLocaleString('en-US', {maximumFractionDigits: 0});
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8b8fa3', font: { size: 10 } },
        grid: { color: 'rgba(45,49,65,0.4)' }
      },
      y: {
        ticks: {
          color: '#8b8fa3',
          font: { family: 'JetBrains Mono', size: 11 },
          callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v.toFixed(0))
        },
        grid: { color: 'rgba(45,49,65,0.4)' }
      }
    }
  }
});

// ── Chart setup (Backtest Broker Margin) ───────────
const blkCtx = document.getElementById('btBlockedChart').getContext('2d');
const blkChart = new Chart(blkCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Margin ($)',
        data: [],
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.08)',
        fill: true,
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 1.5,
        stepped: 'before',
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: true, labels: { color: '#8b8fa3', font: { size: 10 } } },
      tooltip: {
        backgroundColor: '#1a1d27',
        borderColor: '#2d3141',
        borderWidth: 1,
        titleColor: '#e4e6ed',
        bodyColor: '#e4e6ed',
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        padding: 10,
        callbacks: {
          label: function(c) {
            return c.dataset.label + ': $' + c.parsed.y.toLocaleString('en-US', {maximumFractionDigits: 0});
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8b8fa3', font: { size: 10 } },
        grid: { color: 'rgba(45,49,65,0.4)' }
      },
      y: {
        ticks: {
          color: '#8b8fa3',
          font: { family: 'JetBrains Mono', size: 11 },
          callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v.toFixed(0))
        },
        grid: { color: 'rgba(45,49,65,0.4)' }
      }
    }
  }
});

// ── Formatters ───────────────────────────────────────
function fmt(n) {
  if (n === undefined || n === null) return '—';
  const s = Math.abs(n).toFixed(2);
  return (n < 0 ? '-$' : '$') + s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function fmtK(n) {
  if (n === undefined || n === null) return '—';
  if (Math.abs(n) >= 1000000) return '$' + (n/1000000).toFixed(2) + 'M';
  if (Math.abs(n) >= 1000) return '$' + (n/1000).toFixed(0) + 'K';
  return '$' + n.toFixed(0);
}

function fmtPrice(n) {
  if (!n) return '—';
  return '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
}

function fmtTime(t) {
  if (!t) return '—';
  const d = new Date(t);
  return d.toLocaleDateString('en-US', {month:'short', day:'numeric'}) + ' '
       + d.toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit', hour12:false});
}

function fmtShort(t) {
  if (!t) return '';
  const d = new Date(t);
  return (d.getMonth()+1) + '/' + d.getDate() + ' ' + d.getHours() + ':' + String(d.getMinutes()).padStart(2,'0');
}

function colorClass(n) {
  if (n > 0) return 'pos';
  if (n < 0) return 'neg';
  return '';
}

function actionClass(a) {
  return 'action-' + (a || '').toLowerCase();
}

// ── Regime helpers ────────────────────────────────────
const CLUSTER_DISPLAY = {
  bull: 'Positive Momentum',
  choppy: 'Range',
  bear: 'Volatile',
  neg_momentum_skip: 'Negative Momentum',
};

function regimeDisplayName(r) {
  return (r && CLUSTER_DISPLAY[r]) ? CLUSTER_DISPLAY[r] : (r || 'Unknown');
}

function regimeClass(r) {
  if (!r) return '';
  const l = r.toLowerCase();
  if (l.includes('bull') || l === 'positive momentum') return 'bull';
  if (l.includes('neg_momentum') || l === 'negative momentum') return 'neg_momentum';
  if (l.includes('bear') || l === 'volatile') return 'bear';
  return 'choppy';
}

function regimeColor(r) {
  const c = regimeClass(r);
  if (c === 'bull')           return '#22c55e';
  if (c === 'bear')           return '#ef4444';
  if (c === 'neg_momentum')   return '#f97316';
  return '#3b82f6';
}

function regimeChip(r) {
  if (!r) return '—';
  const c = regimeClass(r);
  const name = regimeDisplayName(r);
  return `<span class="regime-chip ${c}">${name}</span>`;
}

// ── Toast ─────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + (type || '');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, 3000);
}

// ── Config editing ───────────────────────────────────
let currentConfigValues = {
  paper_balance: 1000000,
  max_exposure: 500000,
  cooldown_hours: 3,
  max_contracts: 3
};

function editConfig(field, el) {
  if (el.querySelector('input')) return;

  const raw = currentConfigValues[field];
  const parentEl = el.parentElement;

  let unit = '';
  let step = '1';
  let displayVal = raw;
  if (field === 'paper_balance' || field === 'max_exposure') {
    displayVal = raw;
    unit = ' USD';
  } else if (field === 'cooldown_hours') {
    displayVal = raw;
    unit = ' hrs';
    step = '0.25';
  } else if (field === 'max_contracts') {
    displayVal = raw;
    unit = ' ct';
    step = '1';
  }

  parentEl.innerHTML = `
    <input type="number" class="config-input" id="cfg-input-${field}" 
           value="${displayVal}" step="${step}" min="0"
           onkeydown="if(event.key==='Enter')saveConfig('${field}');if(event.key==='Escape')cancelEdit('${field}')">
    <button class="config-save" onclick="saveConfig('${field}')">&#x2713;</button>
    <button class="config-cancel" onclick="cancelEdit('${field}')">&#x2715;</button>
  `;
  const inp = document.getElementById('cfg-input-' + field);
  inp.focus();
  inp.select();
}

async function saveConfig(field) {
  const inp = document.getElementById('cfg-input-' + field);
  if (!inp) return;
  
  let val = parseFloat(inp.value);
  if (isNaN(val) || val < 0) {
    showToast('Invalid value', 'err');
    return;
  }

  if (field === 'paper_balance' || field === 'max_exposure') {
    val = Math.round(val);
  } else if (field === 'cooldown_hours') {
    val = Math.round(val * 4) / 4;
  } else if (field === 'max_contracts') {
    val = Math.max(1, Math.round(val));
  }

  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [field]: val }),
    });
    const data = await res.json();
    if (data.ok) {
      currentConfigValues[field] = val;
      showToast('Updated ' + field.replace(/_/g, ' ') + ' \u2192 ' + val, 'ok');
      cancelEdit(field);
      poll();
    } else {
      showToast('Error: ' + (data.error || 'unknown'), 'err');
    }
  } catch (e) {
    showToast('Request failed: ' + e.message, 'err');
  }
}

function cancelEdit(field) {
  const val = currentConfigValues[field];
  let displayText = '';
  if (field === 'paper_balance') {
    displayText = fmtK(val);
    const wrap = document.getElementById('ap-balance-wrap');
    wrap.innerHTML = `<span class="ap-editable" id="ap-balance" onclick="editConfig('paper_balance', this)" title="Click to edit">${displayText} <span class="edit-icon">&#x270E;</span></span>`;
  } else if (field === 'max_exposure') {
    displayText = fmtK(val);
    const wrap = document.getElementById('ap-max-exp-wrap');
    wrap.innerHTML = `<span class="ap-editable" id="ap-max-exp" onclick="editConfig('max_exposure', this)" title="Click to edit">${displayText} <span class="edit-icon">&#x270E;</span></span>`;
  } else if (field === 'cooldown_hours') {
    displayText = val + 'h';
    const wrap = document.getElementById('ap-cooldown-wrap');
    wrap.innerHTML = `<span class="ap-editable" id="ap-cooldown" onclick="editConfig('cooldown_hours', this)" title="Click to edit">${displayText} <span class="edit-icon">&#x270E;</span></span>`;
  } else if (field === 'max_contracts') {
    const mctEl = document.getElementById('ap-max-ct');
    if (mctEl) mctEl.innerHTML = 'max: ' + val + ' <span class="edit-icon">&#x270E;</span>';
    document.getElementById('ap-max-ct-display').textContent = val;
    return;
  }
}

// ── Control commands ──────────────────────────────────
async function sendCommand(command, requireConfirm) {
  if (requireConfirm) {
    if (!confirm('Are you sure you want to send: ' + command.toUpperCase().replace('_', ' & ') + '?')) return;
  }
  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
    });
    const data = await res.json();
    if (data.ok) {
      showToast('Command sent: ' + command, 'ok');
    } else {
      showToast('Error: ' + (data.error || 'unknown'), 'err');
    }
  } catch (e) {
    showToast('Request failed: ' + e.message, 'err');
  }
}

// ── Gauge helpers ─────────────────────────────────────
function updateGauge(currentExposure, maxExposure) {
  const pct = maxExposure > 0 ? Math.min(100, (currentExposure / maxExposure) * 100) : 0;
  const fill = document.getElementById('gauge-fill');
  const pctText = document.getElementById('gauge-pct-text');
  const pctLabel = document.getElementById('ap-exp-pct-label');

  const cls = pct >= 80 ? 'red' : pct >= 50 ? 'amber' : '';
  fill.style.width = pct.toFixed(1) + '%';
  fill.className = 'gauge-fill' + (cls ? ' ' + cls : '');
  pctText.className = 'gauge-pct' + (cls ? ' ' + cls : '');
  pctText.textContent = pct.toFixed(1) + '%';
  pctLabel.textContent = pct.toFixed(1) + '% used';
}

// ── Backtest view updater ────────────────────────────
function updateBacktestView(data) {
  const bt = data.backtest_results || {};
  const metrics = bt.metrics || data.metrics || {};
  const trades = bt.trades || data.trades || [];
  const regimes = bt.regimes || [];
  const regimeSummary = bt.regime_summary || bt.metrics_by_regime || {};
  const equityCurve = bt.equity_curve || (data.metrics || {}).equity_curve || [];

  // ── Banner sub-text ──
  const bannerSub = document.getElementById('bt-banner-sub');
  if (metrics.total_trades !== undefined) {
    const start = regimes.length > 0 ? regimes[0].start || regimes[0].start_time || '' : '';
    const end   = regimes.length > 0 ? (regimes[regimes.length-1].end || regimes[regimes.length-1].end_time || '') : '';
    bannerSub.textContent = [
      metrics.total_trades + ' trades',
      regimes.length + ' regime periods',
      start ? fmtTime(start) + ' → ' + fmtTime(end) : ''
    ].filter(Boolean).join(' · ');
  }

  // ── KPIs ──
  const pnl = metrics.cumulative_pnl ?? metrics.total_pnl ?? 0;
  const pnlEl = document.getElementById('bt-kpi-pnl');
  pnlEl.textContent = fmt(pnl);
  pnlEl.className = 'kpi-value ' + colorClass(pnl);
  document.getElementById('bt-kpi-pnl-sub').textContent = (metrics.total_trades || 0) + ' trades';

  document.getElementById('bt-kpi-winrate').textContent = (metrics.win_rate || 0) + '%';
  const wins = Math.round((metrics.win_rate||0)/100 * (metrics.total_trades||0));
  const losses = (metrics.total_trades||0) - wins;
  document.getElementById('bt-kpi-winrate-sub').textContent = wins + 'W / ' + losses + 'L';

  document.getElementById('bt-kpi-dd').textContent = fmt(-(metrics.max_drawdown||0));

  const avgEl = document.getElementById('bt-kpi-avg');
  avgEl.textContent = fmt(metrics.avg_pnl);
  avgEl.className = 'kpi-value ' + colorClass(metrics.avg_pnl);

  const pf = metrics.profit_factor;
  document.getElementById('bt-kpi-pf').textContent = 'PF: ' + (pf === Infinity || pf === null || pf === undefined ? '—' : Number(pf).toFixed(2));
  document.getElementById('bt-kpi-best').textContent = fmt(metrics.best_trade);
  document.getElementById('bt-kpi-worst').textContent = fmt(metrics.worst_trade);

  // Trade duration
  const avgDur = metrics.avg_duration_days;
  const minDur = metrics.min_duration_days;
  const maxDur = metrics.max_duration_days;
  const medDur = metrics.median_duration_days;
  if (avgDur !== undefined && avgDur > 0) {
    document.getElementById('bt-kpi-duration').textContent = avgDur.toFixed(1) + 'd avg';
    document.getElementById('bt-kpi-duration-sub').textContent =
      minDur.toFixed(0) + 'd min · ' + medDur.toFixed(0) + 'd med · ' + maxDur.toFixed(0) + 'd max';
  }

  // ── Equity curve (regime-colored points) ──
  if (equityCurve.length > 0) {
    btChart.data.labels = equityCurve.map(e => fmtShort(e.time || e.timestamp || ''));
    btChart.data.datasets[0].data = equityCurve.map(e => e.pnl ?? e.cumulative_pnl ?? 0);

    // Color points by regime
    const ptColors = equityCurve.map(e => {
      const r = (e.regime || '').toLowerCase();
      if (r.includes('bull'))  return '#22c55e';
      if (r.includes('bear'))  return '#ef4444';
      if (r)                   return '#3b82f6';
      // Fallback: color by trade pnl direction
      return (e.trade_pnl ?? 0) >= 0 ? '#22c55e' : '#ef4444';
    });
    btChart.data.datasets[0].pointBackgroundColor = ptColors;
    btChart.data.datasets[0].pointBorderColor = ptColors;
    // Store regimes for tooltip
    btChart.data.datasets[0].regimes = equityCurve.map(e => e.regime || '');
    btChart.update('none');
    document.getElementById('bt-chart-label').textContent = equityCurve.length + ' data points';
  }

  // ── Notional exposure chart (from equity_curve.notional) ──
  if (equityCurve.length > 0 && equityCurve[0].notional !== undefined) {
    // Downsample: the equity curve has one entry per bar (~hourly)
    // Show one point per day (every 24 bars) to keep the chart readable
    const step = Math.max(1, Math.floor(equityCurve.length / 500));
    const sampled = equityCurve.filter((_, i) => i % step === 0);
    ntChart.data.labels = sampled.map(e => fmtShort(e.time || e.timestamp || ''));
    ntChart.data.datasets[0].data = sampled.map(e => e.notional || 0);
    ntChart.update('none');
    const maxNotional = Math.max(...sampled.map(e => e.notional || 0));
    document.getElementById('bt-notional-label').textContent =
      'Peak: ' + fmtK(maxNotional) + ' \u00b7 ' + sampled.length + ' samples';

    // ── Broker Margin chart (actual collateral held by broker over time) ──
    blkChart.data.labels = sampled.map(e => fmtShort(e.time || e.timestamp || ''));
    blkChart.data.datasets[0].data = sampled.map(e => e.margin || 0);
    blkChart.update('none');
    const maxMargin = Math.max(...sampled.map(e => e.margin || 0));
    document.getElementById('bt-blocked-label').textContent =
      'Peak: ' + fmtK(maxMargin);
  }

  // ── Exposure & ROI panel ──
  const exposure = bt.exposure || {};
  const expPanel = document.getElementById('bt-exposure-roi');
  if (exposure.notional_max) {
    const pnl = metrics.cumulative_pnl ?? metrics.total_pnl ?? 0;
    let html = '';

    // Card 1: Required Capital (the headline card)
    const reqCap = exposure.required_capital || 0;
    const reqDD = exposure.required_capital_max_dd || 0;
    const reqMult = exposure.required_capital_dd_multiplier || 3;
    const reqMargin = exposure.required_capital_margin || 0;
    const reqDDBuf = exposure.required_capital_dd_buffer || 0;
    const roiReqAnn = exposure.roi_required_capital_ann || 0;
    const roiReqCls = roiReqAnn >= 0 ? 'pos' : 'neg';
    html += `
      <div class="exp-card" style="border:1px solid var(--accent);">
        <div class="exp-card-title" style="color:var(--accent);">Required Capital</div>
        <div class="roi-highlight ${roiReqCls}">${roiReqAnn.toFixed(1)}%/yr</div>
        <div class="roi-ann">Annual return on capital needed to run this strategy</div>
        <div class="exp-stat" style="margin-top:10px;"><span class="es-label">Peak Margin (broker deposit)</span>
          <span class="es-val">${fmtK(reqMargin)}</span></div>
        <div class="exp-stat"><span class="es-label">Max Drawdown</span>
          <span class="es-val">${fmtK(reqDD)}</span></div>
        <div class="exp-stat"><span class="es-label">\u00d7 ${reqMult} (loss buffer)</span>
          <span class="es-val">${fmtK(reqDDBuf)}</span></div>
        <div class="exp-stat" style="border-top:1px solid var(--border); padding-top:6px; margin-top:6px;"><span class="es-label" style="font-weight:600;">= Deposit at Broker</span>
          <span class="es-val" style="font-weight:600;">${fmtK(reqCap)}</span></div>
        <div class="exp-stat" style="margin-top:8px; font-size:11px; color:var(--text-dim);">Peak Margin is the collateral your broker holds at your largest position. The 3\u00d7 Max Drawdown buffer ensures you can absorb the worst historical loss, keep trading through a repeat, and still have a safety margin for worse-than-historical scenarios.</div>
      </div>`;

    // Card 2: Investment Amount
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Investment Amount</div>
        <div class="exp-stat"><span class="es-label">Range</span>
          <span class="es-val">${fmtK(exposure.investment_min || exposure.notional_min)} \u2014 ${fmtK(exposure.investment_max || exposure.notional_max)}</span></div>
        <div class="exp-stat"><span class="es-label">Average</span>
          <span class="es-val">${fmtK(exposure.investment_mean || exposure.notional_mean)}</span></div>
        <div class="exp-stat"><span class="es-label">Median</span>
          <span class="es-val">${fmtK(exposure.investment_median || exposure.notional_median)}</span></div>
        <div class="exp-stat"><span class="es-label">Entries</span>
          <span class="es-val">${exposure.total_entries}</span></div>
        <div class="exp-stat" style="margin-top:8px; font-size:11px; color:var(--text-dim);">The dollar value of each position at entry (contracts \u00d7 price \u00d7 multiplier). This is the market value at risk, not the margin deposit.</div>
      </div>`;

    // Card 3: Margin Requirement
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Margin Requirement</div>
        <div class="exp-stat"><span class="es-label">Peak Margin</span>
          <span class="es-val">${fmtK(exposure.margin_max)}</span></div>
        <div class="exp-stat"><span class="es-label">Avg Margin</span>
          <span class="es-val">${fmtK(exposure.margin_mean)}</span></div>
        <div class="exp-stat"><span class="es-label">Contracts Range</span>
          <span class="es-val">${exposure.contracts_min} \u2014 ${exposure.contracts_max}</span></div>
        <div class="exp-stat"><span class="es-label">@ $${(exposure.margin_per_contract || 1500).toLocaleString()}/contract</span>
          <span class="es-val es-dim" style="color:var(--text-dim);">${fmtK(exposure.margin_min)} \u2014 ${fmtK(exposure.margin_max)}</span></div>
        <div class="exp-stat" style="margin-top:8px; font-size:11px; color:var(--text-dim);">The deposit IB holds as collateral while the position is open. For futures, this is much less than the investment amount \u2014 you control a large position with a small deposit (leverage).</div>
      </div>`;

    // Card 4: Annualized ROI on Peak Investment
    const roiInvAnn = exposure.roi_max_investment_ann || exposure.roi_max_notional_ann || 0;
    const roiInvCls = roiInvAnn >= 0 ? 'pos' : 'neg';
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Annual ROI on Peak Investment</div>
        <div class="roi-highlight ${roiInvCls}">${roiInvAnn.toFixed(1)}%/yr</div>
        <div class="roi-ann">PnL \u00f7 peak investment \u00f7 years</div>
        <div class="exp-stat" style="margin-top:8px;"><span class="es-label">Peak Investment</span>
          <span class="es-val">${fmtK(exposure.investment_max || exposure.notional_max)}</span></div>
        <div class="exp-stat"><span class="es-label">Total PnL</span>
          <span class="es-val ${colorClass(pnl)}">${fmt(pnl)}</span></div>
        <div class="exp-stat"><span class="es-label">Period</span>
          <span class="es-val">${(exposure.backtest_years || 0).toFixed(1)} years</span></div>
      </div>`;

    // Card 5: Annualized ROI on Avg Investment
    const roiAvgInvAnn = exposure.roi_avg_investment_ann || exposure.roi_avg_notional_ann || 0;
    const roiAvgCls = roiAvgInvAnn >= 0 ? 'pos' : 'neg';
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Annual ROI on Avg Investment</div>
        <div class="roi-highlight ${roiAvgCls}">${roiAvgInvAnn.toFixed(1)}%/yr</div>
        <div class="roi-ann">PnL \u00f7 avg investment \u00f7 years</div>
        <div class="exp-stat" style="margin-top:8px;"><span class="es-label">Avg Investment</span>
          <span class="es-val">${fmtK(exposure.investment_mean || exposure.notional_mean)}</span></div>
        <div class="exp-stat"><span class="es-label">Total PnL</span>
          <span class="es-val ${colorClass(pnl)}">${fmt(pnl)}</span></div>
      </div>`;

    // Card 6: Capital Utilization
    const tradeable = exposure.tradeable_days || 0;
    const exposed = exposure.days_exposed || 0;
    const notExposed = exposure.days_not_exposed || 0;
    const negMomDays = exposure.neg_momentum_days || 0;
    const totalCal = exposure.total_calendar_days || 0;
    const utilPct = exposure.utilization_pct || 0;
    const utilTotalPct = exposure.utilization_total_pct || 0;
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Capital Utilization</div>
        <div class="roi-highlight ${utilPct >= 50 ? 'pos' : 'neg'}">${utilPct.toFixed(1)}%</div>
        <div class="roi-ann">of tradeable days with capital at risk</div>
        <div class="exp-stat" style="margin-top:8px;"><span class="es-label">Calendar Days</span>
          <span class="es-val">${totalCal.toLocaleString()}</span></div>
        <div class="exp-stat"><span class="es-label">Neg-Momentum (no trade)</span>
          <span class="es-val">${negMomDays.toLocaleString()}</span></div>
        <div class="exp-stat"><span class="es-label">Tradeable Days</span>
          <span class="es-val">${tradeable.toLocaleString()}</span></div>
        <div class="exp-stat"><span class="es-label">Days Exposed</span>
          <span class="es-val">${exposed.toLocaleString()}</span></div>
        <div class="exp-stat"><span class="es-label">Days Not Exposed</span>
          <span class="es-val">${notExposed.toLocaleString()}</span></div>
        <div class="exp-stat"><span class="es-label">% of Total Calendar</span>
          <span class="es-val">${utilTotalPct.toFixed(1)}%</span></div>
      </div>`;

    // Card 7: Annualized ROI on Peak/Avg Capital Invested
    const peakCap = exposure.peak_capital_invested || 0;
    const avgCap = exposure.avg_capital_invested || 0;
    const roiPeakAnn = exposure.roi_peak_capital_ann || 0;
    const roiAvgCapAnn = exposure.roi_avg_capital_ann || 0;
    const roiPeakCls = roiPeakAnn >= 0 ? 'pos' : 'neg';
    html += `
      <div class="exp-card">
        <div class="exp-card-title">Annual ROI on Daily Capital</div>
        <div class="roi-highlight ${roiPeakCls}">${roiPeakAnn.toFixed(1)}%/yr</div>
        <div class="roi-ann">on peak daily investment amount</div>
        <div class="exp-stat" style="margin-top:8px;"><span class="es-label">Peak Daily Investment</span>
          <span class="es-val">${fmtK(peakCap)}</span></div>
        <div class="exp-stat"><span class="es-label">Avg Daily Investment</span>
          <span class="es-val">${fmtK(avgCap)}</span></div>
        <div class="exp-stat"><span class="es-label">Annual ROI on Avg</span>
          <span class="es-val ${roiAvgCapAnn >= 0 ? 'pos' : 'neg'}">${roiAvgCapAnn.toFixed(1)}%/yr</span></div>
      </div>`;

    expPanel.innerHTML = html;
  }

  // ── Performance by regime cards ──
  const cardsEl = document.getElementById('bt-regime-cards');
  // Build from regime_summary / metrics_by_regime or derive from trades
  let summaryData = null;

  // Normalise: regime_summary could be array [{regime, ...}] or object {BULL: {...}}
  if (Array.isArray(regimeSummary)) {
    summaryData = regimeSummary;
  } else if (typeof regimeSummary === 'object' && Object.keys(regimeSummary).length > 0) {
    summaryData = Object.entries(regimeSummary).map(([k, v]) => ({ regime: k, ...v }));
  } else {
    // Derive from trades list
    const byRegime = {};
    for (const t of trades) {
      const r = t.regime || 'Unknown';
      if (!byRegime[r]) byRegime[r] = { regime: r, trades: 0, wins: 0, pnl: 0, periods: 0, bars: 0, win_rate: 0 };
      if (t.action === 'SELL' || t.action === 'COVER') {
        const tpnl = t.net_pnl ?? t.pnl ?? 0;
        byRegime[r].trades++;
        byRegime[r].pnl += tpnl;
        if (tpnl >= 0) byRegime[r].wins++;
      }
    }
    // Count periods
    for (const rp of regimes) {
      const r = rp.regime || 'Unknown';
      if (!byRegime[r]) byRegime[r] = { regime: r, trades: 0, wins: 0, pnl: 0, periods: 0, bars: 0, win_rate: 0 };
      byRegime[r].periods++;
      byRegime[r].bars += rp.bars || rp.duration_bars || 0;
    }
    // Compute win_rate for derived data
    for (const k in byRegime) {
      const e = byRegime[k];
      e.win_rate = e.trades > 0 ? ((e.wins / e.trades) * 100).toFixed(1) : '0.0';
    }
    summaryData = Object.values(byRegime);
  }

  // Enrich summaryData with periods/bars from regimes array if not already present
  if (summaryData && regimes && regimes.length > 0) {
    const periodsByRegime = {};
    for (const rp of regimes) {
      const r = rp.regime || 'Unknown';
      if (!periodsByRegime[r]) periodsByRegime[r] = { periods: 0, bars: 0 };
      periodsByRegime[r].periods++;
      periodsByRegime[r].bars += rp.bars || rp.duration_bars || 0;
    }
    for (const s of summaryData) {
      const r = s.regime || 'Unknown';
      if (periodsByRegime[r]) {
        if (!s.periods && !s.total_periods) s.periods = periodsByRegime[r].periods;
        if (!s.bars && !s.total_bars) s.bars = periodsByRegime[r].bars;
      }
    }
  }

  if (summaryData && summaryData.length > 0) {
    let cardsHtml = '';
    for (const s of summaryData) {
      const r = s.regime || 'Unknown';
      const rc = regimeClass(r);
      const dname = s.display_name || regimeDisplayName(r);
      const totalTrades = s.trades ?? s.total_trades ?? 0;
      const wr = s.win_rate ?? (totalTrades > 0 ? (((s.wins ?? s.winning_trades ?? 0) / totalTrades) * 100).toFixed(1) : '0.0');
      const totalPnl = s.pnl ?? s.total_pnl ?? s.cumulative_pnl ?? 0;
      const periods = s.periods ?? s.total_periods ?? 0;
      const bars = s.bars ?? s.total_bars ?? 0;
      const clusterDays = s.cluster_days ?? 0;
      const daysExposed = s.days_exposed ?? 0;
      const utilPct = s.utilization_pct ?? 0;
      const peakCap = s.peak_capital ?? 0;
      const avgCap = s.avg_capital ?? 0;
      const roiPeak = s.roi_peak ?? 0;
      const roiPeakAnn = s.roi_peak_ann ?? 0;
      const roiAvg = s.roi_avg ?? 0;
      const roiAvgAnn = s.roi_avg_ann ?? 0;
      const bestTrade = s.best_trade ?? 0;
      const worstTrade = s.worst_trade ?? 0;
      const profitFactor = s.profit_factor ?? 0;
      const avgPnl = s.avg_pnl ?? 0;
      const isNegMom = (r === 'neg_momentum_skip');

      cardsHtml += `
        <div class="regime-card">
          <div class="regime-card-header">
            <span class="regime-card-title ${rc}">${dname}</span>
            ${regimeChip(r)}
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Periods / Days</span>
            <span class="rcs-val">${periods} / ${clusterDays.toLocaleString()}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Trades</span>
            <span class="rcs-val">${totalTrades}${isNegMom ? ' (no trading)' : ''}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Win Rate</span>
            <span class="rcs-val">${wr}%</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Total PnL</span>
            <span class="rcs-val ${colorClass(totalPnl)}">${fmt(totalPnl)}</span>
          </div>`;

      // Extra trade stats (non-neg_momentum only)
      if (totalTrades > 0) {
        cardsHtml += `
          <div class="regime-card-stat">
            <span class="rcs-label">Avg PnL / Trade</span>
            <span class="rcs-val ${colorClass(avgPnl)}">${fmt(avgPnl)}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Profit Factor</span>
            <span class="rcs-val">${profitFactor === Infinity ? '\u221e' : Number(profitFactor).toFixed(2)}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Best / Worst</span>
            <span class="rcs-val"><span class="pos">${fmt(bestTrade)}</span> / <span class="neg">${fmt(worstTrade)}</span></span>
          </div>`;
      }

      // Capital utilization section
      cardsHtml += `
          <div style="border-top: 1px solid rgba(45,49,65,0.5); margin-top:8px; padding-top:8px;">
            <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-dim); margin-bottom:6px;">Capital Utilization</div>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Days Exposed / Available</span>
            <span class="rcs-val">${daysExposed} / ${clusterDays}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Utilization</span>
            <span class="rcs-val">${utilPct.toFixed ? utilPct.toFixed(1) : utilPct}%</span>
          </div>`;

      if (peakCap > 0) {
        cardsHtml += `
          <div class="regime-card-stat">
            <span class="rcs-label">Peak Capital</span>
            <span class="rcs-val">${fmtK(peakCap)}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">Avg Capital</span>
            <span class="rcs-val">${fmtK(avgCap)}</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">ROI (peak cap)</span>
            <span class="rcs-val ${colorClass(roiPeak)}">${roiPeak.toFixed ? roiPeak.toFixed(1) : roiPeak}% / ${roiPeakAnn.toFixed ? roiPeakAnn.toFixed(1) : roiPeakAnn}% ann</span>
          </div>
          <div class="regime-card-stat">
            <span class="rcs-label">ROI (avg cap)</span>
            <span class="rcs-val ${colorClass(roiAvg)}">${roiAvg.toFixed ? roiAvg.toFixed(1) : roiAvg}% / ${roiAvgAnn.toFixed ? roiAvgAnn.toFixed(1) : roiAvgAnn}% ann</span>
          </div>`;
      }

      cardsHtml += `</div>`;
    }
    cardsEl.innerHTML = cardsHtml;
  } else {
    cardsEl.innerHTML = '<div class="empty-state" style="padding:20px;">No regime performance data available</div>';
  }

  // ── Regime periods table ──
  const regimeTbody = document.getElementById('bt-regime-tbody');
  document.getElementById('bt-regime-count').textContent = regimes.length + ' periods';
  if (regimes.length > 0) {
    let html = '';
    regimes.forEach((rp, i) => {
      const r = rp.regime || '—';
      const rc = regimeClass(r);
      const startT = rp.start || rp.start_time || '';
      const endT   = rp.end   || rp.end_time   || '';
      const durationBars = rp.bars || rp.duration_bars || rp.duration || 0;
      const durationDays = durationBars > 0 ? (durationBars / 24).toFixed(1) + 'd' : '—';
      const startPrice = rp.start_price ?? rp.open_price ?? null;
      const endPrice   = rp.end_price   ?? rp.close_price ?? null;
      let ret = '—';
      const retPct = rp.return_pct;
      if (retPct !== null && retPct !== undefined) {
        ret = `<span class="${colorClass(retPct)}">${retPct >= 0 ? '+' : ''}${Number(retPct).toFixed(2)}%</span>`;
      } else if (startPrice && endPrice && startPrice > 0) {
        const calcPct = ((endPrice - startPrice) / startPrice * 100);
        ret = `<span class="${colorClass(calcPct)}">${calcPct >= 0 ? '+' : ''}${calcPct.toFixed(2)}%</span>`;
      }
      html += `<tr class="regime-${rc}">
        <td>${i + 1}</td>
        <td>${regimeChip(r)}</td>
        <td>${fmtTime(startT)}</td>
        <td>${fmtTime(endT)}</td>
        <td>${durationDays}</td>
        <td>${startPrice ? fmtPrice(startPrice) : '—'}</td>
        <td>${endPrice   ? fmtPrice(endPrice)   : '—'}</td>
        <td>${ret}</td>
      </tr>`;
    });
    regimeTbody.innerHTML = html;
  } else {
    regimeTbody.innerHTML = '<tr><td colspan="8" class="empty-state">No regime periods</td></tr>';
  }

  // ── Trade history (backtest) ──
  const btTbody = document.getElementById('bt-trade-tbody');
  const allTrades = trades.length > 0 ? trades : (data.trades || []);
  document.getElementById('bt-trade-count').textContent = allTrades.length + ' total entries';
  if (allTrades.length > 0) {
    const recent = allTrades.slice().reverse().slice(0, 50);
    let html = '';
    for (const t of recent) {
      const isClosed = t.action === 'SELL' || t.action === 'COVER';
      const tradeRegime = t.regime || '—';
      const rc = regimeClass(tradeRegime);
      html += `<tr class="regime-${rc}">`;
      html += `<td>${fmtTime(t.time)}</td>`;
      html += `<td class="${actionClass(t.action)}">${t.action}</td>`;
      const btSide = t.side || (t.action === 'BUY' || t.action === 'SELL' || t.action === 'PYRAMID' ? 'long' : t.action === 'SHORT' || t.action === 'COVER' ? 'short' : '');
      html += `<td>${btSide ? '<span class="side-'+btSide+'">'+btSide.toUpperCase()+'</span>' : '—'}</td>`;
      html += `<td>${fmtPrice(t.fill_price || t.price)}</td>`;
      html += `<td>${isClosed ? fmtPrice(t.entry_price) : '—'}</td>`;
      html += `<td>${t.notional ? fmtK(t.notional) : '—'}</td>`;
      const tradePnl = t.net_pnl ?? t.pnl;
      html += `<td class="${colorClass(tradePnl)}">${isClosed ? fmt(tradePnl) : '—'}</td>`;
      html += `<td>${tradeRegime !== '—' ? regimeChip(tradeRegime) : '—'}</td>`;
      html += `<td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--font);font-size:11px;color:var(--text-dim);">${t.reason || '—'}</td>`;
      html += '</tr>';
    }
    btTbody.innerHTML = html;
  } else {
    btTbody.innerHTML = '<tr><td colspan="9" class="empty-state">No backtest trades</td></tr>';
  }
}

// ── Update UI (live mode) ─────────────────────────────
function updateLiveView(data) {
  const m = data.metrics || {};
  const s = data.state || {};
  const trades = data.trades || [];
  const ss = s.strategy_status || {};
  const pos = ss.position || {};

  // ── Status badge ──
  const badge = document.getElementById('status-badge');
  const statusText = document.getElementById('status-text');
  if (s.running && !s.paused) {
    badge.className = 'status-badge';
    statusText.textContent = 'Trading';
  } else if (s.paused) {
    badge.className = 'status-badge paused';
    statusText.textContent = 'Paused';
  } else {
    badge.className = 'status-badge offline';
    statusText.textContent = s.regime ? 'Stopped' : 'Waiting...';
  }

  // ── Pause/Resume toggle ──
  const btnPause  = document.getElementById('btn-pause');
  const btnResume = document.getElementById('btn-resume');
  if (s.paused) {
    btnPause.style.display  = 'none';
    btnResume.style.display = '';
  } else {
    btnPause.style.display  = '';
    btnResume.style.display = 'none';
  }

  // ── Regime indicator (live mode) ──
  const regimeInd = document.getElementById('regime-indicator');
  const detectedRegime = s.detected_regime || s.regime || '';
  const regimeConf = s.regime_confidence;
  if (detectedRegime) {
    const rc = regimeClass(detectedRegime);
    let text = 'Regime: ' + regimeDisplayName(detectedRegime);
    if (regimeConf !== undefined && regimeConf !== null) {
      text += ' (' + Math.round(Number(regimeConf) * 100) + '%)';
    }
    regimeInd.className = 'regime-indicator ' + rc;
    regimeInd.innerHTML = '<span class="regime-dot"></span>' + text;
    regimeInd.style.display = '';
  } else {
    regimeInd.style.display = 'none';
  }

  // ── KPIs ──
  const pnlEl = document.getElementById('kpi-pnl');
  pnlEl.textContent = fmt(m.cumulative_pnl);
  pnlEl.className = 'kpi-value ' + colorClass(m.cumulative_pnl);
  document.getElementById('kpi-pnl-sub').textContent = (m.total_trades||0) + ' trades';

  document.getElementById('kpi-winrate').textContent = (m.win_rate||0) + '%';
  const wins = Math.round((m.win_rate||0)/100 * (m.total_trades||0));
  const losses = (m.total_trades||0) - wins;
  document.getElementById('kpi-winrate-sub').textContent = wins + 'W / ' + losses + 'L';

  document.getElementById('kpi-dd').textContent = fmt(-(m.max_drawdown||0));

  const avgEl = document.getElementById('kpi-avg');
  avgEl.textContent = fmt(m.avg_pnl);
  avgEl.className = 'kpi-value ' + colorClass(m.avg_pnl);

  const pf = m.profit_factor;
  document.getElementById('kpi-pf').textContent = 'PF: ' + (pf === Infinity ? '∞' : (pf||0).toFixed(2));

  document.getElementById('kpi-best').textContent = fmt(m.best_trade);
  document.getElementById('kpi-worst').textContent = fmt(m.worst_trade);

  // ── Account & Exposure Panel ──
  const paperBalance   = data.paper_balance   ?? s.paper_balance   ?? 1000000;
  const maxExposure    = data.max_exposure    ?? s.max_exposure    ?? 500000;
  const curExposure    = data.current_exposure ?? s.current_exposure ?? 0;
  const curContracts   = data.current_contracts ?? s.current_contracts ?? 0;
  const maxContracts   = data.max_contracts   ?? s.max_contracts   ?? 3;
  const contractSymbol = data.contract_symbol ?? s.contract_symbol ?? '';
  const daysToExpiry   = data.days_to_expiry  ?? s.days_to_expiry;

  currentConfigValues.paper_balance = paperBalance;
  currentConfigValues.max_exposure = maxExposure;
  const cooldownHours = data.cooldown_hours ?? 3;
  currentConfigValues.cooldown_hours = cooldownHours;

  if (!document.getElementById('cfg-input-paper_balance')) {
    const balEl = document.getElementById('ap-balance');
    if (balEl) balEl.innerHTML = fmtK(paperBalance) + ' <span class="edit-icon">&#x270E;</span>';
  }
  if (!document.getElementById('cfg-input-max_exposure')) {
    const expEl = document.getElementById('ap-max-exp');
    if (expEl) expEl.innerHTML = fmtK(maxExposure) + ' <span class="edit-icon">&#x270E;</span>';
  }
  if (!document.getElementById('cfg-input-cooldown_hours')) {
    const cdEl = document.getElementById('ap-cooldown');
    if (cdEl) cdEl.innerHTML = cooldownHours + 'h <span class="edit-icon">&#x270E;</span>';
  }

  document.getElementById('ap-cur-ct').textContent = curContracts;
  document.getElementById('ap-max-ct-display').textContent = maxContracts;
  currentConfigValues.max_contracts = maxContracts;
  currentConfigValues._currentContracts = curContracts;
  if (!document.getElementById('cfg-input-max_contracts')) {
    const mctEl = document.getElementById('ap-max-ct');
    if (mctEl) mctEl.innerHTML = 'max: ' + maxContracts + ' <span class="edit-icon">&#x270E;</span>';
  }
  document.getElementById('ap-symbol').textContent = contractSymbol || '—';
  document.getElementById('ap-cur-exp').textContent = fmtK(curExposure);

  // Expiry display
  const expiryEl    = document.getElementById('ap-expiry');
  const expirySub   = document.getElementById('ap-expiry-sub');
  if (daysToExpiry !== null && daysToExpiry !== undefined) {
    const d = Number(daysToExpiry);
    const cls = d <= 3 ? 'expiry-red' : d <= 10 ? 'expiry-amber' : 'expiry-green';
    expiryEl.textContent = d + 'd';
    expiryEl.className = 'ap-value ' + cls;
    expirySub.textContent = d <= 3 ? '⚠ Expiring soon' : d <= 10 ? 'Rolling soon' : 'Until expiry';
  } else {
    expiryEl.textContent = '—';
    expiryEl.className = 'ap-value';
    expirySub.textContent = '';
  }

  // Recalibration info
  const recalCount = data.recalibrations ?? s.recalibrations ?? 0;
  const recalTime = data.last_recal_time ?? s.last_recal_time;
  document.getElementById('ap-recal-count').textContent = recalCount + ' done';
  if (recalTime) {
    document.getElementById('ap-recal-time').textContent = 'Last: ' + fmtTime(recalTime);
  } else {
    document.getElementById('ap-recal-time').textContent = 'Pending first calibration';
  }

  // Gauge
  updateGauge(curExposure, maxExposure);

  // ── Equity curve ──
  const eq = m.equity_curve || [];
  if (eq.length > 0) {
    chart.data.labels = eq.map(e => fmtShort(e.time));
    chart.data.datasets[0].data = eq.map(e => e.pnl);
    chart.data.datasets[0].pointBackgroundColor = eq.map(e =>
      e.trade_pnl >= 0 ? '#22c55e' : '#ef4444'
    );
    chart.data.datasets[0].pointBorderColor = eq.map(e =>
      e.trade_pnl >= 0 ? '#22c55e' : '#ef4444'
    );
    chart.update('none');
    document.getElementById('chart-label').textContent =
      eq.length + ' closed trades';
  }

  // ── Current Position ──
  const posBody  = document.getElementById('position-body');
  const posBadge = document.getElementById('pos-side');

  if (pos.side && pos.side !== 'flat') {
    posBadge.className = pos.side === 'long' ? 'side-long' : 'side-short';
    posBadge.textContent = pos.side.toUpperCase();

    const lastPrice = s.last_price || 0;
    let unrealPnl = 0;
    let unrealPct = 0;
    if (pos.entry_price > 0 && lastPrice > 0) {
      if (pos.side === 'long') {
        unrealPnl = (lastPrice - pos.entry_price) * 0.1 * (pos.contracts||1);
        unrealPct = (lastPrice / pos.entry_price - 1) * 100;
      } else {
        unrealPnl = (pos.entry_price - lastPrice) * 0.1 * (pos.contracts||1);
        unrealPct = (pos.entry_price / lastPrice - 1) * 100;
      }
    }

    let html = '';
    html += row('Entry Price', fmtPrice(pos.entry_price));
    html += row('Current Price', fmtPrice(lastPrice));
    html += row('Unrealized PnL', `<span class="${colorClass(unrealPnl)}">${fmt(unrealPnl)} (${unrealPct >= 0 ? '+' : ''}${unrealPct.toFixed(2)}%)</span>`);
    html += row('Contracts', pos.contracts || 1);
    html += row('Target', fmtPrice(pos.target_price));

    if (pos.side === 'short') {
      html += row('Stop Loss', fmtPrice(pos.stop_loss));
      html += row('Trailing Stop', fmtPrice(pos.trailing_stop));
    } else {
      html += row('Stop Loss', '<span style="color:var(--text-dim)">None (patient)</span>');
    }

    if (pos.entry_time) {
      const held = Math.round((Date.now() - new Date(pos.entry_time).getTime()) / 3600000);
      html += row('Held', held + 'h');
    }

    // Range position bar
    if (pos.support > 0 && pos.resistance > 0 && lastPrice > 0) {
      const rngPos = Math.max(0, Math.min(1, (lastPrice - pos.support) / (pos.resistance - pos.support)));
      html += `<div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-dim);">
          <span>S: ${fmtPrice(pos.support)}</span>
          <span>${(rngPos*100).toFixed(0)}% of range</span>
          <span>R: ${fmtPrice(pos.resistance)}</span>
        </div>
        <div class="range-bar">
          <div class="fill" style="width:${rngPos*100}%;background:${pos.side==='long' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}"></div>
          <div class="marker" style="left:${rngPos*100}%"></div>
        </div>
      </div>`;
    }

    posBody.innerHTML = html;
  } else {
    posBadge.className = 'side-flat';
    posBadge.textContent = 'FLAT';

    let html = '<div class="empty-state">No active position';
    if (ss.cooldown_until) {
      html += '<br><span style="font-size:11px;">Cooldown until ' + fmtTime(ss.cooldown_until) + '</span>';
    }
    if (ss.support > 0) {
      html += '<br><span style="font-size:11px;">Range: ' + fmtPrice(ss.support) + ' — ' + fmtPrice(ss.resistance) + '</span>';
    }
    html += '</div>';
    posBody.innerHTML = html;
  }

  // ── Long vs Short breakdown ──
  const longPnlEl = document.getElementById('long-pnl');
  longPnlEl.textContent = fmt(m.long_pnl);
  longPnlEl.className = 'breakdown-pnl ' + colorClass(m.long_pnl);
  document.getElementById('long-detail').textContent =
    (m.long_trades||0) + ' trades, ' + (m.long_wins||0) + ' wins';

  const shortPnlEl = document.getElementById('short-pnl');
  shortPnlEl.textContent = fmt(m.short_pnl);
  shortPnlEl.className = 'breakdown-pnl ' + colorClass(m.short_pnl);
  document.getElementById('short-detail').textContent =
    (m.short_trades||0) + ' trades, ' + (m.short_wins||0) + ' wins';

  // ── Trade table (most recent first, limit 20) ──
  const tbody = document.getElementById('trade-tbody');
  if (trades.length > 0) {
    const recent = trades.slice().reverse().slice(0, 20);
    let html = '';
    for (const t of recent) {
      const isClosed = t.action === 'SELL' || t.action === 'COVER';
      html += '<tr>';
      html += `<td>${fmtTime(t.time)}</td>`;
      html += `<td class="${actionClass(t.action)}">${t.action}</td>`;
      const liveSide = t.side || (t.action === 'BUY' || t.action === 'SELL' ? 'long' : t.action === 'SHORT' || t.action === 'COVER' ? 'short' : '');
      html += `<td>${liveSide ? '<span class="side-'+liveSide+'">'+liveSide.toUpperCase()+'</span>' : '—'}</td>`;
      html += `<td>${fmtPrice(t.fill_price || t.price)}</td>`;
      html += `<td>${isClosed ? fmtPrice(t.entry_price) : '—'}</td>`;
      const livePnl = t.net_pnl ?? t.pnl;
      html += `<td class="${colorClass(livePnl)}">${isClosed ? fmt(livePnl) : '—'}</td>`;
      html += `<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--font);font-size:11px;color:var(--text-dim);">${t.reason || '—'}</td>`;
      html += '</tr>';
    }
    tbody.innerHTML = html;
    document.getElementById('trade-count').textContent =
      trades.length + ' total entries';
  }

  // ── Footer ──
  document.getElementById('footer').textContent =
    'Last updated: ' + new Date().toLocaleTimeString() +
    (s.last_price ? '  |  BTC: ' + fmtPrice(s.last_price) : '') +
    (ss.bars_in_window ? '  |  ' + ss.bars_in_window + ' bars in window' : '');

  // ── Entry parameters panel ──
  const liveRsi = ss.rsi ?? 50;
  const liveAdx = ss.adx ?? 20;
  const liveRngPos = ss.range_position ?? 0.5;
  const livePrice = s.last_price || 0;
  const lastSig = ss.last_signal_reason || '';
  const cooldownActive = !!ss.cooldown_until;

  document.getElementById('param-price').textContent = livePrice > 0 ? fmtPrice(livePrice) : '—';
  document.getElementById('param-rng-pos').textContent = (liveRngPos * 100).toFixed(1) + '%';
  document.getElementById('param-rsi').textContent = liveRsi.toFixed(1);
  document.getElementById('param-adx').textContent = liveAdx.toFixed(1);
  document.getElementById('param-signal').textContent = lastSig || 'Waiting for signal...';
  document.getElementById('signal-status').textContent = cooldownActive ? '⏳ In cooldown' : (pos.side && pos.side !== 'flat' ? '📊 In position' : '👀 Scanning');

  // Long checks
  const chkLongRng = document.getElementById('chk-long-rng');
  chkLongRng.textContent = liveRngPos <= 0.30 ? '\u2713' : '\u2717';
  chkLongRng.className = 'param-check ' + (liveRngPos <= 0.30 ? 'pass' : 'fail');

  const chkLongRsi = document.getElementById('chk-long-rsi');
  chkLongRsi.textContent = liveRsi < 45 ? '\u2713' : '\u2717';
  chkLongRsi.className = 'param-check ' + (liveRsi < 45 ? 'pass' : 'fail');

  const chkLongCd = document.getElementById('chk-long-cd');
  chkLongCd.textContent = !cooldownActive ? '\u2713' : '\u2717';
  chkLongCd.className = 'param-check ' + (!cooldownActive ? 'pass' : 'fail');

  // Short checks
  const chkShortRng = document.getElementById('chk-short-rng');
  chkShortRng.textContent = liveRngPos >= 0.70 ? '\u2713' : '\u2717';
  chkShortRng.className = 'param-check ' + (liveRngPos >= 0.70 ? 'pass' : 'fail');

  const chkShortRsi = document.getElementById('chk-short-rsi');
  chkShortRsi.textContent = liveRsi > 55 ? '\u2713' : '\u2717';
  chkShortRsi.className = 'param-check ' + (liveRsi > 55 ? 'pass' : 'fail');

  const chkShortAdx = document.getElementById('chk-short-adx');
  chkShortAdx.textContent = liveAdx < 25 ? '\u2713' : '\u2717';
  chkShortAdx.className = 'param-check ' + (liveAdx < 25 ? 'pass' : 'fail');

  const chkShortCd = document.getElementById('chk-short-cd');
  chkShortCd.textContent = !cooldownActive ? '\u2713' : '\u2717';
  chkShortCd.className = 'param-check ' + (!cooldownActive ? 'pass' : 'fail');
}

// ── Top-level update dispatcher ──────────────────────
let currentMode = null;

function update(data) {
  const mode = data.mode || (data.state && data.state.mode) || 'live';
  const isBacktest = mode === 'backtest';

  // Switch views if mode changed
  if (mode !== currentMode) {
    currentMode = mode;
    document.getElementById('live-view').style.display = isBacktest ? 'none' : 'block';
    document.getElementById('bt-view').style.display   = isBacktest ? 'block' : 'none';

    // Mode badge
    const modeBadge = document.getElementById('mode-badge');
    modeBadge.textContent = isBacktest ? 'BACKTEST' : 'LIVE PAPER TRADING';
    modeBadge.className = 'mode-badge ' + (isBacktest ? 'backtest' : 'live');

    // Control bar vs disabled note
    document.getElementById('control-bar').style.display        = isBacktest ? 'none' : '';
    document.getElementById('bt-controls-note').style.display   = isBacktest ? ''     : 'none';

    // Hide regime indicator in backtest mode (has its own regime UI)
    if (isBacktest) document.getElementById('regime-indicator').style.display = 'none';

    // Set status badge for backtest mode
    if (isBacktest) {
      const badge = document.getElementById('status-badge');
      const statusText = document.getElementById('status-text');
      badge.className = 'status-badge paused';  // amber style for backtest
      statusText.textContent = 'Backtest';
    }
  }

  // Update footer timestamp always
  document.getElementById('footer').textContent =
    'Last updated: ' + new Date().toLocaleTimeString() +
    (isBacktest ? '  |  Backtest mode' : '');

  if (isBacktest) {
    updateBacktestView(data);
  } else {
    updateLiveView(data);
  }
}

function row(label, value) {
  return `<div class="position-row"><span class="label">${label}</span><span class="value">${value}</span></div>`;
}

// ── Polling ───────────────────────────────────────────
let failCount = 0;

async function poll() {
  try {
    const res = await fetch('/api/all');
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    update(data);
    failCount = 0;

    // Update status badge connectivity
    const badge = document.getElementById('status-badge');
    const statusText = document.getElementById('status-text');
    if (failCount === 0 && badge.className.includes('offline') && currentMode !== 'backtest') {
      // Let updateLiveView handle it
    }
  } catch (e) {
    failCount++;
    if (failCount > 3) {
      document.getElementById('status-badge').className = 'status-badge offline';
      document.getElementById('status-text').textContent = 'Disconnected';
    }
  }
}

// Initial + every 3 seconds
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


def run_dashboard(port=DASHBOARD_PORT):
    """Start the dashboard HTTP server."""
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  Dashboard running at http://localhost:{port}")
    print(f"  Reading from: {cfg.TRADE_LOG}, {cfg.STATE_FILE}")
    print(f"  Control file: {cfg.CONTROL_FILE}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Trader v15 — Live Dashboard")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="HTTP port (default: 8080)")
    args = parser.parse_args()

    # Change to the project directory so we find trades.json / state.json
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_dashboard(args.port)
