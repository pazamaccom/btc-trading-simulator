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

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
            state = self._get_state()
            mode = state.get("mode", "live")
            base = {
                "state": state,
                "trades": self._get_trades(),
                "metrics": self._compute_metrics(),
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
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
                config_path = Path("config_update.json")
                config_path.write_text(json.dumps(data))
                self._serve_json({"ok": True, "updated": data})
            except Exception as e:
                self._serve_json({"ok": False, "error": str(e)})
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

    def _serve_json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
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
        closed = [t for t in trades if t.get("action") in ("SELL", "COVER")]
        entries = [t for t in trades if t.get("action") in ("BUY", "SHORT")]
        if not closed:
            return {
                "total_trades": 0, "cumulative_pnl": 0, "win_rate": 0,
                "avg_pnl": 0, "max_drawdown": 0, "best_trade": 0,
                "worst_trade": 0, "long_trades": 0, "long_pnl": 0,
                "long_wins": 0, "short_trades": 0, "short_pnl": 0,
                "short_wins": 0, "equity_curve": [], "profit_factor": 0,
                "avg_hold_hours": 0,
            }
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
            "avg_hold_hours": 0,
        }


# See full HTML dashboard below (embedded in DASHBOARD_HTML variable)
# The complete file with all HTML/CSS/JS is available in the workspace
# at /home/user/workspace/dashboard.py

# NOTE: This is a truncated push — the DASHBOARD_HTML variable containing
# the full single-page HTML application is too large to push via this method.
# The complete file (2620 lines, 86KB) must be pushed via git CLI or GitHub API.

DASHBOARD_HTML = ""


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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_dashboard(args.port)
