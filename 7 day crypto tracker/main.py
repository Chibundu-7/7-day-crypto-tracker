"""
Crypto 7‑Day Analyzer (CLI + PyQt5 GUI)
---------------------------------------
- Fetches last 7 days of price data for a coin from CoinGecko.
- Saves raw timestamped prices to CSV.
- Aggregates to daily averages.
- Computes max/min/mean and 7‑day % change.
- Plots both raw series and daily averages.
- Provides a simple PyQt5 GUI with live updating.

Run as CLI:
    python crypto_7day_analyzer.py bitcoin

Run GUI:
    python crypto_7day_analyzer.py

Notes:
- Coin IDs are CoinGecko IDs (e.g., 'bitcoin', 'ethereum', 'solana').
- Requires: requests, numpy, matplotlib, PyQt5
"""
from __future__ import annotations

import csv
import sys
import argparse
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
import numpy as np
from datetime import datetime

# Matplotlib (for CLI plot and embedding in PyQt5)
import matplotlib
matplotlib.use("Agg")  # Safe default; GUI will set interactive canvas
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

# Night mode style
plt.style.use("dark_background")

# --- PyQt5 imports guarded so CLI still works without GUI ---
try:
    from PyQt5 import QtCore, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


# ---------------------------------
# Data Structures
# ---------------------------------
@dataclass
class PricePoint:
    dt: datetime
    price: float


# ---------------------------------
# Core Logic
# ---------------------------------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=7"


def fetch_market_chart(coin: str) -> Dict:
    url = COINGECKO_URL.format(coin=coin)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "prices" not in data:
        raise ValueError("Unexpected API response: missing 'prices'")
    return data


def extract_prices(data: Dict) -> List[PricePoint]:
    prices: List[PricePoint] = []
    for ts_ms, price in data.get("prices", []):
        dt = datetime.fromtimestamp(ts_ms / 1000)
        prices.append(PricePoint(dt=dt, price=float(price)))
    if not prices:
        raise ValueError("No price data returned from API")
    return prices


def write_prices_csv(prices: List[PricePoint], filepath: str) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "price"])
        for p in prices:
            writer.writerow([p.dt.strftime('%Y-%m-%d %H:%M:%S'), f"{p.price:.8f}"])


def read_prices_csv(filepath: str) -> List[PricePoint]:
    out: List[PricePoint] = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            price = float(row["price"])
            out.append(PricePoint(dt, price))
    if not out:
        raise ValueError("CSV is empty or malformed")
    return out


def daily_averages(prices: List[PricePoint]) -> List[Tuple[datetime, float]]:
    buckets: Dict[str, List[float]] = {}
    for p in prices:
        dkey = p.dt.strftime('%Y-%m-%d')
        buckets.setdefault(dkey, []).append(p.price)
    result: List[Tuple[datetime, float]] = []
    for dkey in sorted(buckets.keys()):
        day_dt = datetime.strptime(dkey, '%Y-%m-%d')
        avg = float(np.mean(buckets[dkey]))
        result.append((day_dt, avg))
    return result


def compute_stats(daily_avg: List[Tuple[datetime, float]]) -> Dict[str, float]:
    values = np.array([v for _, v in daily_avg], dtype=float)
    max_v = float(np.max(values))
    min_v = float(np.min(values))
    mean_v = float(np.mean(values))
    if len(values) < 2 or values[0] == 0:
        pct_change = float('nan')
    else:
        pct_change = float((values[-1] - values[0]) / values[0] * 100.0)
    return {
        "max": max_v,
        "min": min_v,
        "mean": mean_v,
        "pct_change": pct_change,
    }


# ---------------------------------
# Plotting Helpers
# ---------------------------------

def make_plot(raw_points: List[PricePoint], daily_avg: List[Tuple[datetime, float]], title: str = "7‑Day Price"):
    fig, ax = plt.subplots(figsize=(9, 4))
    raw_dates = [p.dt for p in raw_points]
    raw_prices = [p.price for p in raw_points]
    ax.plot(raw_dates, raw_prices, linewidth=1.0, label="Raw")
    if daily_avg:
        d_dates = [d for d, _ in daily_avg]
        d_prices = [v for _, v in daily_avg]
        ax.plot(d_dates, d_prices, linewidth=2.0, marker="o", label="Daily Avg")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    locator = AutoDateLocator()
    formatter = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------
# PyQt5 GUI with Live Updating
# ---------------------------------
if QT_AVAILABLE:

    class MplCanvas(FigureCanvas):
        def __init__(self):
            self.fig, self.ax = plt.subplots(figsize=(9, 4))
            super().__init__(self.fig)

        def draw_plot(self, raw_points: List[PricePoint], daily_avg: List[Tuple[datetime, float]], title: str):
            self.ax.clear()
            raw_dates = [p.dt for p in raw_points]
            raw_prices = [p.price for p in raw_points]
            self.ax.plot(raw_dates, raw_prices, linewidth=1.0, label="Raw")
            if daily_avg:
                d_dates = [d for d, _ in daily_avg]
                d_prices = [v for _, v in daily_avg]
                self.ax.plot(d_dates, d_prices, linewidth=2.0, marker="o", label="Daily Avg")
            self.ax.set_title(title)
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Price (USD)")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            locator = AutoDateLocator()
            formatter = DateFormatter('%Y-%m-%d')
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(formatter)
            self.fig.autofmt_xdate()
            self.fig.tight_layout()
            self.draw()

    class Worker(QtCore.QObject):
        finished = QtCore.pyqtSignal(object, object, object)
        error = QtCore.pyqtSignal(str)

        @QtCore.pyqtSlot(str)
        def run(self, coin: str):
            try:
                data = fetch_market_chart(coin)
                prices = extract_prices(data)
                daily_avg = daily_averages(prices)
                stats = compute_stats(daily_avg)
                write_prices_csv(prices, "raw.csv")
                self.finished.emit(prices, daily_avg, stats)
            except Exception as e:
                msg = f"Error: {e}\n\n{traceback.format_exc()}"
                self.error.emit(msg)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Crypto 7‑Day Analyzer (Live)")
            self.resize(1000, 650)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            # Controls
            ctrl = QtWidgets.QHBoxLayout()
            self.coin_edit = QtWidgets.QLineEdit()
            self.coin_edit.setPlaceholderText("e.g., bitcoin, ethereum, solana")
            self.interval_spin = QtWidgets.QSpinBox()
            self.interval_spin.setRange(30, 3600)
            self.interval_spin.setValue(60)
            self.toggle_btn = QtWidgets.QPushButton("Start")
            self.status_lbl = QtWidgets.QLabel("")
            self.status_lbl.setStyleSheet("color: #aaa")

            ctrl.addWidget(QtWidgets.QLabel("Coin ID:"))
            ctrl.addWidget(self.coin_edit, 1)
            ctrl.addWidget(QtWidgets.QLabel("Interval (s):"))
            ctrl.addWidget(self.interval_spin)
            ctrl.addWidget(self.toggle_btn)
            ctrl.addWidget(self.status_lbl)
            layout.addLayout(ctrl)

            # Stats
            stats_box = QtWidgets.QGroupBox("Stats (Daily Averages)")
            stats_layout = QtWidgets.QFormLayout(stats_box)
            self.max_lbl = QtWidgets.QLabel("–")
            self.min_lbl = QtWidgets.QLabel("–")
            self.mean_lbl = QtWidgets.QLabel("–")
            self.pct_lbl = QtWidgets.QLabel("–")
            stats_layout.addRow("Max:", self.max_lbl)
            stats_layout.addRow("Min:", self.min_lbl)
            stats_layout.addRow("Mean:", self.mean_lbl)
            stats_layout.addRow("7‑Day % Change:", self.pct_lbl)
            layout.addWidget(stats_box)

            # Plot canvas
            self.canvas = MplCanvas()
            layout.addWidget(self.canvas, 1)

            # Worker thread
            self.thread = QtCore.QThread(self)
            self.worker = Worker()
            self.worker.moveToThread(self.thread)
            self.thread.start()

            # Timer for live updates
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_fetch)

            # Signals
            self.toggle_btn.clicked.connect(self.on_toggle)
            self.worker.finished.connect(self.on_finished)
            self.worker.error.connect(self.on_error)

        def on_toggle(self):
            if self.timer.isActive():
                self.timer.stop()
                self.toggle_btn.setText("Start")
                self.status_lbl.setText("Stopped")
            else:
                interval = self.interval_spin.value() * 1000
                self.timer.start(interval)
                self.on_fetch()  # immediate fetch
                self.toggle_btn.setText("Stop")
                self.status_lbl.setText("Running…")

        def on_fetch(self):
            coin = self.coin_edit.text().strip() or "bitcoin"
            QtCore.QMetaObject.invokeMethod(
                self.worker,
                "run",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, coin),
            )

        def on_finished(self, prices, daily_avg, stats):
            title = f"{self.coin_edit.text().strip() or 'bitcoin'} — Last 7 Days"
            self.canvas.draw_plot(prices, daily_avg, title)
            self.max_lbl.setText(f"${stats['max']:.2f}")
            self.min_lbl.setText(f"${stats['min']:.2f}")
            self.mean_lbl.setText(f"${stats['mean']:.2f}")
            pct = stats['pct_change']
            self.pct_lbl.setText(f"{pct:.2f}%" if np.isfinite(pct) else "n/a")

        def on_error(self, msg: str):
            self.status_lbl.setText("Error")
            QtWidgets.QMessageBox.critical(self, "Error", msg)


# ---------------------------------
# CLI Entry
# ---------------------------------

def cli_main(coin: str) -> int:
    try:
        data = fetch_market_chart(coin)
        prices = extract_prices(data)
        write_prices_csv(prices, "raw.csv")
        daily_avg = daily_averages(prices)
        stats = compute_stats(daily_avg)

        print("-----------")
        print(f"{coin} : Last 7 days")
        print(f"Highest Price: ${stats['max']:.2f}")
        print(f"Lowest Price:  ${stats['min']:.2f}")
        print(f"Average Price: ${stats['mean']:.2f}")
        print(f"7-Day % Change: {stats['pct_change']:.2f}%")
        print("-----------")

        fig = make_plot(prices, daily_avg, title=f"{coin} — Last 7 Days")
        fig.savefig("plot.png", dpi=150)
        print("Saved CSV: raw.csv\nSaved plot: plot.png")
        return 0
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Crypto 7‑Day Analyzer")
    parser.add_argument("coin", nargs="?", help="CoinGecko coin id (e.g., bitcoin)")
    args = parser.parse_args()

    if args.coin:
        sys.exit(cli_main(args.coin))
    else:
        if not QT_AVAILABLE:
            print("PyQt5 is not installed. Run 'pip install PyQt5' or pass a coin id for CLI mode.")
            sys.exit(2)
        app = QtWidgets.QApplication(sys.argv)
        matplotlib.use("Qt5Agg", force=True)
        win = MainWindow()
        win.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()