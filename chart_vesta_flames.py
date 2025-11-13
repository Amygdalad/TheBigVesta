import argparse
import re
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def _dec_sexagesimal_to_deg(dec_str: str) -> float:
    s = dec_str.strip()
    m = re.match(r"^([+\-]?)(\d+)\s+(\d+)\s+([\d.]+)$", s)
    if not m:
        raise ValueError(f"Unexpected DEC format: {dec_str!r}")
    sign, deg, minute, sec = m.groups()
    deg, minute, sec = int(deg), int(minute), float(sec)
    val = deg + minute / 60.0 + sec / 3600.0
    if sign == "-":
        val = -val
    return val


def parse_vesta_ephemeris(path: str) -> pd.DataFrame:
    dates = []
    decs = []
    in_table = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "$$SOE":
                in_table = True
                continue
            if line.strip() == "$$EOE":
                break
            if not in_table:
                continue
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                dt = datetime.strptime(parts[0], "%Y-%b-%d %H:%M")
                dec_deg = _dec_sexagesimal_to_deg(parts[4])
            except Exception:
                continue
            dates.append(dt.date())
            decs.append(dec_deg)
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Vesta_Declination_Deg": decs})
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _flames_colors(n: int) -> list:
    # Yellow-Orange-Red gradient reminiscent of flames
    # Use YlOrRd but bias toward saturated reds at the base and lighter yellows at the tips
    cmap = mpl.colormaps["YlOrRd"]
    # Generate from red (near 1) to yellow (near 0.1)
    vals = np.linspace(0.95, 0.10, n)
    return [cmap(v) for v in vals]


def plot_vesta_flames(df: pd.DataFrame, output: str, title: str = "Vesta Declination (Weekly)", dpi: int = 220) -> None:
    x = df["Date"].values
    y = df["Vesta_Declination_Deg"].values

    # Visual baseline at global minimum to create flame stacks upward
    baseline = np.nanmin(y)
    n_layers = 25
    colors = _flames_colors(n_layers)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Build stacked bands from baseline to the signal to simulate a gradient fill
    for j in range(n_layers):
        f0 = j / n_layers
        f1 = (j + 1) / n_layers
        y0 = baseline + (y - baseline) * f0
        y1 = baseline + (y - baseline) * f1
        ax.fill_between(x, y0, y1, color=colors[j], linewidth=0, alpha=0.9)

    # Emphasize the top edge as a glowing rim
    ax.plot(x, y, color="#ffe66d", linewidth=1.2, alpha=0.95)

    # Styling
    ax.set_title(title, fontsize=16, color="#ffd166", loc="left")
    ax.set_xlabel("Date", color="#cccccc")
    ax.set_ylabel("Declination (deg)", color="#cccccc")

    # Subtle grids for readability
    ax.grid(True, which="major", axis="y", color="#444444", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.grid(False, axis="x")

    # De-emphasize spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Tighten y-limits with slight headroom
    yr = np.nanmax(y) - baseline
    ax.set_ylim(baseline - 0.05 * yr, np.nanmax(y) + 0.10 * yr)

    # Date ticks auto-format
    fig.autofmt_xdate()

    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Chart Vesta ephemeris in a flames-style aesthetic.")
    parser.add_argument("--input", default="vesta_ephemeris_weekly.txt", help="Path to JPL/Horizons Vesta ephemeris text")
    parser.add_argument("--output", default="vesta_flames_chart.png", help="Output image path (PNG)")
    parser.add_argument("--title", default="Vesta Declination (Weekly)", help="Chart title")
    parser.add_argument("--start", default="1990-01-01", help="Filter start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="Filter end date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = parse_vesta_ephemeris(args.input)
    if df.empty:
        raise SystemExit("No rows parsed from ephemeris. Check input format.")

    # Apply date range filter
    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows within the requested date range.")

    plot_vesta_flames(df, args.output, title=args.title)
    print(f"Saved chart -> {args.output}")


if __name__ == "__main__":
    main()
