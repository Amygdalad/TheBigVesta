import re
import pandas as pd
import yfinance as yf
from datetime import datetime


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
            # parts[0] = 'YYYY-Mon-DD HH:MM'
            # parts[4] = DEC in sexagesimal string
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


def fetch_sp500_weekly(start: str = "1950-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    sp500 = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)
    if sp500.empty:
        raise RuntimeError("No S&P 500 data returned from yfinance.")
    # Align to weekly series ending Monday to match ephemeris weekly cadence
    weekly_close = sp500["Close"].resample("W-MON").last().dropna()
    out = weekly_close.reset_index()
    out.columns = ["Date", "Close_Price"]
    out["Date"] = out["Date"].dt.normalize()
    return out


if __name__ == "__main__":
    import os

    merged_path = "SP500_Vesta_Weekly_1950-2025.csv"
    spx_path = "SP500_Weekly_1950-2025.csv"

    if os.path.exists(merged_path) and os.path.exists(spx_path):
        print("Paired CSVs already exist; skipping download.")
    else:
        # Fetch S&P weekly (Mon end) and save
        spx_df = fetch_sp500_weekly()
        spx_df.to_csv(spx_path, index=False)

        # Parse Vesta weekly ephemeris and merge
        vesta_df = parse_vesta_ephemeris("vesta_ephemeris_weekly.txt")

        merged = pd.merge(spx_df, vesta_df, on="Date", how="inner")
        merged.to_csv(merged_path, index=False)

        print(f"SPX weekly rows: {len(spx_df)}")
        print(f"Vesta weekly rows: {len(vesta_df)}")
        print(f"Merged rows: {len(merged)}")
        print(merged.head())
