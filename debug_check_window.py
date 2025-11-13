from datetime import datetime, timedelta
import sys
import pandas as pd
import identify_vesta_declination_reversals as m


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_check_window.py YYYY-MM-DD [months=6]")
        sys.exit(1)
    dt = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    months = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

    df = m.read_horizons_weekly_ephemeris('vesta_ephemeris_weekly.txt')
    half_days = 0.5 * months * (365.2425 / 12.0)
    lo = dt - timedelta(days=half_days)
    hi = dt + timedelta(days=half_days)
    w = df[(df.dt >= lo) & (df.dt <= hi)].copy()
    print(f"window rows: {len(w)} {w.dt.min()} -> {w.dt.max()}")
    print(f"min dec in window: {w.dec_deg.min():.6f}")
    print(w[['dt', 'dec_deg']].to_string(index=False))


if __name__ == '__main__':
    main()

