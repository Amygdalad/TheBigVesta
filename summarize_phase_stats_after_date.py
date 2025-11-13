from __future__ import annotations

import sys
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python summarize_phase_stats_after_date.py <intervals_csv> <YYYY-MM-DD>")
        sys.exit(1)
    path = sys.argv[1]
    cutoff = pd.to_datetime(sys.argv[2])

    df = pd.read_csv(path)
    df['sp_start_date'] = pd.to_datetime(df['sp_start_date'])
    df = df[df['sp_start_date'] >= cutoff]
    if df.empty:
        print("No intervals on/after cutoff date.")
        return

    def stats(group):
        s = group['total_return']
        return pd.Series({
            'count': s.count(),
            'mean': s.mean(),
            'median': s.median(),
            'stdev': s.std(ddof=1) if s.count() > 1 else 0.0,
        })

    out = df.groupby('phase', as_index=False).apply(stats).reset_index(drop=True)
    # Print nicely
    for _, row in out.iterrows():
        phase = row['phase']
        print(f"{phase}: count={int(row['count'])} mean={row['mean']:.4%} median={row['median']:.4%} stdev={row['stdev']:.4%}")


if __name__ == '__main__':
    main()

