from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class IntervalReturn:
    start_dt: datetime
    end_dt: datetime
    phase: str  # 'max_to_min' or 'min_to_max'
    sp_start_date: datetime
    sp_end_date: datetime
    weeks: int
    start_price: float
    end_price: float
    total_return: float  # (end/start - 1)


def load_sp500_weekly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def load_declination_reversals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    # Expect columns: datetime_utc, reversal_type in {'declination_max','declination_min'}
    return df.sort_values('datetime_utc').reset_index(drop=True)


def build_intervals(rev_df: pd.DataFrame) -> List[Tuple[datetime, datetime, str]]:
    rev_df = rev_df.sort_values('datetime_utc').reset_index(drop=True)
    rows = rev_df[['datetime_utc', 'reversal_type']].to_records(index=False)
    out: List[Tuple[datetime, datetime, str]] = []
    for i in range(len(rows) - 1):
        t0, k0 = rows[i]
        t1, k1 = rows[i + 1]
        if k0 == 'declination_max' and k1 == 'declination_min':
            phase = 'max_to_min'
        elif k0 == 'declination_min' and k1 == 'declination_max':
            phase = 'min_to_max'
        else:
            continue
        out.append((pd.Timestamp(t0).to_pydatetime(), pd.Timestamp(t1).to_pydatetime(), phase))
    return out


def align_to_sp500(sp: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> Optional[Tuple[int, int]]:
    # Find first S&P row with Date >= start_dt and last with Date <= end_dt
    start_idx = int(pd.Index(sp['Date']).searchsorted(pd.Timestamp(start_dt), side='left'))
    if start_idx >= len(sp):
        return None
    end_idx = int(pd.Index(sp['Date']).searchsorted(pd.Timestamp(end_dt), side='right')) - 1
    if end_idx <= start_idx:
        return None
    return start_idx, end_idx


def compute_interval_returns(sp: pd.DataFrame, intervals: List[Tuple[datetime, datetime, str]]) -> List[IntervalReturn]:
    out: List[IntervalReturn] = []
    for (t0, t1, phase) in intervals:
        idxs = align_to_sp500(sp, t0, t1)
        if idxs is None:
            continue
        s_idx, e_idx = idxs
        sdt = sp.iloc[s_idx]['Date'].to_pydatetime()
        edt = sp.iloc[e_idx]['Date'].to_pydatetime()
        sprice = float(sp.iloc[s_idx]['Close_Price'])
        eprice = float(sp.iloc[e_idx]['Close_Price'])
        weeks = int(e_idx - s_idx)
        if sprice <= 0:
            continue
        total_return = eprice / sprice - 1.0
        out.append(
            IntervalReturn(
                start_dt=t0,
                end_dt=t1,
                phase=phase,
                sp_start_date=sdt,
                sp_end_date=edt,
                weeks=weeks,
                start_price=sprice,
                end_price=eprice,
                total_return=total_return,
            )
        )
    return out


def summarize(intervals: List[IntervalReturn]):
    def stats(vals: List[float]):
        if not vals:
            return {
                'count': 0,
                'mean': float('nan'),
                'median': float('nan'),
                'stdev': float('nan'),
            }
        s = pd.Series(vals)
        return {
            'count': len(vals),
            'mean': s.mean(),
            'median': s.median(),
            'stdev': s.std(ddof=1) if len(vals) > 1 else 0.0,
        }

    max_to_min = [ir.total_return for ir in intervals if ir.phase == 'max_to_min']
    min_to_max = [ir.total_return for ir in intervals if ir.phase == 'min_to_max']
    return {
        'max_to_min': stats(max_to_min),
        'min_to_max': stats(min_to_max),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute S&P500 returns between declination phases (DEC ICRF).')
    parser.add_argument('--sp500', default='SP500_Weekly_1950-2025.csv', help='S&P500 weekly CSV path')
    parser.add_argument('--reversals', default='vesta_declination_reversals.csv', help='Declination reversals CSV path')
    parser.add_argument('--output-intervals', default='sp500_returns_by_declination_intervals.csv', help='Per-interval output CSV')
    parser.add_argument('--preview', action='store_true', help='Print summary to stdout')
    args = parser.parse_args()

    sp = load_sp500_weekly(args.sp500)
    rev = load_declination_reversals(args.reversals)

    # Restrict to S&P500 data range
    rev = rev[rev['datetime_utc'] >= sp['Date'].min()]
    intervals = build_intervals(rev)
    interval_returns = compute_interval_returns(sp, intervals)

    # Write per-interval CSV
    out_df = pd.DataFrame([
        {
            'phase': ir.phase,
            'interval_start_utc': ir.start_dt.strftime('%Y-%m-%d %H:%M'),
            'interval_end_utc': ir.end_dt.strftime('%Y-%m-%d %H:%M'),
            'sp_start_date': ir.sp_start_date.strftime('%Y-%m-%d'),
            'sp_end_date': ir.sp_end_date.strftime('%Y-%m-%d'),
            'weeks': ir.weeks,
            'start_price': round(ir.start_price, 6),
            'end_price': round(ir.end_price, 6),
            'total_return': round(ir.total_return, 6),
        }
        for ir in interval_returns
    ])
    out_df.to_csv(args.output_intervals, index=False)

    summary = summarize(interval_returns)
    if args.preview:
        print('Summary of S&P500 total returns per declination phase (end/start - 1):')
        for phase in ['max_to_min', 'min_to_max']:
            s = summary[phase]
            print(f"{phase}: count={s['count']} mean={s['mean']:.4%} median={s['median']:.4%} stdev={s['stdev']:.4%}")
        print(f"Wrote per-interval details to {args.output_intervals}")
    else:
        print(f"Wrote per-interval details to {args.output_intervals}")


if __name__ == '__main__':
    main()

