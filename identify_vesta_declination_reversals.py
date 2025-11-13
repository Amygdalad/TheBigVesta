import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd


@dataclass
class DeclinationReversal:
    dt: datetime
    kind: str  # 'declination_max' or 'declination_min'
    declination_deg: float
    idx: int  # index of center sample in 3-point window


def parse_ra_to_deg(ra_str: str) -> float:
    parts = ra_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected RA format: {ra_str}")
    h, m, s = parts
    return (float(h) + float(m) / 60.0 + float(s) / 3600.0) * 15.0


def parse_dec_to_deg(dec_str: str) -> float:
    parts = dec_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected Dec format: {dec_str}")
    sign = -1.0 if parts[0].startswith('-') else 1.0
    deg = float(parts[0].replace('+', '').replace('-', ''))
    arcmin = float(parts[1])
    arcsec = float(parts[2])
    return sign * (deg + arcmin / 60.0 + arcsec / 3600.0)


def read_horizons_weekly_ephemeris(filepath: str) -> pd.DataFrame:
    """Read Horizons weekly ephemeris bounded by $$SOE/$$EOE and return dt, dec_deg."""
    rows: List[Tuple[datetime, float]] = []
    in_table = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == '$$SOE':
                in_table = True
                continue
            if line.strip() == '$$EOE':
                break
            if not in_table or not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue

            dt_str = parts[0]
            # parts[3] is RA, parts[4] is Dec; RA parsing kept for validation if needed
            dec_str = parts[4]

            try:
                dt = datetime.strptime(dt_str, '%Y-%b-%d %H:%M')
            except Exception:
                continue

            try:
                dec_deg = parse_dec_to_deg(dec_str)
            except Exception:
                continue

            rows.append((dt, dec_deg))

    if not rows:
        raise RuntimeError('No ephemeris rows parsed from file.')

    return pd.DataFrame(rows, columns=['dt', 'dec_deg'])


def find_declination_reversals(df: pd.DataFrame) -> List[DeclinationReversal]:
    if df.shape[0] < 3:
        return []

    dts = df['dt'].to_list()
    decs = df['dec_deg'].to_list()

    spacings = [(dts[i + 1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
    if not spacings:
        return []
    median_spacing_sec = sorted(spacings)[len(spacings) // 2]
    h = timedelta(seconds=median_spacing_sec)

    out: List[DeclinationReversal] = []
    for i in range(1, len(decs) - 1):
        y0, y1, y2 = decs[i - 1], decs[i], decs[i + 1]
        d0 = y1 - y0
        d1 = y2 - y1

        if d0 == 0 or d1 == 0:
            continue

        if d0 > 0 and d1 < 0:
            kind = 'declination_max'
        elif d0 < 0 and d1 > 0:
            kind = 'declination_min'
        else:
            continue

        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-12:
            t_ext = dts[i]
            y_tau = y1
        else:
            t_offset = (h.total_seconds() * (y0 - y2)) / (2.0 * denom)
            t_offset = max(-h.total_seconds(), min(h.total_seconds(), t_offset))
            t_ext = dts[i] + timedelta(seconds=t_offset)
            tau = t_offset / h.total_seconds()
            y_tau = (
                y0 * 0.5 * tau * (tau - 1.0)
                + y1 * (1.0 - tau * tau)
                + y2 * 0.5 * tau * (tau + 1.0)
            )

        out.append(DeclinationReversal(dt=t_ext, kind=kind, declination_deg=y_tau, idx=i))

    return out


def filter_minima_by_window(
    revs: List[DeclinationReversal],
    df: pd.DataFrame,
    months: float = 6.0,
    tolerance_deg: float = 0.05,
    mode: str = 'centered',  # 'centered', 'trailing', 'forward'
) -> List[DeclinationReversal]:
    """
    Keep only those declination minima that are the minimum within a centered
    rolling window of `months`. Maxima are not filtered.

    - months: window size in months (uses mean month length of 365.2425/12 days)
    - tolerance_deg: allow candidate within this many degrees of the discrete
      minimum in the window to account for refinement vs. sample discretization.
    """
    if not revs:
        return revs

    days_per_month = 365.2425 / 12.0
    half_window = timedelta(days=0.5 * months * days_per_month)
    full_window = timedelta(days=months * days_per_month)

    dts = df['dt'].to_list()
    decs = df['dec_deg'].to_list()

    filtered: List[DeclinationReversal] = []
    for r in revs:
        if r.kind != 'declination_min':
            filtered.append(r)
            continue
        if mode == 'trailing':
            t0 = r.dt - full_window
            t1 = r.dt
        elif mode == 'forward':
            t0 = r.dt
            t1 = r.dt + full_window
        else:
            t0 = r.dt - half_window
            t1 = r.dt + half_window
        in_idx = [i for i, d in enumerate(dts) if t0 <= d <= t1]
        if not in_idx:
            continue
        window_min = min(decs[i] for i in in_idx)
        if r.declination_deg <= window_min + tolerance_deg:
            filtered.append(r)

    return filtered


def filter_maxima_by_window(
    revs: List[DeclinationReversal],
    df: pd.DataFrame,
    months: float = 0.0,
    tolerance_deg: float = 0.05,
    mode: str = 'centered',  # 'centered', 'trailing', 'forward'
) -> List[DeclinationReversal]:
    """
    Keep only declination maxima that are the highest within a specified window.
    """
    if not revs:
        return revs

    days_per_month = 365.2425 / 12.0
    half_window = timedelta(days=0.5 * months * days_per_month)
    full_window = timedelta(days=months * days_per_month)

    dts = df['dt'].to_list()
    decs = df['dec_deg'].to_list()

    filtered: List[DeclinationReversal] = []
    for r in revs:
        if r.kind != 'declination_max':
            filtered.append(r)
            continue

        if mode == 'trailing':
            t0 = r.dt - full_window
            t1 = r.dt
        elif mode == 'forward':
            t0 = r.dt
            t1 = r.dt + full_window
        else:
            t0 = r.dt - half_window
            t1 = r.dt + half_window

        in_idx = [i for i, d in enumerate(dts) if t0 <= d <= t1]
        if not in_idx:
            continue
        window_max = max(decs[i] for i in in_idx)
        if r.declination_deg >= window_max - tolerance_deg:
            filtered.append(r)

    return filtered


def filter_minima_by_amplitude(
    revs: List[DeclinationReversal],
    df: pd.DataFrame,
    months_each_side: float = 3.0,
    amplitude_threshold_deg: float = 0.0,
) -> List[DeclinationReversal]:
    """
    Keep only minima that have sufficient dip depth relative to nearby maxima
    on both sides within months_each_side window. A minimum passes if
    min(left_max - min_val, right_max - min_val) >= amplitude_threshold_deg.
    """
    if amplitude_threshold_deg <= 0.0 or not revs:
        return revs

    days_per_month = 365.2425 / 12.0
    half = pd.Timedelta(days=months_each_side * days_per_month)

    dts = df['dt'].to_list()
    decs = df['dec_deg'].to_list()

    filtered: List[DeclinationReversal] = []
    for r in revs:
        if r.kind != 'declination_min':
            filtered.append(r)
            continue

        t = pd.Timestamp(r.dt)
        left_idx = [i for i, d in enumerate(dts) if (t - half) <= d <= t]
        right_idx = [i for i, d in enumerate(dts) if t <= d <= (t + half)]
        if not left_idx or not right_idx:
            # If we cannot establish context on both sides, drop conservatively
            continue

        left_max = max(decs[i] for i in left_idx)
        right_max = max(decs[i] for i in right_idx)
        depth_left = left_max - r.declination_deg
        depth_right = right_max - r.declination_deg
        depth = min(depth_left, depth_right)

        if depth >= amplitude_threshold_deg:
            filtered.append(r)

    return filtered


def enforce_min_separation(
    revs: List[DeclinationReversal],
    min_days: int = 365,
) -> List[DeclinationReversal]:
    """
    Ensure declination minima are separated by at least `min_days`.
    If multiple minima violate separation, keep the deepest (lowest declination)
    and drop the others within the exclusion window.
    """
    if min_days <= 0:
        return revs

    mins = [r for r in revs if r.kind == 'declination_min']
    if len(mins) <= 1:
        return revs

    # Sort minima by depth (ascending declination: more negative is deeper)
    order = sorted(range(len(mins)), key=lambda i: mins[i].declination_deg)
    selected_idx = set()
    suppressed = [False] * len(mins)

    for i in order:
        if suppressed[i]:
            continue
        selected_idx.add(i)
        ti = mins[i].dt
        # Suppress any minima within min_days of this one
        for j in range(len(mins)):
            if j == i or suppressed[j]:
                continue
            tj = mins[j].dt
            if abs((tj - ti).days) < min_days:
                suppressed[j] = True

    selected_mins = [mins[i] for i in sorted(selected_idx, key=lambda i: mins[i].dt)]

    # Merge back, preserving chronological order
    selected_set = set(id(m) for m in selected_mins)
    out: List[DeclinationReversal] = []
    for r in sorted(revs, key=lambda r: r.dt):
        if r.kind == 'declination_min':
            if id(r) in selected_set:
                out.append(r)
        else:
            out.append(r)
    return out


def enforce_max_separation(
    revs: List[DeclinationReversal],
    min_days: int = 365,
) -> List[DeclinationReversal]:
    """
    Ensure declination maxima are separated by at least `min_days`.
    If multiple maxima violate separation, keep the highest (greatest declination)
    and drop the others within the exclusion window.
    """
    if min_days <= 0:
        return revs

    maxs = [r for r in revs if r.kind == 'declination_max']
    if len(maxs) <= 1:
        return revs

    order = sorted(range(len(maxs)), key=lambda i: maxs[i].declination_deg, reverse=True)
    selected_idx = set()
    suppressed = [False] * len(maxs)

    for i in order:
        if suppressed[i]:
            continue
        selected_idx.add(i)
        ti = maxs[i].dt
        for j in range(len(maxs)):
            if j == i or suppressed[j]:
                continue
            tj = maxs[j].dt
            if abs((tj - ti).days) < min_days:
                suppressed[j] = True

    selected_maxs = [maxs[i] for i in sorted(selected_idx, key=lambda i: maxs[i].dt)]

    selected_max_set = set(id(m) for m in selected_maxs)
    out: List[DeclinationReversal] = []
    for r in sorted(revs, key=lambda r: r.dt):
        if r.kind == 'declination_max':
            if id(r) in selected_max_set:
                out.append(r)
        else:
            out.append(r)
    return out


def enforce_cross_separation(
    revs: List[DeclinationReversal],
    min_days: int = 365,
) -> List[DeclinationReversal]:
    """
    Ensure no declination minimum and maximum occur within `min_days` of each other.
    Greedy selection keeps the more extreme event by absolute declination |dec|
    and suppresses other events within the exclusion window.
    """
    if min_days <= 0 or len(revs) <= 1:
        return revs

    events = list(revs)
    order = sorted(range(len(events)), key=lambda i: abs(events[i].declination_deg), reverse=True)
    suppressed = [False] * len(events)
    keep_idx = []

    for i in order:
        if suppressed[i]:
            continue
        keep_idx.append(i)
        ti = events[i].dt
        for j in range(len(events)):
            if j == i or suppressed[j]:
                continue
            tj = events[j].dt
            if abs((tj - ti).days) < min_days:
                suppressed[j] = True

    kept = [events[i] for i in sorted(keep_idx, key=lambda k: events[k].dt)]
    return kept


def enforce_alternation(revs: List[DeclinationReversal]) -> List[DeclinationReversal]:
    """
    Collapse any consecutive same-type extrema runs to a single, most-extreme event.
    - For 'declination_max' runs, keep the highest declination value.
    - For 'declination_min' runs, keep the lowest declination value.
    Assumes `revs` are already sorted by dt and validated by window/separation rules.
    """
    if not revs:
        return revs

    revs_sorted = sorted(revs, key=lambda r: r.dt)
    out: List[DeclinationReversal] = []

    i = 0
    n = len(revs_sorted)
    while i < n:
        j = i
        kind = revs_sorted[i].kind
        # accumulate run [i..j) where kind stays the same
        best = revs_sorted[i]
        while j < n and revs_sorted[j].kind == kind:
            rj = revs_sorted[j]
            if kind == 'declination_max':
                if rj.declination_deg > best.declination_deg:
                    best = rj
            else:  # declination_min
                if rj.declination_deg < best.declination_deg:
                    best = rj
            j += 1
        out.append(best)
        i = j

    # Ensure strict alternation by design: we collapsed each same-type run to one
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Identify Vesta declination reversal dates (max/min).')
    parser.add_argument('--input', '-i', default='vesta_ephemeris_weekly.txt', help='Horizons ephemeris file path')
    parser.add_argument('--output', '-o', default='vesta_declination_reversals.csv', help='Output CSV path')
    parser.add_argument('--preview', action='store_true', help='Print preview to stdout')
    parser.add_argument('--min-window-months', type=float, default=6.0,
                        help='Window in months for validating minima (default: 6).')
    parser.add_argument('--min-tolerance-deg', type=float, default=0.05,
                        help='Tolerance in degrees comparing candidate min vs window min (default: 0.05).')
    parser.add_argument('--max-window-months', type=float, default=0.0,
                        help='Window in months for validating maxima (default: 0, disabled).')
    parser.add_argument('--max-tolerance-deg', type=float, default=0.05,
                        help='Tolerance in degrees comparing candidate max vs window max (default: 0.05).')
    parser.add_argument('--window-mode', choices=['centered','trailing','forward'], default='centered',
                        help='Window mode for window validations (default: centered).')
    parser.add_argument('--min-amplitude-deg', type=float, default=0.0,
                        help='Require minima dip depth vs nearby maxima on both sides (default: 0, disabled).')
    parser.add_argument('--min-separation-days', type=int, default=365,
                        help='Minimum separation between declination minima in days (default: 365).')
    parser.add_argument('--max-separation-days', type=int, default=365,
                        help='Minimum separation between declination maxima in days (default: 365).')
    parser.add_argument('--cross-separation-days', type=int, default=365,
                        help='Minimum separation between any min and any max (default: 365).')
    args = parser.parse_args()

    df = read_horizons_weekly_ephemeris(args.input)
    revs = find_declination_reversals(df)
    revs = filter_minima_by_window(
        revs, df,
        months=args.min_window_months,
        tolerance_deg=args.min_tolerance_deg,
        mode=args.window_mode,
    )
    if args.max_window_months and args.max_window_months > 0:
        revs = filter_maxima_by_window(
            revs, df,
            months=args.max_window_months,
            tolerance_deg=args.max_tolerance_deg,
            mode=args.window_mode,
        )
    revs = filter_minima_by_amplitude(
        revs, df,
        months_each_side=max(0.5 * args.min_window_months, 3.0),
        amplitude_threshold_deg=args.min_amplitude_deg,
    )
    # Enforce spacing with extremeness selection
    revs = enforce_min_separation(revs, min_days=args.min_separation_days)
    revs = enforce_max_separation(revs, min_days=args.max_separation_days)
    revs = enforce_cross_separation(revs, min_days=args.cross_separation_days)
    revs = enforce_alternation(revs)

    if not revs:
        print('No declination reversals detected.')
        return

    out_df = pd.DataFrame(
        [
            {
                'datetime_utc': r.dt.strftime('%Y-%m-%d %H:%M'),
                'reversal_type': r.kind,
                'declination_deg': round(r.declination_deg, 6),
                'row_index_center': r.idx,
            }
            for r in revs
        ]
    )

    out_df.to_csv(args.output, index=False)

    if args.preview:
        print(out_df.head(20).to_string(index=False))
        print(f"Total declination reversals detected: {len(out_df)}")
    else:
        print(f"Wrote {len(out_df)} declination reversals to {args.output}")


if __name__ == '__main__':
    main()
