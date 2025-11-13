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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Identify Vesta declination reversal dates (max/min).')
    parser.add_argument('--input', '-i', default='vesta_ephemeris_weekly.txt', help='Horizons ephemeris file path')
    parser.add_argument('--output', '-o', default='vesta_declination_reversals.csv', help='Output CSV path')
    parser.add_argument('--preview', action='store_true', help='Print preview to stdout')
    args = parser.parse_args()

    df = read_horizons_weekly_ephemeris(args.input)
    revs = find_declination_reversals(df)

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

