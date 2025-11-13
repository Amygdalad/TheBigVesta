import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd


@dataclass
class Reversal:
    dt: datetime
    kind: str  # 'station_direct' or 'station_retrograde'
    ecl_lon_deg: float
    idx: int  # index near which reversal detected (center index of the 3-point window)


def parse_ra_to_deg(ra_str: str) -> float:
    """Parse RA string like '01 20 11.03' to degrees."""
    parts = ra_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected RA format: {ra_str}")
    h, m, s = parts
    h = float(h)
    m = float(m)
    s = float(s)
    return (h + m / 60.0 + s / 3600.0) * 15.0


def parse_dec_to_deg(dec_str: str) -> float:
    """Parse Dec string like '+00 34 36.8' to degrees."""
    parts = dec_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected Dec format: {dec_str}")
    sign = -1.0 if parts[0].startswith('-') else 1.0
    # Remove sign from degrees
    deg = float(parts[0].replace('+', '').replace('-', ''))
    arcmin = float(parts[1])
    arcsec = float(parts[2])
    return sign * (deg + arcmin / 60.0 + arcsec / 3600.0)


def equatorial_to_ecliptic_longitude_deg(ra_deg: float, dec_deg: float) -> float:
    """
    Convert ICRF/J2000 equatorial (RA, Dec) to ecliptic longitude (deg).
    Uses fixed mean obliquity for J2000: 23.43929111 degrees.
    """
    epsilon = math.radians(23.43929111)
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)

    # Using vector rotation approach
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)

    # Rotate about x-axis by +epsilon: equatorial -> ecliptic
    y_ecl = y * math.cos(epsilon) + z * math.sin(epsilon)
    x_ecl = x
    # z_ecl not needed for longitude

    lam = math.atan2(y_ecl, x_ecl)
    lam_deg = math.degrees(lam)
    if lam_deg < 0:
        lam_deg += 360.0
    return lam_deg


def read_horizons_weekly_ephemeris(filepath: str) -> pd.DataFrame:
    """
    Read Horizons CSV-like ephemeris exported with weekly step, bounded by $$SOE/$$EOE.
    Returns DataFrame with columns: dt (datetime), ra_deg, dec_deg, ecl_lon_deg.
    """
    rows: List[Tuple[datetime, float, float]] = []
    in_table = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == '$$SOE':
                in_table = True
                continue
            if line.strip() == '$$EOE':
                break
            if not in_table:
                continue
            if not line.strip():
                continue

            # Split CSV columns. Horizon inserts ' , , ' placeholders; split on comma.
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                # Skip malformed lines
                continue

            dt_str = parts[0]
            ra_str = parts[3]
            dec_str = parts[4]

            try:
                dt = datetime.strptime(dt_str, '%Y-%b-%d %H:%M')
            except ValueError:
                # Some exports might include extra spaces or different month case
                try:
                    dt = datetime.strptime(dt_str, '%Y-%b-%d %H:%M')
                except Exception:
                    # Skip if we cannot parse
                    continue

            try:
                ra_deg = parse_ra_to_deg(ra_str)
                dec_deg = parse_dec_to_deg(dec_str)
                ecl_lon_deg = equatorial_to_ecliptic_longitude_deg(ra_deg, dec_deg)
            except Exception:
                continue

            rows.append((dt, ra_deg, dec_deg, ecl_lon_deg))

    if not rows:
        raise RuntimeError('No ephemeris rows parsed from file.')

    df = pd.DataFrame(rows, columns=['dt', 'ra_deg', 'dec_deg', 'ecl_lon_deg'])

    # Unwrap ecliptic longitude to a continuous series for derivative testing
    # Convert to radians for unwrap, then back to degrees
    lam_rad = pd.Series([math.radians(v) for v in df['ecl_lon_deg'].to_list()])
    # Simple unwrap implementation to avoid numpy dependency
    unwrapped = []
    prev = None
    offset = 0.0
    for val in lam_rad:
        v = float(val)
        if prev is not None:
            dv = v + offset - prev
            if dv > math.pi:
                offset -= 2 * math.pi
            elif dv < -math.pi:
                offset += 2 * math.pi
        unwrapped.append(v + offset)
        prev = unwrapped[-1]

    df['ecl_lon_unwrapped_deg'] = [math.degrees(v) for v in unwrapped]
    return df


def find_reversals(df: pd.DataFrame) -> List[Reversal]:
    """
    Identify reversal points (stationary direct/retrograde) based on ecliptic longitude.
    Uses a 3-point parabolic fit to refine the date/time of the extremum.
    Assumes uniform weekly spacing; if spacing deviates, the refinement falls back to center date.
    """
    if df.shape[0] < 3:
        return []

    # Compute time deltas; expect constant weekly step
    dts = df['dt'].to_list()
    lam = df['ecl_lon_unwrapped_deg'].to_list()

    # Estimate typical spacing (median)
    spacings = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
    if not spacings:
        return []
    median_spacing_sec = sorted(spacings)[len(spacings)//2]
    h = timedelta(seconds=median_spacing_sec)

    reversals: List[Reversal] = []
    for i in range(1, len(lam) - 1):
        y0, y1, y2 = lam[i-1], lam[i], lam[i+1]
        d0 = y1 - y0
        d1 = y2 - y1

        # Detect sign change (local extremum)
        if d0 == 0 or d1 == 0:
            continue
        if d0 > 0 and d1 < 0:
            kind = 'station_retrograde'  # turning from direct (+) to retrograde (-)
        elif d0 < 0 and d1 > 0:
            kind = 'station_direct'  # turning from retrograde (-) to direct (+)
        else:
            continue

        # Parabolic time-of-extremum refinement
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-9:
            # Fallback: take center time
            t_ext = dts[i]
            y_tau = y1
        else:
            # t_offset from center (i) in seconds: t_offset = h * (y0 - y2) / (2*(y0 - 2*y1 + y2))
            t_offset = (h.total_seconds() * (y0 - y2)) / (2.0 * denom)
            # Clamp within one interval to avoid runaway due to noise
            t_offset = max(-h.total_seconds(), min(h.total_seconds(), t_offset))
            t_ext = dts[i] + timedelta(seconds=t_offset)

            # Ecliptic longitude at extremum via Lagrange form (normalized time tau)
            tau = t_offset / h.total_seconds()
            y_tau = (
                y0 * 0.5 * tau * (tau - 1.0)
                + y1 * (1.0 - tau * tau)
                + y2 * 0.5 * tau * (tau + 1.0)
            )

        reversals.append(
            Reversal(dt=t_ext, kind=kind, ecl_lon_deg=y_tau % 360.0, idx=i)
        )

    return reversals


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Identify Vesta reversal (station) dates from Horizons weekly ephemeris.'
    )
    parser.add_argument(
        '--input', '-i', default='vesta_ephemeris_weekly.txt', help='Path to Horizons ephemeris file.'
    )
    parser.add_argument(
        '--output', '-o', default='vesta_reversals.csv', help='Path to write CSV of reversals.'
    )
    parser.add_argument(
        '--preview', action='store_true', help='Print a preview of detected reversals.'
    )
    args = parser.parse_args()

    df = read_horizons_weekly_ephemeris(args.input)
    revs = find_reversals(df)

    if not revs:
        print('No reversals detected.')
        return

    out_df = pd.DataFrame(
        [
            {
                'datetime_utc': r.dt.strftime('%Y-%m-%d %H:%M'),
                'station_type': r.kind,
                'ecliptic_longitude_deg': round(r.ecl_lon_deg, 6),
                'row_index_center': r.idx,
            }
            for r in revs
        ]
    )
    out_df.to_csv(args.output, index=False)

    if args.preview:
        print(out_df.head(20).to_string(index=False))
        print(f"Total reversals detected: {len(out_df)}")
    else:
        print(f"Wrote {len(out_df)} reversals to {args.output}")


if __name__ == '__main__':
    main()

