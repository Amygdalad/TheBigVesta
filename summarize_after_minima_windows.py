import argparse
import pandas as pd

import compute_sp500_returns_after_minima as mod


def main():
    ap = argparse.ArgumentParser(description='Summarize S&P returns after declination minima for fixed windows')
    ap.add_argument('--sp500', default='SP500_Weekly_1950-2025.csv', help='S&P500 weekly CSV path')
    ap.add_argument('--reversals', default='vesta_declination_reversals.csv', help='Declination reversals CSV path')
    ap.add_argument('--windows', default='1,3,6,12', help='Comma-separated window sizes in months')
    ap.add_argument('--minima-dates', default=None,
                    help='Override minima list with comma-separated YYYY-MM-DD dates')
    ap.add_argument('--output', default='after_minima_window_summary.csv', help='Output summary CSV path')
    ap.add_argument('--preview', action='store_true', help='Print the summary table')
    args = ap.parse_args()

    sp = mod.load_sp500_weekly(args.sp500)

    if args.minima_dates:
        dates = [d.strip() for d in args.minima_dates.split(',') if d.strip()]
        minima = pd.DataFrame({'datetime_utc': pd.to_datetime(dates)})
    else:
        minima = mod.load_declination_minima(args.reversals)[['datetime_utc']]

    windows = [int(x.strip()) for x in args.windows.split(',') if x.strip()]

    results = mod.compute_after_minima_returns(sp, minima, windows)
    if not results:
        raise SystemExit('No results computed.')

    df = pd.DataFrame([
        {
            'min_datetime_utc': r.min_datetime_utc,
            'window_months': r.window_months,
            'total_return': r.total_return,
            'percent_return': r.total_return * 100.0,
        }
        for r in results
    ])

    # Summary by window, include list of minima dates used
    grp = df.groupby('window_months', as_index=False)
    summary = grp['percent_return'].agg(
        n_minima='count',
        mean_return_pct='mean',
        median_return_pct='median',
        stdev_pct='std',
    )
    dates_per_window = (
        df.assign(date_str=df['min_datetime_utc'].dt.strftime('%Y-%m-%d'))
          .groupby('window_months')['date_str']
          .apply(lambda s: ';'.join(sorted(s.unique())))
          .reset_index(name='minima_dates')
    )
    summary = summary.merge(dates_per_window, on='window_months', how='left')

    # Round for readability (percent units)
    for col in ['mean_return_pct', 'median_return_pct', 'stdev_pct']:
        summary[col] = summary[col].round(1)

    # Column order for clarity
    summary = summary[
        ['window_months', 'n_minima', 'mean_return_pct', 'median_return_pct', 'stdev_pct', 'minima_dates']
    ].sort_values('window_months')

    summary.to_csv(args.output, index=False)

    # Per-minimum table (wide): one row per minima date
    pivot = df.pivot_table(index='min_datetime_utc', columns='window_months', values='percent_return', aggfunc='first').reset_index()
    new_cols = ['min_datetime_utc'] + [f'return_{int(c)}m_pct' for c in pivot.columns.tolist() if c != 'min_datetime_utc']
    pivot.columns = new_cols
    pivot['min_datetime_utc'] = pivot['min_datetime_utc'].dt.strftime('%Y-%m-%d')
    pivot.sort_values('min_datetime_utc', inplace=True)
    pivot.to_csv('after_minima_window_per_min.csv', index=False)

    if args.preview:
        print(summary.to_string(index=False))
        print(f'Saved summary -> {args.output}')
        print('Per-minimum table -> after_minima_window_per_min.csv')
    else:
        print(f'Saved summary -> {args.output}')


if __name__ == '__main__':
    main()
