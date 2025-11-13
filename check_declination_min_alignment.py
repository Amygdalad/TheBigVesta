import pandas as pd

EVENTS = {
    'dotcom_peak_spx': '2000-03-24',
    'housing_peak_spx': '2007-10-09',
    'gfc_panic_low_spx': '2008-11-20',
}


def main():
    df = pd.read_csv('vesta_declination_reversals.csv', parse_dates=['datetime_utc'])
    mins = df[df['reversal_type'] == 'declination_min'].copy()
    for name, date_str in EVENTS.items():
        t = pd.to_datetime(date_str)
        mins['diff_days'] = (mins['datetime_utc'] - t).dt.days
        i = mins['diff_days'].abs().idxmin()
        row = mins.loc[i]
        print(
            f"{name}: nearest min {row['datetime_utc'].strftime('%Y-%m-%d %H:%M')} "
            f"(diff {int(row['diff_days'])} days), decl={row['declination_deg']:.6f}"
        )


if __name__ == '__main__':
    main()

