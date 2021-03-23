import sys
from datetime import date

import pandas as pd
import requests
from dateutil.parser import parse as parse_date

from constants import msoa, new_cases_sum, new_cases_rate, new_cases_change
from download import download_phe, find_latest, get_release_timestamp, WrongDate
from msoa_composite import check_path, main as composite


def is_msoa_data_ready(dt):
    release_timestamp = get_release_timestamp()
    response = requests.head('https://coronavirus.data.gov.uk/downloads/maps/msoa_data_latest.geojson')
    msoa_timestamp = parse_date(response.headers['Last-Modified'])
    print(f'requested: {dt}, release: {release_timestamp}, msoa: {msoa_timestamp}')
    return dt <= release_timestamp and dt < msoa_timestamp


def main():
    _, latest = find_latest('msoa_????-*')

    start_for_composite = None

    for dt in pd.date_range(latest, date.today(), closed='right', tz='Europe/London'):
        if is_msoa_data_ready(dt):
            try:
                path = download_phe(
                    msoa, msoa,
                    new_cases_sum, new_cases_rate, new_cases_change, 'release',
                    release=dt.date()
                )
            except WrongDate as e:
                if e.requested < e.actual:
                    print(f'Missed {e.requested} :-(')
                else:
                    raise
            else:
                print(path)
                check_path(path)
                if start_for_composite is None:
                    start_for_composite = str(dt.date())
        else:
            print(f"MSOA NOT DOWNLOADED FOR {dt}!")
            sys.exit(1)

    if start_for_composite:
        print('\nAdding to composite...')
        composite(['--start', start_for_composite])


if __name__ == '__main__':
    main()
