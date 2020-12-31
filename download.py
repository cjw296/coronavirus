import concurrent.futures
import json
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from time import sleep
from urllib.parse import parse_qs, urlparse

import pandas as pd
from dateutil.parser import parse as parse_date
import requests
from requests import ReadTimeout
from tqdm.notebook import tqdm

from args import add_date_arg
from constants import base_path, nation, region, ltla, standard_metrics


def download(url, path):
    response = requests.get(url)
    assert response.status_code == 200
    with path.open('wb') as target:
        target.write(response.content)


def find_all(glob, date_index=-1, earliest=None):
    possible = []
    for path in base_path.glob(glob):
        dt = parse_date(str(path.stem).rsplit('_')[date_index]).date()
        if earliest is None or dt >= earliest:
            possible.append((dt, path))
    return possible


def find_latest(glob, date_index=-1):
    possible = find_all(glob, date_index)
    if not possible:
        raise FileNotFoundError(glob)
    dt, path = sorted(possible, reverse=True)[0]
    return path, dt


PHE_URL = 'https://api.coronavirus.data.gov.uk/v1/data'


def get_phe(filters, structure, **params):
    _params = {
        'filters': ';'.join(f'{k}={v}' for (k, v) in filters.items()),
        'structure': json.dumps({element:element for element in structure}),
    }
    _params.update(params)
    response = requests.get(PHE_URL, timeout=20, params=_params)
    if response.status_code != 200:
        raise ValueError(f'{response.status_code}:{response.content}')
    return response.json()


def query_phe(filters, structure, max_workers=None, **params):
    page = 1
    response = get_phe(filters, structure, page=page, **params)
    result = response['data']
    max_page = int(parse_qs(urlparse(response['pagination']['last']).query)['page'][0])
    if max_page > 1:
        t = tqdm(total=max_page)
        t.update(1)
        todo = range(2, max_page+1)
        attempt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or max_page-1) as executor:
            while todo:
                attempt += 1
                bad = []
                t.set_postfix({'errors': len(bad), 'attempt': attempt})
                futures = {executor.submit(get_phe, filters, structure, page=page, **params): page
                           for page in todo}
                for future in concurrent.futures.as_completed(futures):
                    page = futures[future]
                    try:
                        response = future.result()
                    except Exception as exc:
                        bad.append(page)
                        t.set_postfix({'errors': len(bad), 'attempt': attempt})
                    else:
                        result.extend(response['data'])
                        t.update(1)
                todo = bad
        t.close()
    return pd.DataFrame(result)


class RateLimited(ValueError):

    def __init__(self, retry_after):
        self.retry_after = retry_after


def download_phe(name, area_type, *metrics, area_name=None, release=None, format='csv'):
    release = release or date.today()

    _params = {
        'areaType': area_type,
        'metric': metrics,
        'format': format,
        'release': str(release),
    }
    if area_name:
        _params['areaName'] = area_name
    response = requests.get(
        'https://api.coronavirus.data.gov.uk/v2/data', timeout=20, params=_params
    )

    if response.status_code in (429, 403):
        raise RateLimited(int(response.headers['retry-after']))

    if response.status_code != 200:
        raise ValueError(f'{response.status_code}:{response.content}')

    actual_release = datetime.strptime(
        response.headers['Content-Disposition'].rsplit('_')[-1], f'%Y-%m-%d.{format}"'
    ).date()
    if str(actual_release) != str(release):
        raise ValueError(f'downloaded: {actual_release}, requested: {release}')
    path = (base_path / f'{name}_{actual_release}.csv')
    path.write_bytes(response.content)
    return path


def main():
    parser = ArgumentParser()
    add_date_arg(parser, '--start')
    add_date_arg(parser, '--end')
    add_date_arg(parser, '--date')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not (args.date or (args.start and args.end)):
        parser.error('--date or --start and --end must be supplied')

    if args.date:
        dates = [args.date]
    else:
        points = args.start, args.end
        dates = [dt.date() for dt in pd.date_range(min(*points), max(*points))]

    print('calls to make: ', len(dates)*3)

    for d in reversed(dates):
        for area_type in nation, region, ltla:
            data_path = (base_path / f'{area_type}_{d}.csv')
            if data_path.exists() and not args.overwrite:
                print('already exists:', data_path)
                continue
            while True:
                try:
                    data_path = download_phe(area_type, area_type, *standard_metrics, release=d)
                except RateLimited as e:
                    dt = datetime.now()+timedelta(seconds=e.retry_after)
                    print(f'retrying after {e.retry_after}s at {dt:%H:%M:%S}')
                    sleep(e.retry_after)
                except ReadTimeout:
                    print('read timeout')
                else:
                    break
            print('downloaded: ', data_path)


if __name__ == '__main__':
    main()
