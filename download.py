import concurrent.futures
import json
from argparse import ArgumentParser
from csv import DictWriter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import chain
from pathlib import Path
from time import sleep
from typing import List, Tuple, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

import pandas as pd
from dateutil.parser import parse as parse_date
import requests
from requests import ReadTimeout
from tqdm.notebook import tqdm

from args import add_date_arg
from constants import base_path, nation, region, ltla, standard_metrics, new_admissions, \
    vaccination_cumulative, vaccination_new_and_weekly, england_metrics, case_demographics, \
    overview, death_demographics, admission_demographics, nhs_region


def download(url, path):
    response = requests.get(url)
    assert response.status_code == 200
    with path.open('wb') as target:
        target.write(response.content)


def write_csv(rows: Iterable[Mapping[str, str]], filename: str):
    rows = iter(rows)
    row = next(rows)
    with (base_path / filename).open('w') as target:
        writer = DictWriter(target, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def find_all(glob, date_index=-1, earliest=None) -> List[Tuple[date, Path]]:
    possible = []
    for path in base_path.glob(glob):
        dt = parse_date(str(path.stem).rsplit('_')[date_index]).date()
        if earliest is None or dt >= earliest:
            possible.append((dt, path))
    return possible


def find_latest(glob, date_index=-1, on_or_before=None):
    possible = find_all(glob, date_index)
    for dt, path in sorted(possible, reverse=True):
        if on_or_before is not None and pd.to_datetime(dt) > on_or_before:
            continue
        return path, dt
    raise FileNotFoundError(glob)


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

    def __init__(self, status_code, content, retry_after):
        self.status_code = status_code
        self.content = content
        self.retry_after = retry_after

    def __str__(self):
        return f'{self.status_code}: {self.content}'


class NoContent(ValueError):
    pass


class WrongDate(ValueError):

    def __init__(self, requested, actual):
        self.requested, self.actual = requested, actual

    def __str__(self):
        return f'requested: {self.requested}, actual: {self.actual}'


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

    if response.status_code == 204:
        raise NoContent

    if response.status_code in (429, 403):
        raise RateLimited(
            response.status_code,
            response.content,
            int(response.headers['retry-after'])
        )

    if response.status_code != 200:
        raise ValueError(f'{response.status_code}:{response.content}')

    actual_release = datetime.strptime(
        response.headers['Content-Disposition'].rsplit('_')[-1], f'%Y-%m-%d.{format}"'
    ).date()
    if str(actual_release) != str(release):
        raise WrongDate(release, actual_release)
    path = (base_path / f'{name}_{actual_release}.csv')
    path.write_bytes(response.content)
    return path


def get_release_timestamp():
    response = requests.get('https://api.coronavirus.data.gov.uk/v1/timestamp')
    return parse_date(response.json()['websiteTimestamp'])


@dataclass
class Download:
    area_type: str
    metrics: List[str]
    area_name: str = None
    name: str = None


def main():
    parser = ArgumentParser()
    parser.add_argument('sets', choices=list(SETS), nargs='+')
    parser.add_argument('--name')
    add_date_arg(parser, '--start')
    add_date_arg(parser, '--end')
    add_date_arg(parser, '--date', default=date.today())
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.start and args.end:
        points = args.start, args.end
        dates = [dt.date() for dt in pd.date_range(min(*points), max(*points))]
    else:
        dates = [args.date]

    release_timestamp = get_release_timestamp()

    for dt in reversed(dates):

        if pd.to_datetime(dt, utc=True) > release_timestamp:
            print(f'{dt} not yet available, current: {release_timestamp}')
            continue

        for dl in chain(*(SETS[s] for s in args.sets)):
            name = dl.name or dl.area_name or dl.area_type
            if args.name and name != args.name:
                continue
            data_path = (base_path / f'{name}_{dt}.csv')
            if data_path.exists() and not args.overwrite:
                print('already exists:', data_path)
                continue
            try:
                while True:
                    try:
                        data_path = download_phe(name, dl.area_type, *dl.metrics,
                                                 area_name=dl.area_name, release=dt)
                    except RateLimited as e:
                        rldt = datetime.now()+timedelta(seconds=e.retry_after)
                        print(f'retrying after {e.retry_after}s at {rldt:%H:%M:%S} ({e})')
                        sleep(e.retry_after)
                    except ReadTimeout:
                        print('read timeout')
                    else:
                        break
            except NoContent:
                print(f'no content for {name} on {dt}')
            else:
                print('downloaded: ', data_path)


SETS = {
    'daily': [
        Download(nation, england_metrics, area_name='england'),
        Download(nation, vaccination_new_and_weekly, name='vaccination'),
        Download(nation, vaccination_cumulative, name='vaccination_cum'),
        Download(nation, [new_admissions]+standard_metrics),
    ]+[
        Download(area_type, standard_metrics) for area_type in (region, ltla)
    ],
    'demographics': [
        Download(overview, [case_demographics], name=f'case_demographics_{overview}'),
        Download(nation, [death_demographics], name=f'death_demographics_{nation}'),
        Download(nation, [admission_demographics], name=f'admission_demographics_{nation}'),
    ],
    'healthcare': [
        Download(nhs_region, [new_admissions]),
    ]
}


if __name__ == '__main__':
    main()
