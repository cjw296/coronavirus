import concurrent.futures
import json
import logging
import sys
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictWriter, DictReader
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from itertools import chain
from pathlib import Path
from time import sleep
from typing import List, Tuple, Iterable, Mapping, Optional
from urllib.parse import parse_qs, urlparse, urlencode, quote

import pandas as pd
import requests
from dateutil.parser import parse as parse_date
from requests import ReadTimeout
from tqdm.notebook import tqdm

from args import add_date_arg
from constants import base_path, nation, region, ltla, standard_metrics, new_admissions, \
    vaccination_publish_date_metrics, england_metrics, case_demographics, \
    death_demographics, admission_demographics, nhs_region, new_virus_tests, \
    utla, all_nation_metrics, case_demographics_male, \
    case_demographics_female, area_code_lookup, reported_virus_tests_sum, in_hospital, in_mv_beds, \
    nhs_trust, new_cases_by_specimen_date, new_deaths_by_death_date

MAX_METRICS = 5

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


def find_latest(glob, date_index=-1, on_or_before=None) -> Tuple[Path, date]:
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


class HttpError(ValueError):

    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content

    def __str__(self):
        return f'{self.status_code}: {self.content}'


class RateLimited(ValueError):

    def __init__(self, status_code, content, retry_after):
        super().__init__(status_code, content)
        self.retry_after = retry_after


class NoContent(ValueError):
    pass


class WrongDate(ValueError):

    def __init__(self, requested, actual):
        self.requested, self.actual = requested, actual

    def __str__(self):
        return f'requested: {self.requested}, actual: {self.actual}'


def download_phe_batch(name, area_type, release: date, area_name: Optional[str], *metrics):

    _params = {
        'areaType': area_type,
        'metric': metrics,
        'format': 'csv',
        'release': str(release),
    }
    if area_type == 'msoa':
        del _params['release']
        # support appears to have been dropped :-/
    if area_name:
        area_code = area_code_lookup[area_name]
        if area_code:
            _params['areaCode'] = area_code
        else:
            _params['areaName'] = area_name
    response = requests.get(
        'https://api.coronavirus.data.gov.uk/v2/data', timeout=20,
        params=urlencode(_params, quote_via=quote, doseq=True)
    )

    if response.status_code == 204 or response.content == b'':
        raise NoContent

    if response.status_code in (429, 403):
        raise RateLimited(
            response.status_code,
            response.content,
            int(response.headers['retry-after'])
        )

    if response.status_code != 200:
        raise HttpError(response.status_code, response.content)

    actual_release = datetime.strptime(
        response.headers['Content-Disposition'].rsplit('_')[-1], f'%Y-%m-%d.csv"'
    ).date()
    if str(actual_release) != str(release):
        raise WrongDate(release, actual_release)
    return response.text


def download_phe(name, area_type, *metrics, area_name: str = None, release: date = None):
    area_name = ' '.join(part.capitalize() for part in area_name.split()) if area_name else None
    release = release or date.today()
    if len(metrics) <= MAX_METRICS:
        content = download_phe_batch(name, area_type, release, area_name, *metrics)
    else:
        metric_sets = [metrics[i:i + MAX_METRICS] for i in range(0, len(metrics), MAX_METRICS)]
        rows = defaultdict(dict)
        columns = set()

        for metrics in metric_sets:
            content = download_phe_batch(name, area_type, release, area_name, *metrics)
            if not content:
                raise NoContent(name, area_type, release, area_name, metrics)
            reader = DictReader(StringIO(content))
            for row in reader:
                rows[row['areaCode'], row['date']].update(row)
            columns.update(reader.fieldnames)

        output = StringIO()
        csv_cols = ['areaCode', 'areaName', 'areaType', 'date']
        for col in csv_cols:
            columns.remove(col)
        csv_cols.extend(sorted(columns))
        writer = DictWriter(output, csv_cols)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
        content = output.getvalue()

    path = (base_path / f'{name}_{release}.csv')
    path.write_text(content)

    return path


def retrying_phe_download(
        name, area_type, *metrics, area_name: str = None, release: date = None,
):
    attempt = 1
    while attempt < 10_000:
        if attempt > 1:
            print(f'attempt {attempt} for {name} on {release}')
        try:
            return download_phe(name, area_type, *metrics,
                                area_name=area_name, release=release)
        except RateLimited as e:
            rldt = datetime.now() + timedelta(seconds=e.retry_after)
            print(f'retrying after {e.retry_after}s at {rldt:%H:%M:%S} ({e})')
            sleep(e.retry_after)
        except HttpError as e:
            if 400 <= e.status_code < 500:
               raise
            print(f'retrying as {e}')
        except ReadTimeout:
            print('read timeout')
        attempt += 1


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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.start and args.end:
        points = args.start, args.end
        dates = [dt.date() for dt in pd.date_range(min(*points), max(*points))]
    elif args.start or args.end:
        parser.error('--start and --end must both be specified')
    else:
        dates = [args.date]

    release_timestamp = get_release_timestamp()

    expected = 0
    downloaded = 0

    for dt in reversed(dates):

        if pd.to_datetime(dt, utc=True) > release_timestamp:
            print(f'{dt} not yet available, current: {release_timestamp}')
            continue

        for dl in chain(*(SETS[s] for s in args.sets)):
            name = dl.name or dl.area_name or dl.area_type
            if args.name and name != args.name:
                continue
            expected += 1
            data_path = (base_path / f'{name}_{dt}.csv')
            if data_path.exists() and not args.overwrite:
                print('already exists:', data_path)
                downloaded += 1
                continue
            try:
                data_path = retrying_phe_download(name, dl.area_type, *dl.metrics,
                                                  area_name=dl.area_name, release=dt)
            except NoContent:
                print(f'no content for {name} on {dt}')
            else:
                print('downloaded: ', data_path)
                downloaded += 1

    if downloaded != expected:
        print(f'expected {expected}, but {downloaded} downloaded')
        return 1

    return 0


SETS = {
    'daily': [
        Download(nation, england_metrics, area_name='england'),
        Download(nation, vaccination_publish_date_metrics, name='vaccination'),
        Download(nation, [new_admissions, reported_virus_tests_sum]+standard_metrics),
        Download(nation, [case_demographics], name=f'case_demographics_{nation}'),
        Download(nation, [case_demographics_male], name=f'case_demographics_male'),
        Download(nation, [case_demographics_female], name=f'case_demographics_female'),
        Download(nhs_region, [new_admissions, in_hospital, in_mv_beds]),
    ]+[
        Download(area_type, standard_metrics) for area_type in (region, ltla)
    ]+[
        Download(nation, all_nation_metrics, area_name=area)
        for area in ('scotland', 'wales', 'northern ireland')
    ],
    'demographics': [
        Download(nation, [death_demographics], name=f'death_demographics_{nation}'),
        Download(nation, [admission_demographics], name=f'admission_demographics_{nation}'),
    ],
    'healthcare': [
        Download(nhs_trust, [new_admissions, in_hospital, in_mv_beds]),
    ],
    'deaths': [Download(area_type,
                        metrics=['newDeathsByDeathDate',
                                 'cumDeathsByDeathDate',
                                 'newDeathsByPublishDate',
                                 'cumDeathsByPublishDate'],
                        name=f'deaths_archive_{area_type}')
               for area_type in (nation, region, utla, ltla)],
}


if __name__ == '__main__':
    sys.exit(main())
