import re
from argparse import ArgumentParser
from csv import DictReader, DictWriter
from datetime import datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Iterable

import requests

from constants import base_path

google_docs_csv = 'https://docs.google.com/spreadsheets/d/{key}/export?format=csv&sheet=table_1'


def fix_date(row, key):
    row[key] = str(datetime.strptime(row[key].rstrip('*'), '%d/%m/%Y').date())


def fix_number(row, key):
    row[key] = row[key].replace(',', '')


def fix_ci(row, key, name):
    ci = re.search(r'(\d+)% CI', key).group(1)
    actual, lower, upper = re.match(r'([\d.]+)% \(([\d.]+)%, +([\d.]+)%\)', row.pop(key)).groups()
    row[name] = str(Decimal(actual)/100)
    row[f'{name}_lower_{ci}'] = str(Decimal(lower)/100)
    row[f'{name}_upper_{ci}'] = str(Decimal(upper)/100)


def download(key) -> str:
    response = requests.get(google_docs_csv.format(key=key))
    assert response.status_code == 200, response.status_code
    return response.text


def parse(text: str) -> Iterable:
    source = StringIO(text)
    next(source)
    reader = DictReader(source)
    for row in reader:
        if row['Round'].startswith('*'):
            break
        del row['']
        fix_number(row, 'Tested swabs')
        fix_number(row, 'Positive swabs')
        fix_ci(row, 'Unweighted prevalence (95% CI)', 'unweighted')
        fix_ci(row, 'Weighted prevalence (95% CI)', 'weighted')
        fix_date(row, 'First sample')
        fix_date(row, 'Last sample')
        yield row


def write(rows: Iterable, path: Path):
    rows = iter(rows)
    row = next(rows)
    with path.open('w') as target:
        writer = DictWriter(target, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def parse_date(text):
    return datetime.strptime(text, '%Y-%m-%d').date()


def main():
    parser = ArgumentParser()
    parser.add_argument('url', help='google docs url from report')
    parser.add_argument('date', type=parse_date, help='yyyy-mm-dd')
    args = parser.parse_args()
    key = re.search('d/(.+)/edit', args.url).group(1)
    path = base_path / f'react_{args.date}.csv'

    text = download(key)
    rows = parse(text)
    write(rows, path)


if __name__ == '__main__':
    main()
