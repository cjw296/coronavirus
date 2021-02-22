import re
from argparse import ArgumentParser
from csv import DictReader
from datetime import datetime
from decimal import Decimal
from io import StringIO
from typing import Iterable

import requests

from constants import england
from download import write_csv
from phe import load_population

google_docs_csv = 'https://docs.google.com/spreadsheets/d/{key}/export?format=csv&sheet=table_1'


def fix_date(row, key):
    row[key] = datetime.strptime(row[key].rstrip('*'), '%d/%m/%Y').date()


def fix_number(row, key):
    row[key] = row[key].replace(',', '')


def split_confidence(row, key, name):
    ci = re.search(r'(\d+)% CI', key).group(1)
    actual, lower, upper = re.match(r'([\d.]+)% \(([\d.]+)%, +([\d.]+)%\)', row.pop(key)).groups()
    row[name] = str(Decimal(actual)/100)
    row[f'{name}-lower-{ci}'] = str(Decimal(lower)/100)
    row[f'{name}-upper-{ci}'] = str(Decimal(upper)/100)


def add_mid_date(row, start, end):
    row['mid'] = row[start] + (row[end]-row[start])/2


def add_with_population(row, source, dest):
    england_population = load_population().loc[england].item()
    for suffix in '', '-lower-95', '-upper-95':
        row[dest+suffix] = int(float(row[source+suffix])*england_population)


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
        split_confidence(row, 'Unweighted prevalence (95% CI)', 'unweighted')
        split_confidence(row, 'Weighted prevalence (95% CI)', 'weighted')
        fix_date(row, 'First sample')
        fix_date(row, 'Last sample')
        add_mid_date(row, 'First sample', 'Last sample')
        add_with_population(row, 'weighted', 'people')
        yield row


def parse_date(text):
    return datetime.strptime(text, '%Y-%m-%d').date()


def main():
    parser = ArgumentParser()
    parser.add_argument('url', help='google docs url from report')
    parser.add_argument('date', type=parse_date, help='yyyy-mm-dd')
    args = parser.parse_args()
    key = re.search('d/(.+)/edit', args.url).group(1)
    filename = f'react_england_{args.date}.csv'

    text = download(key)
    rows = parse(text)
    write_csv(rows, filename)


if __name__ == '__main__':
    main()
