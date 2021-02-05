from argparse import ArgumentParser
from collections import Counter
from csv import DictReader, DictWriter
from datetime import date, datetime
from pathlib import Path
from typing import Sequence, Tuple, Optional

from tqdm.auto import tqdm

from args import add_date_arg
from constants import date_col, area_code, base_path, release_timestamp
from download import find_all, find_latest


def key(row):
    return row[date_col], row[area_code]


def lines_from(path: Path):
    with open(path) as source:
        lines = source.read().splitlines()

    for line in tqdm(lines, desc=path.name):
        yield line


class Checker:

    expected_msoa = 6791
    expected_diff = 7
    expected_gap = 5

    def __init__(self, path_dt: Optional[date], path: Path):
        self.name = path.name
        self.mtime = datetime.fromtimestamp(path.stat().st_mtime)
        self.path_date = path_dt
        self.is_composite = path_dt is None
        if self.is_composite:
            self.expected_diff = 1
        self.max_date: str = str(date.min)
        self.rows_per_date = Counter()

    def add_row(self, row):
        row_date = row[date_col]
        self.rows_per_date[row_date] += 1
        self.max_date = max(self.max_date, row_date)

    def check(self):
        last_date = None
        for key, count in sorted(self.rows_per_date.items()):
            assert count == self.expected_msoa, f'{key}: bad count: {count}'
            d = datetime.strptime(key, '%Y-%m-%d')
            if last_date is not None:
                diff = (d-last_date).days
                assert diff == self.expected_diff, f'{key}: bad diff: {diff}'
            last_date = d

        if not self.is_composite:
            max_date = datetime.strptime(self.max_date, '%Y-%m-%d').date()
            diff = (self.path_date - max_date).days
            assert diff == self.expected_gap, f'bad date gap: {diff} days'


def check_path(path):
    path, dt = find_latest(Path(path).name)
    checker = Checker(dt, path)
    for row in DictReader(lines_from(path)):
        checker.add_row(row)
    checker.check()


def add_from(path: Path, rows: dict, dt: date = None, check: bool = True):
    checker = Checker(dt, path)
    reader = DictReader(lines_from(path))
    max_date = str(date.min)
    for row in reader:
        if dt:
            row[release_timestamp] = dt
        rows[key(row)] = row
        if check:
            checker.add_row(row)
        if row[date_col] > max_date:
            max_date = row[date_col]
    if check:
        checker.check()
    tqdm.write(f'latest date in {path.name}: {max_date}')
    return reader.fieldnames


def msoa_files(earliest) -> Sequence[Tuple[datetime, Path]]:
    for dt, path in tqdm(
            sorted(find_all('msoa_????-??-??.csv', earliest=earliest)),
            desc='source files'
    ):
        yield dt, path


def main(args=None):
    parser = ArgumentParser()
    add_date_arg(parser, '--start', default=date.today())
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--output', default=base_path / 'msoa_composite.csv', type=Path)
    parser.add_argument('--no-check', dest='check', action='store_false')
    args = parser.parse_args(args)

    fieldnames = None
    rows = {}

    if args.start < date(2020, 1, 1):
        print("It's 2021 now!")
        return

    if args.output.exists() and not args.clean:
        add_from(args.output, rows, check=False)

    for dt, path in msoa_files(args.start):
        fieldnames = add_from(path, rows, dt, args.check)

    if fieldnames is None or not rows:
        return

    with open(args.output, 'w') as target:
        writer = DictWriter(target, fieldnames+[release_timestamp])
        writer.writeheader()
        for _, row in tqdm(sorted(rows.items()), desc='writing'):
            writer.writerow(row)


if __name__ == '__main__':
    main()
