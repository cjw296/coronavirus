from argparse import ArgumentParser
from csv import DictReader, DictWriter
from datetime import date
from pathlib import Path

from tqdm import tqdm

from args import add_date_arg
from constants import date_col, area_code, base_path, release_timestamp
from download import find_all


def key(row):
    return row[date_col], row[area_code]


def lines_from(path: Path):
    with open(path) as source:
        lines = source.read().splitlines()

    for line in tqdm(lines, desc=path.name):
        yield line


def add_from(path: Path, rows: dict, dt: date = None):
    reader = DictReader(lines_from(path))
    for row in reader:
        if dt:
            row[release_timestamp] = dt
        rows[key(row)] = row
    return reader.fieldnames


def msoa_files(earliest) -> Sequence[Tuple[datetime, Path]]:
    for dt, path in tqdm(
            sorted(find_all('msoa_????-??-??.csv', earliest=earliest)),
            desc='source files'
    ):
        yield dt, path


def main():
    parser = ArgumentParser()
    add_date_arg(parser, '--start', default=date.today())
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--output', default=base_path / 'msoa_composite.csv', type=Path)
    args = parser.parse_args()

    fieldnames = None

    rows = {}
    if args.output.exists() and not args.clean:
        add_from(args.output, rows)
    if rows:
        print('latest date found: ', sorted(rows.keys())[-1][0])

    for dt, path in msoa_files(args.start):
        fieldnames = add_from(path, rows, dt)

    if fieldnames is None or not rows:
        return

    with open(args.output, 'w') as target:
        writer = DictWriter(target, fieldnames+[release_timestamp])
        writer.writeheader()
        for _, row in tqdm(sorted(rows.items()), desc='writing'):
            writer.writerow(row)


if __name__ == '__main__':
    main()
