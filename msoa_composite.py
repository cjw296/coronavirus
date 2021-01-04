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
    with tqdm(total=path.stat().st_size, desc=path.name) as progress:
        with open(path) as source:
            line = source.readline()
            while line:
                progress.update(source.tell() - progress.n)
                yield line
                line = source.readline()


def add_from(path: Path, rows: dict, dt: date = None):
    reader = DictReader(lines_from(path))
    for row in reader:
        if dt:
            row[release_timestamp] = dt
        rows[key(row)] = row
    return reader.fieldnames


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

    for dt, path in tqdm(
            sorted(find_all('msoa_????-??-??.csv', earliest=args.start)),
            desc='source files'
    ):
        fieldnames = add_from(path, rows, dt)

    if not rows:
        return

    with open(args.output, 'w') as target:
        writer = DictWriter(target, fieldnames+[release_timestamp])
        writer.writeheader()
        for _, row in tqdm(sorted(rows.items()), desc='writing'):
            writer.writerow(row)


if __name__ == '__main__':
    main()
