from argparse import ArgumentParser
from csv import DictReader
from datetime import date

import pandas as pd

from args import add_date_arg
from constants import date_col, earliest_msoa, base_path
from msoa_composite import msoa_files, lines_from, Checker


def main():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    add_date_arg(group, '--start', default=earliest_msoa)
    group.add_argument('--composite', action='store_true')
    args = parser.parse_args()

    checkers = []
    min_date = str(date.max)
    max_date = str(date.min)

    if args.composite:
        files = [(None, base_path / 'msoa_composite.csv')]
    else:
        files = msoa_files(args.start)

    for path_dt, path in files:

        checker = Checker(path_dt, path)
        checkers.append(checker)

        for row in DictReader(lines_from(path)):
            checker.add_row(row)
            row_date = row[date_col]
            min_date = min(min_date, row_date)
            max_date = max(max_date, row_date)

    for dt in pd.date_range(min_date, max_date):
        d = str(dt.date())
        files = [i.name for i in reversed(checkers) if d in i.rows_per_date]
        if args.composite:
            if not files:
                print(f"{d}: missing")
        else:
            print(f"{d}: {', '.join(files)}")

    print()

    for checker in checkers:
        try:
            checker.check()
        except AssertionError as e:
            print(f'{checker.name}: {e}')

    print()
    out_of_date = (date.today()-pd.to_datetime(max_date).date()).days - Checker.expected_gap
    print(f'latest date seen: {max_date} ({out_of_date} days out of date)')


if __name__ == '__main__':
    main()
