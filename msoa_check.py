from argparse import ArgumentParser
from csv import DictReader
from datetime import date

import pandas as pd

from args import add_date_arg
from constants import date_col, earliest_msoa
from msoa_composite import msoa_files, lines_from, Checker


def main():
    parser = ArgumentParser()
    add_date_arg(parser, '--start', default=earliest_msoa)
    args = parser.parse_args()

    checkers = []
    min_date = str(date.max)
    max_date = str(date.min)

    for path_dt, path in msoa_files(args.start):

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
        print(f"{d}: {', '.join(files)}")

    print()

    for checker in checkers:
        try:
            checker.check()
        except AssertionError as e:
            print(f'{checker.name}: {e}')


if __name__ == '__main__':
    main()
