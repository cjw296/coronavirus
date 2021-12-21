from argparse import ArgumentParser
from functools import partial

import pandas as pd

from animated import parallel_render
from args import add_parallel_args, parallel_params
from prevalence import load_data, plot_prevalence


def main():
    parser = ArgumentParser()
    add_parallel_args(parser, default_duration=0.1, default_output='gif', from_date=False)
    parser.add_argument('--max-cases', type=float, default=1_200_000)
    parser.add_argument('--max-hospital', type=float, default=35_000)
    args = parser.parse_args()

    _, latest_date = load_data()
    dates = pd.date_range('2021-02-19', latest_date)

    parallel_render(f'animated_prevalence',
                    partial(plot_prevalence,
                            max_cases=args.max_cases,
                            max_hospital=args.max_hospital,
                            latest=latest_date),
                    dates, **parallel_params(args, dates))


if __name__ == '__main__':
    main()
