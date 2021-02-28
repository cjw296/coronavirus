from argparse import ArgumentParser
from functools import partial

import pandas as pd
from matplotlib import pyplot as plt

import series as s
from animated import parallel_render
from constants import data_start
from maps import Map
from args import add_parallel_args, parallel_to_date, parallel_params
from phe import plot_summary, summary_data


def render_dt(data_date, earliest_date, to_date, dpi, figsize, frame_date, image_path):
    plot_summary(None, data_date, frame_date, earliest_date, to_date,
                 title=False, figsize=figsize)
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--dpi', type=int, default=Map.dpi)
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--height', type=int, default=2)
    add_parallel_args(parser, default_output='none')
    args = parser.parse_args()

    series = (s.unique_people_tested_sum, s.new_cases_sum, s.new_admissions_sum, s.new_deaths_sum)
    df, data_date = summary_data(series, end=args.to_date)

    to_date = parallel_to_date(args, df.index.max().date())
    earliest_date = data_start
    dates = pd.date_range(args.from_date, to_date)

    figsize = (args.width, args.height)
    render = partial(render_dt, data_date, earliest_date, to_date, args.dpi, figsize)

    parallel_render(f'animated_summary_{args.width}_{args.height}',
                    render, dates, **parallel_params(args))


if __name__ == '__main__':
    main()
