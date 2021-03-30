from argparse import ArgumentParser
from functools import partial

import pandas as pd
from matplotlib import pyplot as plt

from animated import parallel_render
from constants import data_start
from maps import Map
from args import add_parallel_args, parallel_to_date, parallel_params
from phe import plot_summary, summary_data
from series import Series
import plotting


def formatter_from_string(text):
    exact = getattr(plotting, text, None)
    return exact or getattr(plotting, f'per{text}_formatter')


def render_dt(data_date, earliest_date, latest_date, dpi, figsize,
              left_series, right_series, left_formatter, right_formatter,
              frame_date, image_path):
    kw = {}
    if left_formatter:
        kw['left_formatter'] = formatter_from_string(left_formatter)
    if right_formatter:
        kw['right_formatter'] = formatter_from_string(right_formatter)

    plot_summary(None, data_date, frame_date, earliest_date, latest_date,
                 title=False, figsize=figsize,
                 left_series=left_series,
                 right_series=right_series,
                 **kw)
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def series_from_string(text):
    return [Series.lookup(metric) for metric in text.split(',')]


def main():
    parser = ArgumentParser()
    parser.add_argument('left', type=series_from_string)
    parser.add_argument('right', type=series_from_string)
    parser.add_argument('--lf')
    parser.add_argument('--rf')
    parser.add_argument('--dpi', type=int, default=Map.dpi)
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--height', type=int, default=2)
    add_parallel_args(parser, default_output='none')
    args = parser.parse_args()

    df, data_date = summary_data(args.left + args.right)

    earliest_date = data_start
    latest_date = df.index.max().date()
    to_date = parallel_to_date(args, latest_date)
    dates = pd.date_range(args.from_date, to_date)

    figsize = (args.width, args.height)
    render = partial(
        render_dt, data_date, earliest_date, latest_date, args.dpi, figsize,
        args.left, args.right, args.lf, args.rf
    )

    parallel_render(f'animated_summary_{args.width}_{args.height}',
                    render, dates, **parallel_params(args))


if __name__ == '__main__':
    main()
