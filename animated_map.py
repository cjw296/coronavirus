from argparse import ArgumentParser
from functools import partial

import pandas as pd

from animated import slowing_durations, parallel_render
from args import add_parallel_args, parallel_params, parallel_to_date
from geo import views
from maps import render_dt, get_map, MAPS


def main():
    parser = ArgumentParser()
    parser.add_argument('area_type', choices=MAPS.keys())
    parser.add_argument('map')
    parser.add_argument('--bare', action='store_true', help='just the map')
    parser.add_argument('--title', help='override title template')
    parser.add_argument('--view', choices=views.keys())
    add_parallel_args(parser, default_duration=None)
    args = parser.parse_args()

    map = get_map(args.area_type, args.map)
    view = args.view or map.default_view
    views[view].check()

    df, data_date = map.data

    to_date = parallel_to_date(args, df.index.max().date(), map.default_exclude)
    earliest_date = df.index.min().date()
    dates = pd.date_range(args.from_date, to_date)

    render = partial(
        render_dt, data_date, earliest_date, to_date, args.area_type, args.map, view,
        args.bare, args.title,
    )

    params = parallel_params(args)
    params['duration'] = args.duration or slowing_durations(dates)
    parallel_render(f'animated_map_{map.area_type}_{args.map}_{view}',
                    render, dates, **params)


if __name__ == '__main__':
    main()
