from argparse import ArgumentParser
from datetime import date
from functools import partial

from animated import parallel_render
from args import add_date_arg, add_parallel_args, parallel_params, parallel_to_date
from bars import plot_bars, BARS, Bars
from constants import second_wave, earliest_testing
from phe import available_dates


def main():
    parser = ArgumentParser()
    add_date_arg(parser, help='first release to use', default=earliest_testing)
    add_parallel_args(parser, default_duration=0.1, default_output='gif', from_date=False)
    parser.add_argument('config', choices=BARS.keys())
    add_date_arg(parser, '--earliest', help='min for x-axis', default=second_wave)
    parser.add_argument('--diff-log-scale', action='store_true')
    parser.add_argument('--diff-no-lims', action='store_true')
    parser.add_argument('--ylim', type=float)
    parser.add_argument('--y-max-factor', type=float, default=1.02)
    args = parser.parse_args()

    config = Bars.get(args.config, earliest=args.earliest)

    from_date = max(args.from_date, args.earliest)
    dates = available_dates(config.metric, config.data_file_stem, earliest=from_date)
    to_date = parallel_to_date(args, dates[0])
    if to_date != dates[0]:
        dates = [d for d in dates if d <= to_date]

    latest_date = dates[0]

    max_metric = config.data_for(latest_date)[0].sum(axis=1).max() * args.y_max_factor

    params = {}

    if args.diff_no_lims:
        params['diff_ylims'] = None

    lines = config.lines_for(latest_date)
    if lines:
        params['line_ylim'] = max(l.data.max() for l in lines) * args.y_max_factor

    parallel_render(f'animated_{args.config}',
                    partial(plot_bars,
                            config=config,
                            ylim=args.ylim or max_metric,
                            diff_log_scale=args.diff_log_scale,
                            show_title=True,
                            to_date=date.today(),
                            **params),
                    dates, **parallel_params(args, item_is_timestamp=False))


if __name__ == '__main__':
    main()
