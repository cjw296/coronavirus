from argparse import ArgumentParser
from functools import partial

from animated import parallel_render
from args import add_date_arg, add_parallel_args, parallel_params, parallel_to_date
from constants import my_areas, london_areas, oxford_areas, region, ltla, second_wave, \
    earliest_testing
from phe import available_dates, best_data, cases_data, tests_data, plot_cases_by_area


def render(date, image_path, **kw):
    plot_with_diff(
        date,
        image_path=image_path,
        title='Evolution of PHE case reporting',
        to_date=date.today(),
        **kw
    )


def main():
    parser = ArgumentParser()
    add_date_arg(parser, help='first release to use', default=earliest_testing)
    add_parallel_args(parser, default_duration=0.1, default_output='gif', from_date=False)
    parser.add_argument('area', choices=PARAMS.keys())
    add_date_arg(parser, '--earliest', help='min for x-axis', default=second_wave)
    parser.add_argument('--diff-log-scale', action='store_true')
    parser.add_argument('--diff-no-lims', action='store_true')
    parser.add_argument('--y-max-factor', type=float, default=1.02)
    args = parser.parse_args()

    params = PARAMS[args.area]

    if args.diff_no_lims:
        params.pop('diff_ylims', None)

    area_type = params.get('area_type', ltla)
    areas = params.get('areas')

    from_date = max(args.from_date, args.earliest)
    dates = available_dates(area_type, earliest=from_date)
    to_date = parallel_to_date(args, dates[0])
    if to_date != dates[0]:
        dates = [d for d in dates if d <= to_date]

    data, _ = best_data(dates[0], area_type, areas, args.earliest)

    max_cases = cases_data(data).sum(axis=1).max()*args.y_max_factor
    max_tests = tests_data(data).max()*args.y_max_factor

    parallel_render(f'animated_cases_{args.area}',
                    partial(render,
                            ylim=max_cases,
                            tested_ylim=max_tests,
                            earliest=args.earliest,
                            diff_log_scale=args.diff_log_scale,
                            **params),
                    dates, **parallel_params(args))


PARAMS = dict(
    my_area=dict(
        diff_ylims=[-2, 350],
        areas=my_areas,
    ),
    oxford=dict(
        areas=oxford_areas,
    ),
    wiltshire=dict(
        areas=['E06000054'],

    ),
    london=dict(
        diff_ylims=[-10, 1800],
        areas=london_areas,
    ),
    regions=dict(
        diff_ylims=[-100, 25_000],
        area_type=region,
    )
)


if __name__ == '__main__':
    main()
