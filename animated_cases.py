from argparse import ArgumentParser
from functools import partial

from animated import parallel_render, add_date_arg
from constants import my_areas, london_areas, oxfordshire, region, ltla
from phe import plot_with_diff, available_dates, best_data, cases_data, tests_data

all_params = dict(
    my_area=dict(
        diff_ylims=[-2, 350],
        areas=my_areas,
    ),
    oxford=dict(
        areas=[oxfordshire],
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


def render(date, image_path, raise_errors=False, **kw):
    try:
        plot_with_diff(
            date,
            uncertain_days=5,
            image_path=image_path,
            title='Evolution of PHE case reporting',
            to_date=date.today(),
            **kw
        )
    except Exception as e:
        print(f'Could not render for {date}: {type(e)}: {e}')
        if raise_errors:
            raise


def main():
    parser = ArgumentParser()
    parser.add_argument('area', choices=all_params.keys())
    add_date_arg(parser)
    add_date_arg(parser, '--earliest', help='min for x-axis', default=None)
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--diff-log-scale', action='store_true')
    parser.add_argument('--diff-no-lims', action='store_true')
    parser.add_argument('--raise-errors', action='store_true')
    parser.add_argument('--y-max-factor', type=float, default=1.02)
    args = parser.parse_args()

    params = all_params[args.area]

    if args.diff_no_lims:
        params.pop('diff_ylims', None)

    area_type = params.get('area_type', ltla)
    areas = params.get('areas')

    from_date = max(args.from_date, args.from_date if args.earliest is None else args.earliest)

    dates = available_dates(area_type, earliest=from_date)

    data, _ = best_data(dates[0], area_type, areas, from_date)

    max_cases = cases_data(data).sum(axis=1).max()*args.y_max_factor
    max_tests = tests_data(data).max()*args.y_max_factor

    parallel_render(f'animated_cases_{args.area}',
                    partial(render,
                            ylim=max_cases,
                            tested_ylim=max_tests,
                            earliest=args.earliest,
                            diff_log_scale=args.diff_log_scale,
                            raise_errors=args.raise_errors,
                            **params),
                    dates, duration=args.duration)


if __name__ == '__main__':
    main()
