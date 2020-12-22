from argparse import ArgumentParser
from functools import partial

from dateutil.parser import parse as parse_date

from animated import parallel_render, add_date_arg
from constants import base_path, my_areas, london_areas, region, oxfordshire, relax_2
from phe import plot_with_diff, data_for_date


areas = dict(
    my_area=dict(
        diff_ylims=[-2, 200],
        data_for_date=partial(data_for_date, areas=my_areas)
    ),
    oxford=dict(
        diff_ylims=None,
        data_for_date=partial(data_for_date, areas=[oxfordshire])
    ),
    london=dict(
        diff_ylims=[-10, 50],
        data_for_date=partial(data_for_date, areas=london_areas)
    ),
    regions=dict(
        data_for_date=partial(data_for_date, area_types=region),
        diff_ylims=[-10, 13_000],
    )
)


def render(date, image_path, **kw):
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
        print(f'Could not render for {date}: {e}')


def main():
    parser = ArgumentParser()
    parser.add_argument('area', choices=areas.keys())
    add_date_arg(parser)
    parser.add_argument('--earliest',
                        help='min for x-axis. 2020-08-01 good for start of second wave')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--diff-log-scale', action='store_true')
    args = parser.parse_args()
    from_date = parse_date(args.from_date).date()
    dates = []
    for path in sorted(base_path.glob('coronavirus-cases_*-*-*.csv'), reverse=True):
        dt = parse_date(path.stem.split('_')[-1]).date()
        if dt >= from_date:
            dates.append(dt)

    data = areas[args.area]['data_for_date'](dates[0])
    max_sum = data.sum(axis=1).max()
    parallel_render(f'animated_cases_{args.area}',
                    partial(render,
                            ylim=max_sum,
                            earliest=args.earliest,
                            diff_log_scale=args.diff_log_scale,
                            **areas[args.area]),
                    dates, duration=args.duration)


if __name__ == '__main__':
    main()
