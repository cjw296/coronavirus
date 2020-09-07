from argparse import ArgumentParser
from functools import partial

from dateutil.parser import parse as parse_date

from animated import parallel_render
from constants import base_path, my_areas, london_areas, region, oxfordshire
from phe import plot_with_diff, data_for_date


areas = dict(
    my_area=dict(
        diff_ylims=[-2, 10],
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
        diff_ylims = [-10, 1000],
    )
)


def render(date, image_path, **kw):
    try:
        plot_with_diff(
            date,
            uncertain_days=5,
            diff_log_scale=True,
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
    parser.add_argument('--duration', type=float, default=0.1)
    args = parser.parse_args()
    dates = []
    for path in sorted(base_path.glob('coronavirus-cases_*-*-*.csv')):
        dates.append(parse_date(path.stem.split('_')[-1]).date())
    parallel_render(f'animated_cases_{args.area}',
                    partial(render, **areas[args.area]),
                    dates, duration=args.duration)


if __name__ == '__main__':
    main()