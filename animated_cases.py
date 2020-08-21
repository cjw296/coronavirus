from functools import partial

from dateutil.parser import parse as parse_date

from animated import parallel_render
from constants import base_path, my_areas, london_areas
from phe import plot_areas

my_areas_ylims = [-2, 10]
london_areas_ylims = [-5, 20]

def render(date, image_path):
    try:
        plot_areas(
            date, uncertain_days=5, image_path=image_path,
            title='Evolution of PHE case reporting',
            areas=london_areas,
            diff_ylims=london_areas_ylims
        )
    except Exception as e:
        print(f'Could not render for {date}: {e}')


def main():
    dates = []
    for path in sorted(base_path.glob('coronavirus-cases_*-*-*.csv')):
        dates.append(parse_date(path.stem.split('_')[-1]).date())
    parallel_render('animated_cases', partial(render), dates, duration=0.12)


if __name__ == '__main__':
    main()
