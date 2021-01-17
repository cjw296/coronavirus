from functools import lru_cache

from animated import map_main, render_map
from constants import per100k, date_col, area_code
from phe import with_population, best_data

rolling_days = 14


@lru_cache
def read_map_data():
    df, data_date = best_data()
    df = with_population(df)
    pivoted = df.pivot_table(values=per100k, index=[date_col], columns=area_code)
    smoothed = pivoted.fillna(0).rolling(rolling_days).mean()
    return smoothed.unstack().reset_index(name=per100k).set_index(date_col), data_date


def render_cases_map(ax, frame_date, view):
    render_map(
        ax, frame_date, read_map_data, view, column=per100k,
        title=f'COVID-19 cases as of {frame_date:%d %b %Y}',
        vmin=0, linthresh=30, vmax=200, linticks=4, logticks=5, lognearest=10,
        legend_kwds={
            'extend': 'max',
            'label': f'{rolling_days} day rolling average of new cases per 100,000 people '
                     f'by specimen date'
        })


def main():
    map_main('animated_map_ltla_cases', read_map_data, render_cases_map, default_exclude=5)


if __name__ == '__main__':
    main()
