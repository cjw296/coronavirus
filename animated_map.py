from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm

from animated import map_main
from constants import per100k, date_col, area_code
from phe import load_geoms, with_population, best_data
from plotting import show_area

rolling_days = 14


@lru_cache
def read_map_data():
    df, data_date = best_data()
    df = with_population(df)
    pivoted = df.pivot_table(values=per100k, index=[date_col], columns=area_code)
    smoothed = pivoted.fillna(0).rolling(14).mean().unstack().reset_index(name=per100k)
    return smoothed.set_index(date_col), data_date


def round_nearest(a, nearest):
    return (a/nearest).round(0) * nearest


def render_map(ax, frame_date, view, vmax=200, linthresh=30):
    df, _ = read_map_data()
    dt = frame_date.date()
    data = df.loc[dt]

    current_pct_geo = pd.merge(load_geoms(), data, how='outer', left_on='lad19cd',
                               right_on=area_code)

    ax = current_pct_geo.plot(
        ax=ax,
        column=per100k,
        legend=True,
        norm=SymLogNorm(linthresh=linthresh, vmin=0, vmax=vmax, base=10),
        cmap='inferno_r',
        vmin=0,
        vmax=vmax,
        legend_kwds={
            'fraction': 0.02,
            'extend': 'max',
            'format': '%.0f',
            'ticks': np.concatenate((np.arange(0, linthresh, 10),
                                     round_nearest(np.geomspace(linthresh, vmax, 5), 10))),
            'label': f'{rolling_days} day rolling average of new cases per 100,000 people'
        },
        missing_kwds={'color': 'lightgrey'},
    )
    show_area(ax, view)
    ax.set_title(f'COVID-19 cases for specimens dated {frame_date:%d %b %Y}')


def main():
    map_main('animated_map_ltla_cases', read_map_data, render_map, default_exclude=7)


if __name__ == '__main__':
    main()
