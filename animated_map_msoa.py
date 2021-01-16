from functools import lru_cache

import pandas as pd
from matplotlib.cm import get_cmap

from animated import map_main, render_map
from constants import base_path, date_col, new_cases_rate, release_timestamp
from geo import msoa_geoms_20
from phe import read_csv


@lru_cache
def read_map_data():
    data = read_csv(base_path / 'msoa_composite.csv', index_col=date_col)
    return data.fillna(0), pd.to_datetime(data.iloc[-1][release_timestamp])


cmap = get_cmap('inferno_r')
cmap.set_under('lightgrey')


def render_cases_map(ax, frame_date, view):
    render_map(
        ax, frame_date, read_map_data, view, column=new_cases_rate,
        title=f'COVID-19 cases as of {frame_date:%d %b %Y}',
        vmin=30, linthresh=700, vmax=4000,
        linticks=7, linnearest=10, logticks=5, lognearest=100,
        load_geoms=msoa_geoms_20, cmap=cmap,
        antialiased=False,
        missing_kwds={'color': 'white'},
        legend_kwds={
            'extend': 'both',
            'label': f'7 day rolling average of new cases per 100,000 people '
                     f'by specimen date'
        })


def main():
    map_main('animated_map_msoa_cases', read_map_data, render_cases_map, dpi=150,
             default_view='england', default_exclude=0)


if __name__ == '__main__':
    main()
