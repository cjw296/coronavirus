from argparse import ArgumentParser
from datetime import timedelta
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm

import series as s
from animated import parallel_render, slowing_durations
from args import add_date_arg
from constants import per100k, date_col, area_code, second_wave
from phe import load_geoms, plot_summary, with_population, best_data
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


def render_map(ax, frame_date, vmax=200, linthresh=30):
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
    show_area(ax)
    ax.set_title(f'COVID-19 cases for specimens dated {frame_date:%d %b %Y}')


def render_dt(data_date, earliest_date, to_date, frame_date, image_path):
    fig, (map_ax, lines_ax) = plt.subplots(
        figsize=(10, 15), nrows=2, gridspec_kw={'height_ratios': [9, 1], 'hspace': 0}
    )
    render_map(map_ax, frame_date)
    plot_summary(lines_ax, data_date, frame_date, earliest_date, to_date,
                 series=(s.new_admissions_sum, s.new_deaths_sum), title=False)
    fig.text(0.25, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=90, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    add_date_arg(parser, default=second_wave)
    parser.add_argument('--exclude-days', default=7, type=int)
    parser.add_argument('--output', default='mp4')
    args = parser.parse_args()

    df, data_date = read_map_data()

    to_date = df[date_col].max() - timedelta(days=args.exclude_days)
    earliest_date = df[date_col].min()
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date, earliest_date, to_date)

    parallel_render('animated_map', render, dates, slowing_durations(dates), args.output)


if __name__ == '__main__':
    main()
