from argparse import ArgumentParser
from datetime import timedelta
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date
from matplotlib.colors import SymLogNorm
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import FuncFormatter

from animated import parallel_render, add_date_arg
from constants import ltla, code, cases, specimen_date, per100k, new_admissions, \
    new_deaths_by_death_date, lockdown1, lockdown2, new_virus_tests
from constants import code, per100k, date_col, new_cases_by_specimen_date, population, area_code
from download import find_latest
from phe import load_geoms, load_population, plot_summary

import series as s

rolling_days = 14



@lru_cache
def read_map_data(path):
    # so we only load it once per process!
    df = pd.read_csv(path)
    df = df[df['Area type'].isin(ltla)][[code, specimen_date, cases]]
    df = pd.merge(df, load_population(), on=code)
    df[per100k] = 100_000 * df[cases] / df[population]
    pivoted = df.pivot_table(values=per100k, index=[specimen_date], columns=code)
    return pivoted.fillna(0).rolling(rolling_days).mean().unstack().reset_index(name=per100k)


def round_nearest(a, nearest):
    return (a/nearest).round(0) * nearest


def render_map(ax, data_path, frame_date, vmax=200, linthresh=30):
    df = read_map_data(data_path)
    dt = str(frame_date.date())
    data = df[df[specimen_date] == dt]

    current_pct_geo = pd.merge(load_geoms(), data, how='outer', left_on='lad19cd',
                               right_on='Area code')

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
    ax.set_axis_off()
    ax.set_ylim(6460000, 7550000)
    ax.set_xlim(-600000, 200000)
    ax.set_title(f'PHE lab-confirmed cases for specimens dated {frame_date:%d %b %Y}')


def render_dt(data_path, data_date, earliest_date, to_date, frame_date, image_path):
    dt = str(frame_date.date())
    fig, (map_ax, lines_ax) = plt.subplots(
        figsize=(10, 15), nrows=2, gridspec_kw={'height_ratios': [9, 1], 'hspace': 0}
    )
    render_map(map_ax, data_path, frame_date)
    plot_summary(lines_ax, data_date, frame_date, earliest_date, to_date,
                 series=(s.new_admissions_sum, s.new_deaths_sum), title=False)
    fig.text(0.25, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{dt}.png', dpi=90, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    add_date_arg(parser)
    parser.add_argument('--exclude-days', default=7, type=int)
    parser.add_argument('--output', default='mp4')
    args = parser.parse_args()

    data_path, data_date = find_latest('coronavirus-cases_*-*-*.csv')
    df = read_map_data(data_path)

    to_date = parse_date(df[specimen_date].max()) - timedelta(days=args.exclude_days)
    earliest_date = parse_date(df[specimen_date].min())
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_path, data_date, earliest_date, to_date)

    durations = np.full((len(dates)), 0.05)
    durations[-30:] = np.geomspace(0.05, 0.3, 30)

    parallel_render('animated_map', render, dates, list(durations), args.output)


if __name__ == '__main__':
    main()
