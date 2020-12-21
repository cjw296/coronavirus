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
from constants import base_path, ltla, code, cases, specimen_date, per100k, new_admissions, \
    new_deaths_by_death_date, lockdown1, lockdown2, new_virus_tests
from download import find_latest
from phe import load_geoms, load_population

rolling_days = 14

population = 'population'


@lru_cache
def read_map_data(data_date):
    # so we only load it once per process!
    df = pd.read_csv(base_path / f'coronavirus-cases_{data_date}.csv')
    df = df[df['Area type'].isin(ltla)][[code, specimen_date, cases]]
    df = pd.merge(df, load_population(), on=code)
    df[per100k] = 100_000 * df[cases] / df[population]
    pivoted = df.pivot_table(values=per100k, index=[specimen_date], columns=code)
    return pivoted.fillna(0).rolling(rolling_days).mean().unstack().reset_index(name=per100k)


def round_nearest(a, nearest):
    return (a/nearest).round(0) * nearest


def render_map(ax, data_date, frame_date, vmax=200, linthresh=30):
    df = read_map_data(data_date)
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


@lru_cache
def read_lines_data(data_date, earliest_date, to_date):
    # so we only load it once per process!
    path, _ = find_latest(f'phe_overview_{data_date}_*.pickle')
    overview_data = pd.read_pickle(path)
    overview_data = overview_data.set_index(pd.to_datetime(overview_data['date'])).sort_index()
    admissions_deaths = overview_data[[new_admissions, new_deaths_by_death_date, new_virus_tests]]
    return admissions_deaths.rolling(14).mean().loc[earliest_date:to_date]


def render_lines(ax, data_date, frame_date, earliest_date, to_date):
    data = read_lines_data(data_date, earliest_date, to_date)
    outcomes_ax = ax.twinx()
    tests_ax = ax

    data.plot(ax=outcomes_ax,
              y=[new_admissions, new_deaths_by_death_date],
              color=['darkblue', 'black'], legend=False)

    data.plot(ax=tests_ax,
              y=new_virus_tests,
              color='darkgreen', legend=False)

    for lockdown in lockdown1, lockdown2:
        lockdown_obj = ax.axvspan(*lockdown, facecolor='black', alpha=0.2)

    lines = outcomes_ax.get_lines() + tests_ax.get_lines() + [lockdown_obj]
    ax.legend(lines, ['hospitalised', 'died', 'tests', 'lockdown'])
    ax.axvline(frame_date, color='red')
    ax.minorticks_off()

    outcomes_ax.tick_params(axis='y', labelcolor='darkblue')
    outcomes_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/1_000:.0f}k"))
    tests_ax.tick_params(axis='y', labelcolor='darkgreen')
    tests_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/1_000_000:.1f}m"))

    xaxis = ax.get_xaxis()
    xaxis.label.set_visible(False)
    xaxis.set_major_locator(MonthLocator(interval=1))
    xaxis.set_major_formatter(DateFormatter('%b'))


def render_dt(data_date, earliest_date, to_date, frame_date, image_path):
    dt = str(frame_date.date())
    fig, (map_ax, lines_ax) = plt.subplots(
        figsize=(10, 15), nrows=2, gridspec_kw={'height_ratios': [9, 1], 'hspace': 0}
    )
    render_map(map_ax, data_date, frame_date)
    render_lines(lines_ax, data_date, frame_date, earliest_date, to_date)
    fig.text(0.25, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{dt}.png', dpi=90, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    add_date_arg(parser)
    parser.add_argument('--exclude-days', default=7, type=int)
    parser.add_argument('--output', default='gif')
    args = parser.parse_args()

    _, data_date = find_latest('coronavirus-cases_*-*-*.csv', index=-1)
    df = read_map_data(data_date)

    to_date = parse_date(df[specimen_date].max()) - timedelta(days=args.exclude_days)
    earliest_date = parse_date(df[specimen_date].min())
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date, earliest_date, to_date)

    durations = np.full((len(dates)), 0.05)
    durations[-30:] = np.geomspace(0.05, 0.3, 30)

    parallel_render('animated_map', render, dates, list(durations), args.output)


if __name__ == '__main__':
    main()
