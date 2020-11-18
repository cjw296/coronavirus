from argparse import ArgumentParser
from datetime import timedelta
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, LogNorm

from animated import parallel_render, output_types
from constants import base_path, ltla, code, cases, specimen_date, relax_2, per100k
from download import find_latest
from phe import load_geoms, load_population

rolling_days = 14

population = 'population'


@lru_cache
def read_data(data_date):
    # so we only load it once per process!
    df = pd.read_csv(base_path / f'coronavirus-cases_{data_date}.csv')
    df = df[df['Area type'].isin(ltla)][[code, specimen_date, cases]]
    df = pd.merge(df, load_population(), on=code)
    df[per100k] = 100_000 * df[cases] / df[population]
    pivoted = df.pivot_table(values=per100k, index=[specimen_date], columns=code)
    return pivoted.fillna(0).rolling(rolling_days).mean().unstack().reset_index(name=per100k)


def round_nearest(a, nearest):
    return (a/nearest).round(0) * nearest


def render_dt(data_date, frame_date, image_path, vmax=200, linthresh=30):
    df = read_data(data_date)
    dt = str(frame_date.date())
    data = df[df[specimen_date] == dt]

    current_pct_geo = pd.merge(load_geoms(), data, how='outer', left_on='lad19cd',
                               right_on='Area code')

    fig, ax = plt.subplots(figsize=(10, 10))

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
    fig.text(0.25, 0.09,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{dt}.png', dpi=90, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--from-date', default=str(relax_2),
                        help='2020-03-07: data start, 2020-07-02: end of lockdown')
    parser.add_argument('--exclude-days', default=7, type=int)
    parser.add_argument('--output', choices=output_types.keys(), default='gif')
    args = parser.parse_args()

    _, data_date = find_latest('coronavirus-cases_*-*-*.csv', index=-1)
    df = read_data(data_date)

    to_date = parse_date(df[specimen_date].max()) - timedelta(days=args.exclude_days)
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date)

    durations = np.full((len(dates)), 0.05)
    durations[-30:] = np.geomspace(0.05, 0.3, 30)

    parallel_render('pngs-phe', render, dates, list(durations), args.output)


if __name__ == '__main__':
    main()
