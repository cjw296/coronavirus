from argparse import ArgumentParser
from datetime import timedelta
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date
from matplotlib.colors import LinearSegmentedColormap

from animated import parallel_render
from constants import base_path, ltla, code, cases, specimen_date, relax_2
from download import find_latest
from phe import load_geoms, load_population

rolling_days = 14

per100k = 'per100k'
population = 'population'

transitions = [
    dict(vmax=0,  color=(0.97, 0.97, 0.94)),
    dict(vmax=5,  color=(255/255, 230/255, 0/255)),
    dict(vmax=30, color=(255/255, 143/255, 31/255)),
    dict(vmax=90, color=(0.7, 0, 0)),
    dict(vmax=200, color=(0, 0, 0)),
]


def make_plot_params_lookup(ts):
    lookup = {}
    transition = ts[0]
    bands = [(transition['vmax'], transition['color'])]
    i = 0
    transition = None
    for vmax in range(0, 130):
        if transition is None or (vmax > transition['vmax'] and i < len(ts)-1):
            i += 1
            transition = ts[i]
            transition_vmax = transition['vmax']
            bands.append((transition_vmax, transition['color']))

        colours = [(threshold / vmax if vmax else 0, colour)
                   for threshold, colour in bands[:-1]]
        prev_vmax, prev_colour = bands[-2]
        final_vmax, final_colour = bands[-1]
        try:
            factor = (vmax-prev_vmax)/(final_vmax-prev_vmax)
        except ZeroDivisionError:
            factor = 0
        colours.append((1, tuple(
            pc+(fc-pc)*factor for fc, pc in zip(final_colour, prev_colour)
        )))

        lookup[vmax] = LinearSegmentedColormap.from_list(f'cmap{i}', colours)
    return lookup


cmap_lookup = make_plot_params_lookup(transitions)


@lru_cache
def read_data(data_date):
    # so we only load it once per process!
    df = pd.read_csv(base_path / f'coronavirus-cases_{data_date}.csv')
    df = df[df['Area type'].isin(ltla)][[code, specimen_date, cases]]
    df = pd.merge(df, load_population(), on=code)
    df[per100k] = 100_000 * df[cases] / df[population]
    pivoted = df.pivot_table(values=per100k, index=[specimen_date], columns=code)
    return pivoted.fillna(0).rolling(rolling_days).mean().unstack().reset_index(name=per100k)


def render_dt(data_date, frame_date, image_path):
    df = read_data(data_date)
    dt = str(frame_date.date())
    data = df[df[specimen_date] == dt]

    current_pct_geo = pd.merge(load_geoms(), data, how='outer', left_on='lad19cd',
                               right_on='Area code')

    vmax = int(data[per100k].max())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = current_pct_geo.plot(
        ax=ax,
        column=per100k,
        k=10,
        cmap=cmap_lookup[vmax], vmin=0, vmax=vmax,
        legend=True,
        legend_kwds={'fraction': 0.02,
                     'format': '%.0f',
                     'label': f'number or new cases, {rolling_days} '
                              f'day rolling average of new cases per 100,000 people'},
        missing_kwds={'color': 'lightgrey'},
    )
    ax.set_axis_off()
    ax.set_ylim(6400000, 7500000)
    ax.set_xlim(-600000, 200000)
    ax.set_title(f'PHE lab-confirmed cases for specimens dated {frame_date:%d %b %Y}')
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(0.19, 0.05,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{dt}.png', dpi=90)
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--from-date', default=str(relax_2),
                        help='2020-03-07: data start, 2020-07-02: end of lockdown')
    parser.add_argument('--exclude-days', default=2, type=int)
    args = parser.parse_args()

    _, data_date = find_latest('coronavirus-cases_*-*-*.csv', index=-1)
    df = read_data(data_date)

    to_date = parse_date(df[specimen_date].max()) - timedelta(days=args.exclude_days)
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date)

    durations = np.full((len(dates)), 0.05)
    durations[-30:] = np.geomspace(0.05, 0.3, 30)

    parallel_render('pngs-phe', render, dates, list(durations))


if __name__ == '__main__':
    main()
