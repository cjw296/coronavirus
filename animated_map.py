from datetime import timedelta
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date

from animated import parallel_render
from constants import base_path, ltla, code, cases, specimen_date
from download import find_latest
from phe import load_geoms, load_population

rolling_days = 14


@lru_cache
def read_data(data_date):
    # so we only load it once per process!
    df = pd.read_csv(base_path / f'coronavirus-cases_{data_date}.csv')
    df = df[df['Area type'].isin(ltla)][[code, specimen_date, cases]]
    pivoted = df.pivot_table(values=cases, index=[specimen_date], columns=code)
    return pivoted.fillna(0).rolling(rolling_days).mean().unstack().reset_index(name=cases)


# use a lower max here as we're smoothing to 14 days.
def render_dt(data_date, frame_date, image_path, vmax=0.01):
    df = read_data(data_date)
    dt = str(frame_date.date())
    data = df[df[specimen_date] == dt]

    current_pct = pd.merge(data, load_population(), how='outer', on=code).dropna()
    current_pct['% of population'] = 100 * current_pct[cases] / current_pct['population']

    current_pct_geo = pd.merge(load_geoms(), current_pct, how='outer', left_on='lad19cd',
                               right_on='Area code')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = current_pct_geo.plot(
        ax=ax,
        column='% of population',
        k=10,
        cmap='Reds', vmin=0, vmax=vmax,
        legend=True,
        legend_kwds={'fraction': 0.02,
                     'anchor': (0, 0),
                     'format': '%.3f',
                     'label': f'number or new cases, {rolling_days} '
                              f'day rolling average as % of area population'},
        missing_kwds={'color': 'lightgrey'},
    )
    ax.set_axis_off()
    ax.set_ylim(6400000, 7500000)
    ax.set_xlim(-600000, 200000)
    ax.set_title(f'PHE lab-confirmed cases for specimens dated {frame_date:%d %b %Y}')
    fig.tight_layout(rect=(0, .05, 1, 1))
    text = fig.text(0.19, 0.05,
                    f'@chriswithers13 - '
                    f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{dt}.png', dpi=90)
    plt.close()


def main():
    _, data_date = find_latest('coronavirus-cases_*-*-*.csv', index=-1)
    df = read_data(data_date)

    from_date = '2020-03-07'
    to_date = parse_date(df[specimen_date].max()) - timedelta(days=5)
    dates = pd.date_range(from_date, to_date)

    render = partial(render_dt, data_date)

    durations = np.full((len(dates)+1,), 0.05)
    durations[-30:] = np.geomspace(0.05, 0.3, 30)
    durations[-2] = 3

    parallel_render('pngs-phe', render, dates, list(durations))


if __name__ == '__main__':
    main()
