from argparse import ArgumentParser
from functools import partial, lru_cache

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter

from animated import parallel_render
from animated_summary import formatter_from_string
from args import add_parallel_args, parallel_params
from bars import Bars, BarsLookup
from plotting import per0_formatter, pct_formatter


@lru_cache
def data_for(series, earliest, pct: bool):
    bars = Bars.get(series, earliest=earliest)
    data, data_date = bars.data_for('*')
    if pct:
        data = data.divide(data.sum(axis='columns'), axis='rows')
    data = data.rolling(7).mean().dropna(how='all')
    return data, data_date, bars


def plot_demographic_bars(
        bars: BarsLookup, dt: pd.Timestamp, from_date: pd.Timestamp, ax:
        Axes, max_x: int = None, pct: bool = False
):
    data, _, bars = data_for(bars, earliest=from_date, pct=pct)
    data.loc[dt].plot.barh(ax=ax, width=0.9)
    ax.get_yaxis().label.set_visible(False)
    label = bars.series.label.capitalize()
    if pct:
        label = f'{label} (% of daily total)'
    ax.set_xlabel(label)
    xaxis = ax.get_xaxis()
    xaxis.label.set_fontsize(16)
    xaxis.set_major_formatter(pct_formatter if pct else per0_formatter)
    ax.set_xlim(0, max_x)


def plot_date(dt, *,
              series, maxes, from_date, pct: bool = False,
              image_path=None, dpi=100):
    dt = pd.to_datetime(dt)
    fig, axes = plt.subplots(nrows=1, ncols=len(series),
                             figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f'PHE data for {dt:%d %b %y}', fontsize=20)
    for i, s in enumerate(series):
        plot_demographic_bars(s, dt, from_date, axes[i], max_x=maxes[s], pct=pct)

    if image_path:
        plt.savefig(image_path / f'{dt.date()}.png', bbox_inches='tight', dpi=dpi)
        plt.close()


def main():
    parser = ArgumentParser()
    add_parallel_args(parser, default_output='gif', from_date='2020-03-15')
    parser.add_argument('--pct', action='store_true')
    args = parser.parse_args()

    data_dates = set()
    dates = None
    maxes = {}
    series = 'deaths_demographics_for_comparison', 'cases_demographics'
    for s in series:
        data, data_date, _ = data_for(s, earliest=args.from_date, pct=args.pct)
        data_dates.add(data_date)
        dates = data.index.values
        maxes[s] = max(data.max())

    assert len(data_dates) == 1, data_dates

    parallel_render(f'animated_demographics',
                    partial(
                        plot_date,
                        series=series,
                        maxes=maxes,
                        from_date=args.from_date,
                        pct=args.pct,
                    ),
                    dates, **parallel_params(args))


if __name__ == '__main__':
    main()
