from itertools import zip_longest
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd

import series as s
from phe import summary_data


def waves(data, metric, title, sax=None, wax=None, rat=None, n=15, logy=True):

    if sax is None:
        fig, (sax, wax, rat) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), dpi=150)
        fig.set_facecolor('white')

    series = data[metric]
    minima = series.index[argrelextrema(series.values, np.less, order=n)[0]]
    maxima = series.index[argrelextrema(series.values, np.greater, order=n)[0]]
    waves = list(zip_longest(minima, maxima[1:]))

    series.plot(logy=logy, c='grey', ax=sax, title=title)
    series.loc[minima].plot(c='g', style='.', ax=sax)
    series.loc[maxima].plot(c='r', style='.', ax=sax)
    sax.title.set_fontweight('bold')

    df = pd.DataFrame()
    colours = []
    maxes = []

    for i, (start, end) in enumerate(waves, start=2):
        colour = cm.tab10(i)
        colours.append(colour)
        wave_data = data[metric].loc[start:end]
        wave_data.plot(ax=sax, color=colour)
        series = wave_data.reset_index()[metric]
        series -= series[0]
        name = f'Wave {i}, {start:%b %y}'
        df[name] = series
        maxes.append((series.max(), name))

    df.index.name = 'days'
    df.plot(logy=logy, title='Relative to Wave Start', ax=wax, color=colours)
    wax.legend(loc='lower right')

    relative_to_name = sorted(maxes)[-1][1]
    relative_to = df[relative_to_name]
    relative_to.fillna(relative_to.max(), inplace=True)
    ratio = df.divide(relative_to, axis='rows')
    ratio.plot(ax=rat, color=colours, legend=False, title='Relative to Worst', ylim=(0, 1))
    rat.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y * 100:,.0f}%"))


def summary_waves(nation, phe_series, title, sax=None, wax=None, rat=None, n=15, logy=True):
    data, _ = summary_data([phe_series], nation=nation)
    data.index.name = None
    metric = phe_series.metric
    waves(data, metric, title, sax, wax, rat, n, logy)


def plot_all(*, figsize, nation='england', dpi=150,
             cases=15, admissions=10, deaths=19,
             logy=True,
             **adjust):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, dpi=dpi)
    fig.set_facecolor('white')
    if adjust:
        fig.subplots_adjust(**adjust)
    for i, (series, title, n) in enumerate((
        (s.new_cases_sum, 'Cases', cases),
        (s.new_admissions_sum, 'Hospital Admissions', admissions),
        (s.new_deaths_sum, 'Deaths', deaths),
    )):
        sax, wax, rat = axes.T[i]
        summary_waves(nation, series, title, sax, wax, rat, n, logy)
