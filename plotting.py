from itertools import zip_longest
from typing import Union, List

import numpy as np
import pandas as pd
from bokeh.io import reset_output, output_notebook, show, output_file, save
from bokeh.resources import INLINE
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba
from matplotlib.dates import DAYS_PER_MONTH
from matplotlib.ticker import FuncFormatter

from geo import views


def show_area(ax, view=views['uk']):
    ax.set_axis_off()
    minx, miny, maxx, maxy = view.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def save_to_disk(p, filename, title, show_inline=True):
    if show_inline:
        reset_output()
        output_notebook(resources=INLINE)
        show(p)
    reset_output()
    output_file(filename, title=title)
    save(p)
    reset_output()
    output_notebook(resources=INLINE)


male_colour = '#1fc3aa'
female_colour = '#8624f5'

nation_tab10_cm_indices = [
    0,  # England
    6,  # NI
    2,  # Scotland
    3,  # Wales
]


def nation_colors(ncolors):
    assert ncolors == 4, ncolors
    return [plt.cm.tab10(i) for i in nation_tab10_cm_indices]


def color_with_alpha(color, alpha, size):
    colors = np.zeros((size, 4))
    for i, c in enumerate(to_rgba(color)):
        colors[:, i] = c
    colors[:, -1] = alpha
    return colors


def stacked_bar_plot(ax, data,
                     colormap: Union[str, callable], normalised_values=None,
                     alpha=None):
    pos_prior = neg_prior = pd.Series(0, data.index)
    ncolors = data.shape[1]

    if isinstance(colormap, str):
        colormap = get_cmap(colormap)
        if normalised_values:
            assert len(normalised_values) == ncolors
        else:
            normalised_values = np.linspace(0, 1, num=ncolors)
        colors = [colormap(value) for value in normalised_values]
    else:
        colors = colormap(ncolors)

    handles = []
    for i, (name, series) in enumerate(data.iteritems()):
        mask = series > 0
        bottom = np.where(mask, pos_prior, neg_prior)
        color = colors[i] if alpha is None else color_with_alpha(colors[i], alpha, data.index.size)
        handles.append(
            ax.bar(data.index, series, width=1.001, bottom=bottom, label=name, color=color)
        )
        pos_prior = pos_prior + np.where(mask, series, 0)
        neg_prior = neg_prior + np.where(mask, 0, series)
    return handles


def stacked_area_plot(
        ax: Axes, series: List[pd.Series], colors: List = None, labels: List[str] = None,
        uncertain_days: int = 0, vertical: bool = False
):
    index = series[0].index
    current_neg = current_pos = pd.Series(0.0, index=index)
    fill_between = ax.fill_betweenx if vertical else ax.fill_between
    for series, color, label in zip_longest(series, colors or (), labels or ()):
        color = color or ax._get_lines.get_next_color()

        next_pos = current_pos + series.where(series > 0, 0)
        fill_between(series.index, current_pos, next_pos, color=color, linewidth=0,
                     label=label or series.name)
        current_pos = next_pos

        next_neg = current_neg + series.where(series < 0, 0)
        fill_between(series.index, current_neg, next_neg, color=color, linewidth=0)
        current_neg = next_neg

    if uncertain_days:
        span = ax.axhspan if vertical else ax.axvspan
        span(index[-uncertain_days], index[-1], color='white', alpha=0.5)


def xaxis_months(ax):
    xaxis = ax.xaxis
    xaxis.label.set_visible(False)
    xaxis.set_tick_params(labelbottom=True)
    formatter = xaxis.get_major_formatter()
    formatter.scaled[1] = '%d %b'
    formatter.scaled[DAYS_PER_MONTH] = '%b %y'


per1m_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m")
per1k_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000:,.0f}k")
per0k_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000:,.1f}k")
per0_formatter = FuncFormatter(lambda y, pos: f"{y:,.0f}")
pct_formatter = FuncFormatter(lambda y, pos: f"{y*100:,.0f}%")
