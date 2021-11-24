from typing import Tuple, cast

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, DatetimeIndex


def heatmap(data: DataFrame, title: str,
            ax=None, figsize: Tuple[float, float] = (16, 9), dpi: int = 150, **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.set_facecolor('white')
    map_data = data.T
    map_data.columns = cast(DatetimeIndex, map_data.columns).strftime('%d %b')
    sns.heatmap(map_data, robust=True, xticklabels=len(map_data.columns) // 10, **kw)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.xaxis.label.set_visible(False)
    ax.xaxis.set_tick_params(rotation=-90)
