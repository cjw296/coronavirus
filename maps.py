from copy import copy
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import StrMethodFormatter

import series as s
from animated import round_nearest
from constants import area_code, metric, date_col, ltla, msoa
from geo import View, area_type_to_geoms, above, views
from phe import plot_summary, best_data, with_population
from plotting import show_area, per1m_formatter
from series import Series


def render_map(ax, frame_date, map: 'Map', view: View, label_top_5=False,
               title: Optional[str] = 'COVID-19 {map.series.label} as of {frame_date:%d %b %Y}'):

    cmap = map.get_cmap()
    r = map.range

    df, _ = map.data
    dt = frame_date.date()
    data_for_dt = df.loc[pd.to_datetime(dt)]

    data = pd.merge(area_type_to_geoms[map.area_type](), data_for_dt,
                    how='outer', left_on='code', right_on=area_code)

    ticks = norm = None
    if map.range.linthresh is not None:
        ticks = np.concatenate((
            round_nearest(np.linspace(r.vmin, r.linthresh, r.linticks), r.linnearest),
            round_nearest(np.geomspace(r.linthresh, r.vmax, r.logticks), r.lognearest))
        )
        norm = SymLogNorm(linthresh=r.linthresh, vmin=r.vmin, vmax=r.vmax, base=10)
    elif map.range.linticks:
        ticks = round_nearest(np.linspace(r.vmin, r.vmax, r.linticks), r.linnearest),

    plot_kwds = {}
    legend_kwds = {
        'fraction': view.legend_fraction,
        'format': StrMethodFormatter(map.tick_format),
        'ticks': ticks,
        'extend': 'max' if map.range.vmin == 0 else 'both',
        'label': map.axis_label()
    }

    if ticks is not None:
        legend_kwds['ticks'] = ticks

    if norm is not None:
        plot_kwds['norm'] = norm

    ax = data.plot(
        ax=ax,
        column=metric,
        legend=True,
        cmap=cmap,
        vmin=r.vmin,
        vmax=r.vmax,
        antialiased=map.antialiased,
        missing_kwds={'color': map.missing_color},
        legend_kwds=legend_kwds,
        **plot_kwds
    )
    show_area(ax, view)
    if title:
        ax.set_title(title.format(map=map, frame_date=frame_date))

    for places in view.outline:
        places.frame().geometry.boundary.plot(
            ax=ax, edgecolor=places.colour, linewidth=places.outline_width
        )

    for places in view.label:
        frame = places.frame()
        for name, geometry in zip(frame['name'], frame['geometry']):
            ax.annotate(
                name,
                xy=places.label_location(geometry),
                ha='center',
                fontsize=places.fontsize,
                fontweight=places.fontweight,
                color=places.colour
            )

    if label_top_5:
        top_5 = data.sort_values(metric, ascending=False).iloc[:5]
        for name, geometry in zip(top_5['name'], top_5['geometry']):
            ax.annotate(
                name,
                xy=above(geometry),
                ha='center',
                fontsize='x-large',
            )


def render_dt(
        data_date, earliest_date, to_date, area_type, map_type, view, bare, title,
        frame_date, image_path
):
    map = get_map(area_type, map_type)
    view = views[view]
    if bare:
        width, height, _ = view.layout(summary_height=0)
        plt.figure(figsize=(width, height), dpi=map.dpi)
        render_map(plt.gca(), frame_date, map, view, title=title)
    else:
        width, height, height_ratio = view.layout()
        fig, (map_ax, lines_ax) = plt.subplots(
            figsize=(width, height),
            nrows=2,
            gridspec_kw={'height_ratios': [height_ratio, 1], 'hspace': view.grid_hspace}
        )
        kw = {}
        if title:
            kw['title'] = title
        render_map(map_ax, frame_date, map, view, **kw)
        plot_summary(lines_ax, data_date, frame_date, earliest_date, to_date,
                     left_formatter=per1m_formatter,
                     right_series=(s.new_admissions_sum, s.new_deaths_sum),
                     title=False)
        fig.text(0.25, 0.07,
                 f'@chriswithers13 - '
                 f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
                 color='darkgrey',
                 zorder=-1)
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=map.dpi, bbox_inches='tight')
    plt.close()


@dataclass
class Range:
    vmin: int = 0
    vmax: int = None
    linthresh: int = None
    linticks: int = None
    linnearest: float = 1
    logticks: int = None
    lognearest: float = 1


@dataclass
class Map:
    series: Series
    range: Range = None
    default_exclude: int = 5
    default_view: str = 'uk'
    dpi: int = 90
    rolling_days: int = None
    cmap: str = None
    below_color: str = 'lightgrey'
    missing_color: str = 'grey'
    antialiased: bool = True
    area_type: str = None
    per_population: Optional[int] = 100_000
    tick_format: str = None

    def __post_init__(self):
        if self.tick_format is None:
            if self.per_population == 100:
                self.tick_format = '{x:,.0f}%'
            else:
                self.tick_format = '{x:,.0f}'

    def axis_label(self):
        label = self.series.title
        if self.rolling_days:
            label = f'{self.rolling_days} day rolling average of {label}'
        if self.per_population and self.per_population != 100:
            label = f'{label} per {self.per_population:,} people'
        return label

    def for_area_type(self, area_type):
        return replace(self, area_type=area_type)

    def get_cmap(self):
        cmap = copy(get_cmap(self.cmap or self.series.cmap))
        cmap.set_under(self.below_color)
        return cmap

    @cached_property
    def data(self):
        df, data_date = best_data(area_type=self.area_type, metric=self.series.metric)

        if self.per_population:
            df = with_population(
                df,
                source_cols=(self.series.metric,),
                dest_cols=(metric,),
                factors=(self.per_population,)
            )
        else:
            df.rename(columns={self.series.metric: metric}, inplace=True)

        if self.rolling_days:
            pivoted = df.pivot_table(values=metric, index=[date_col], columns=area_code)
            smoothed = pivoted.fillna(0).rolling(self.rolling_days).mean()
            df = smoothed.unstack().reset_index(name=metric)
        else:
            df = df.fillna(0)

        return df.set_index(date_col), data_date


@lru_cache
def get_map(area_type: str, map_type: str) -> Map:
    return MAPS[area_type][map_type].for_area_type(area_type)


ltla_cases = Map(
    s.new_cases,
    Range(vmin=0, linthresh=30, vmax=200, linticks=4, logticks=5, lognearest=10),
    rolling_days=14,
)

msoa_cases = Map(
    s.new_cases_rate,
    Range(vmin=30, linthresh=700, vmax=4000,
          linticks=7, linnearest=10, logticks=5, lognearest=100),
    default_exclude=0,
    default_view='england',
    missing_color='white',
    antialiased=False,
    per_population=None,
)

MAPS = {
    ltla: {
        'tested': Map(
            s.unique_people_tested_sum,
            Range(linthresh=3, vmax=15, linticks=4, logticks=5),
            per_population=100,
        ),
        'positivity': Map(
            s.unique_cases_positivity_sum,
            Range(linthresh=20, vmax=100, linticks=4, logticks=5),
            per_population=None,
        ),
        'cases': replace(ltla_cases, cmap='inferno_r'),
        'cases-red': replace(ltla_cases, default_view='england', rolling_days=7),
        'deaths': Map(
            s.new_deaths,
            Range(vmin=0, vmax=8),
            rolling_days=7,
            missing_color='white'
        ),
    },
    msoa: {
        'cases': replace(msoa_cases, dpi=150, cmap='inferno_r'),
        'cases-red': msoa_cases,
    },
}
