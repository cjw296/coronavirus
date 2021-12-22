from copy import copy
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from bokeh.models import GeoJSONDataSource, HoverTool
from bokeh.palettes import Reds
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import StrMethodFormatter

import series as s
from animated import round_nearest
from constants import area_code, metric, date_col, ltla, msoa, nhs_region, pct_population, \
    new_cases_by_specimen_date, population
from geo import View, area_type_to_geoms, above, views
from phe import plot_summary, best_data, with_population, summed_map_data
from plotting import show_area, per1m_formatter, per1k_formatter, save_to_disk
from series import Series


def render_map(ax, frame_date, map: 'Map', view: View, top: int = None,
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
            round_nearest(np.geomspace(r.linthresh, r.vmax, r.logticks), r.lognearest)[1:]
        ))
        norm = SymLogNorm(linthresh=r.linthresh, vmin=r.vmin, vmax=r.vmax, base=10)
    elif map.range.linticks:
        ticks = round_nearest(np.linspace(r.vmin, r.vmax, r.linticks), r.linnearest)

    plot_kwds = {}

    if view.legend_fraction:
        legend = True,
        legend_kwds = {
            'fraction': view.legend_fraction,
            'format': StrMethodFormatter(map.tick_format),
            'ticks': ticks,
            'extend': 'max' if map.range.vmin == 0 else 'both',
            'label': map.axis_label()
        }
        if ticks is not None:
            legend_kwds['ticks'] = ticks
    else:
        legend = False
        legend_kwds = {}
    if map.legend_kwds is not None:
        legend_kwds.update(map.legend_kwds)

    if norm is not None:
        plot_kwds['norm'] = norm

    # workaround until geopandas 0.9.0 available through conda
    if map.missing_color:
        plot_kwds['missing_kwds'] = {'color': map.missing_color}

    ax = data.plot(
        ax=ax,
        column=metric,
        legend=legend,
        cmap=cmap,
        vmin=r.vmin,
        vmax=r.vmax,
        antialiased=map.antialiased,
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

    if top:
        top_n = data.sort_values(metric, ascending=False).iloc[:top]
        for n, (name, geometry) in enumerate(zip(top_n['name'], top_n['geometry']), start=1):
            ax.annotate(
                f'{n}: {name}',
                xy=above(geometry),
                ha='center',
                fontsize='x-large',
            )


def render_dt(
        data_date, earliest_date, area_type, map_type, view, bare, legend, title, top, dpi,
        frame_date, image_path
):
    map = get_map(area_type, map_type)
    view = views[view]
    if not legend:
        view.legend_fraction = 0
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
        render_map(map_ax, frame_date, map, view, top, **kw)
        plot_summary(lines_ax, data_date, frame_date, earliest_date,
                     left_series=(s.reported_virus_tests_sum,),
                     left_formatter=per1m_formatter,
                     right_series=(s.new_admissions_sum, s.new_deaths_sum),
                     right_formatter=per1k_formatter,
                     title=False)
        fig.text(0.25, 0.07,
                 f'@chriswithers13 - '
                 f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
                 color='darkgrey',
                 zorder=-1)
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=dpi, bbox_inches='tight')
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
    missing_color: Optional[str] = 'grey'
    antialiased: bool = True
    area_type: str = None
    per_population: Optional[int] = 100_000
    tick_format: str = None
    file_prefix: str = None
    legend_kwds: dict = None

    def __post_init__(self):
        if self.tick_format is None:
            if self.per_population == 100:
                self.tick_format = '{x:,.0f}%'
            else:
                self.tick_format = '{x:,.0f}'
        if self.file_prefix is None:
            self.file_prefix = self.series.file_prefix

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
        df, data_date = best_data(
            area_type=self.area_type, metric=self.series.metric, file_prefix=self.file_prefix
        )

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
            Range(linthresh=3, vmax=15, linticks=4, logticks=4, lognearest=1),
            per_population=100,
        ),
        'partially_vaccinated': Map(
            s.partially_vaccinated,
            Range(vmax=100, linticks=6, linnearest=1),
            per_population=100,
            default_exclude=0,
        ),
        'positivity': Map(
            s.unique_cases_positivity_sum,
            Range(linthresh=20, vmax=100, linticks=5, linnearest=5, logticks=5, lognearest=5),
            per_population=None,
        ),
        'cases': replace(ltla_cases, cmap='inferno_r'),
        'cases-red': replace(ltla_cases, default_view='england', rolling_days=7),
        'cases-7': replace(ltla_cases, rolling_days=7, cmap='inferno_r'),
        'deaths': Map(
            s.new_deaths,
            Range(vmin=0, linthresh=3, vmax=9, linticks=4, logticks=4),
            rolling_days=7,
            missing_color='white'
        ),
    },
    msoa: {
        'cases': replace(msoa_cases, dpi=150, cmap='inferno_r'),
        'cases-red': msoa_cases,
    },
    nhs_region: {
        'admissions': Map(
            s.new_admissions,
            Range(vmin=0, vmax=9),
            rolling_days=7,
            missing_color=None,
        )
    }
}


def geoplot_matplotlib(df, ax, column, title, legend_kwds, vmax=None, missing_kwds=None):
    df.plot(ax=ax,
        column=column,
        k=10,
        cmap='Reds',
        legend=True,
        legend_kwds=legend_kwds,
        missing_kwds=missing_kwds,
        vmax=vmax,
    )
    show_area(ax)
    ax.set_title(title)


def geoplot_bokeh(data, title, column, tooltips, x_range=None, y_range=None, vmax=None):
    p = figure(title=title,
               plot_height=800,
               plot_width=600,
               toolbar_location='below',
               tools="pan, wheel_zoom, box_zoom, reset",
               active_scroll='wheel_zoom',
               match_aspect=True,
               x_range=x_range,
               y_range=y_range)

    if vmax is None:
        vmax = data[column].max()
    areas = p.patches(
        'xs', 'ys', source=GeoJSONDataSource(geojson=data.to_json()),
        fill_color=linear_cmap(column, tuple(reversed(Reds[256])), 0, vmax, nan_color='gray'),
        line_color='gray',
        line_width=0,
        fill_alpha=1
    )

    p.add_tools(HoverTool(
        renderers=[areas],
        tooltips=tooltips,
    ))

    return p


COMMON_LEGEND_KWDS = {
    'fraction': 0.02,
    'anchor': (0, 0),
    'location': 'bottom',
    'pad': 0.05,
}


def matplotlib_phe_map(ax, phe_recent_geo, phe_recent_date, phe_max):
    days = int(phe_recent_geo['recent_days'].iloc[0])
    max_value = phe_recent_geo[pct_population].max()
    legend_kwds = COMMON_LEGEND_KWDS.copy()
    legend_kwds['label'] = f'{days} day sum of cases as % of population (max: {max_value:.2f})'
    geoplot_matplotlib(phe_recent_geo, ax,
                       column=pct_population,
                       title=f"COVID-19 cases to {phe_recent_date:%d %b %Y}",
                       legend_kwds=legend_kwds,
                       vmax=phe_max,
                       missing_kwds={'color': 'lightgrey'})


def render_inline_map(ax, area_type, view_name, exclude_days, label):
    map_ = get_map(area_type, 'cases')
    legend_kwds = COMMON_LEGEND_KWDS.copy()
    legend_kwds['label'] = label
    map_ = replace(map_, legend_kwds=legend_kwds)
    frame_date = map_.data[0].index.max() - pd.Timedelta(days=exclude_days)
    render_map(ax, frame_date, map_, views[view_name])


def bokeh_phe_map(
        phe_recent_geo, phe_recent_date, phe_max
):
    phe_recent_title = (
        'PHE cases by specimen date summed over last '
        f"{int(phe_recent_geo['recent_days'].iloc[0])} days to {phe_recent_date:%d %b %Y}"
    )
    phe_data = phe_recent_geo[[
        'geometry', 'name', 'code', new_cases_by_specimen_date, population, pct_population
    ]]
    phe = geoplot_bokeh(
        phe_data, phe_recent_title, pct_population,
        vmax=phe_max, tooltips=[
            ('Name', '@name'),
            ('Code', '@code'),
            ('Cases', '@{'+new_cases_by_specimen_date+'}{1}'),
            ('Population', '@{population}{1}'),
            ('Percentage', '@{'+pct_population+'}{1.111}%'),
        ]
    )

    save_to_disk(phe, "phe.html", title='PHE new cases by specimen date', show_inline=False)


def case_maps(sum_vmax=None, exclude_days=0):
    sum_days = 7
    summed_date, summed_with_geoms = summed_map_data(sum_days, exclude_days)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 9), dpi=150)
    fig.set_facecolor('white')

    matplotlib_phe_map(ax[0], summed_with_geoms, summed_date, sum_vmax)
    render_inline_map(ax[1], ltla, 'uk', exclude_days,
                      label='14 day avg of cases per 100k people')
    # MSOA data is already averaged, so only goes up to a 4 days ago
    msoa_exclude_days = max(0, exclude_days-4)
    render_inline_map(ax[2], msoa, 'england', msoa_exclude_days,
                      label='7 day avg of cases per 100k people')

    for ax in plt.gcf().get_axes():
        if ax.get_label() == '<colorbar>':
            ax.tick_params(rotation=-90)

    plt.show()

    bokeh_phe_map(summed_with_geoms, summed_date, sum_vmax)
