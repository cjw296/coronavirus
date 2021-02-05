from argparse import ArgumentParser
from copy import copy
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache, partial, cached_property

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import StrMethodFormatter

import series as s
from animated import round_nearest, slowing_durations, parallel_render
from args import add_date_arg
from constants import date_col, area_code, ltla, second_wave, msoa, metric
from geo import View, above, views, area_type_to_geoms
from phe import with_population, best_data, plot_summary
from plotting import show_area
from series import Series


def render_map(ax, frame_date, map: 'Map', view: View, label_top_5=False):

    cmap = copy(get_cmap(map.cmap))
    cmap.set_under(map.below_color)
    r = map.range

    df, _ = map.data
    dt = frame_date.date()
    data_for_dt = df.loc[dt]

    data = pd.merge(area_type_to_geoms[map.area_type](), data_for_dt,
                    how='outer', left_on='code', right_on=area_code)

    ticks = np.concatenate((
        round_nearest(np.linspace(r.vmin, r.linthresh, r.linticks), r.linnearest),
        round_nearest(np.geomspace(r.linthresh, r.vmax, r.logticks), r.lognearest))
    )

    ax = data.plot(
        ax=ax,
        column=metric,
        legend=True,
        norm=SymLogNorm(linthresh=r.linthresh, vmin=r.vmin, vmax=r.vmax, base=10),
        cmap=cmap,
        vmin=r.vmin,
        vmax=r.vmax,
        antialiased=map.antialiased,
        missing_kwds={'color': map.missing_color},
        legend_kwds={
            'fraction': view.legend_fraction,
            'format': StrMethodFormatter('{x:,.0f}'),
            'ticks': ticks,
            'extend': 'max' if map.range.vmin == 0 else 'both',
            'label': map.axis_label()
        },
    )
    show_area(ax, view)
    ax.set_title(f'COVID-19 {map.series.label} as of {frame_date:%d %b %Y}')

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


def render_dt(data_date, earliest_date, to_date, area_type, map_type, view, frame_date, image_path):
    map = get_map(area_type, map_type)
    view = views[view]
    width, height, height_ratio = view.layout()
    fig, (map_ax, lines_ax) = plt.subplots(
        figsize=(width, height),
        nrows=2,
        gridspec_kw={'height_ratios': [height_ratio, 1], 'hspace': view.grid_hspace}
    )
    render_map(map_ax, frame_date, map, view)
    plot_summary(lines_ax, data_date, frame_date, earliest_date, to_date,
                 series=(s.new_admissions_sum, s.new_deaths_sum), title=False)
    fig.text(0.25, 0.07,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
             color='darkgrey',
             zorder=-1)
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=map.dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('area_type', choices=MAPS.keys())
    parser.add_argument('map')
    add_date_arg(parser, default=second_wave)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--exclude-days', type=int)
    add_date_arg(group, '--to-date')

    parser.add_argument('--output', default='mp4')
    parser.add_argument('--ignore-errors', dest='raise_errors', action='store_false')
    parser.add_argument('--view', choices=views.keys())
    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--duration', type=float, help='fast=0.05, slow=0.3')
    add_date_arg(group, '--single')
    args = parser.parse_args()

    map = get_map(args.area_type, args.map)
    view = args.view or map.default_view
    views[view].check()

    df, data_date = map.data

    exclude_days = map.default_exclude if args.exclude_days is None else map.default_exclude
    to_date = args.to_date or df.index.max().date() - timedelta(days=exclude_days)
    earliest_date = df.index.min().date()
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date, earliest_date, to_date, args.area_type, args.map, view)

    duration = args.duration or slowing_durations(dates)
    parallel_render(f'animated_map_{map.area_type}_{map.series.label}_{view}',
                    render, dates, duration,
                    args.output, raise_errors=args.raise_errors, max_workers=args.max_workers,
                    item=pd.to_datetime(args.single) if args.single else None)


@dataclass
class Range:
    vmin: int = 0
    vmax: int = None
    linthresh: int = None
    linticks: int = 4
    linnearest: float = 1
    logticks: int = 5
    lognearest: float = 1


@dataclass
class Map:
    series: Series
    range: Range = None
    default_exclude: int = 5
    default_view: str = 'uk'
    dpi: int = 90
    rolling_days: int = None
    cmap: str = 'inferno_r'
    below_color: str = 'lightgrey'
    missing_color: str = 'grey'
    antialiased: bool = True
    area_type: str = None
    add_population: bool = True

    def axis_label(self):
        label = self.series.title
        if self.add_population:
            label = f'{label} per 100,000 people'
        if self.rolling_days:
            label = f'{self.rolling_days} day rolling average of {label}'
        return label

    def for_area_type(self, area_type):
        return replace(self, area_type=area_type)

    @cached_property
    def data(self):
        df, data_date = best_data(area_type=self.area_type)

        if self.add_population:
            df = with_population(df, source_cols=(self.series.metric,), dest_cols=(metric,))
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


MAPS = {
    ltla: {
        'cases': Map(
            s.new_cases,
            Range(vmin=0, linthresh=30, vmax=200, linticks=4, logticks=5, lognearest=10),
            rolling_days=14
        )
    },
    msoa: {
        'cases': Map(
            s.new_cases_rate,
            Range(vmin=30, linthresh=700, vmax=4000,
                  linticks=7, linnearest=10, logticks=5, lognearest=100),
            default_exclude=0,
            default_view='england',
            dpi=150,
            missing_color='white',
            antialiased=False,
            add_population=False,
        )
    },
}


if __name__ == '__main__':
    main()
