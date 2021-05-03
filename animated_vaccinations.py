from argparse import ArgumentParser
from datetime import timedelta
from functools import partial

from matplotlib.dates import WeekdayLocator, DateFormatter
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from animated import parallel_render
from args import add_date_arg
from constants import area_name, date_col, earliest_vaccination
from download import find_latest
from pandas_tools import tuple_product_array
from vaccination import raw_vaccination_data


def selection_mapping(dt='*'):
    raw, _ = raw_vaccination_data(dt)
    all_areas = raw.set_index([area_name, date_col])
    area_names = all_areas.index.levels[0]

    first = tuple_product_array(area_names, 'First')
    first.shape = (2, 2)

    second = tuple_product_array(area_names, 'Second')
    second.shape = (2, 2)

    mapping = {'first': first, 'second': second}
    mapping['everything'] = np.hstack((first, second))
    return mapping


def render_plots(to_show, to_date=None, dt='*', size=5, dpi=200, image_path=None):
    if to_date:
        end, _ = raw_vaccination_data(to_date)
        end.set_index([area_name, date_col], inplace=True)
    else:
        end = None

    indexed, data_dt = raw_vaccination_data(dt)
    indexed.set_index([area_name, date_col], inplace=True)
    rows, cols = to_show.shape
    fig, axes = plt.subplots(rows, cols,
                             figsize=(size * cols, size * rows),
                             gridspec_kw={'hspace': 0.4}, dpi=dpi)
    fig.set_facecolor('white')

    for r in range(rows):
        for c in range(cols):
            name, type_ = to_show[r, c]
            ax = axes[r, c] if isinstance(axes, np.ndarray) else axes
            area = indexed.loc[name].sort_index()
            by_publish = f'cumPeopleVaccinated{type_}DoseByPublishDate'

            area[by_publish].plot(
                ax=ax, drawstyle="steps-post", label='Total by Report Date',
                color='red', title=f'{name} - {type_} Dose as reported {data_dt:%d %b}'
            )

            if end is not None:
                area_end = end.loc[name]
                ax.set_xlim(
                    area_end.index.min()-timedelta(days=1),
                    area_end.index.max()+timedelta(days=1),
                )
                ax.set_ylim(
                    0,
                    area_end[by_publish].max()*1.02
                )

    for ax in axes.flat if isinstance(axes, np.ndarray) else [axes]:
        ax.legend(loc='upper left')
        _, ymax = ax.get_ylim()
        xaxis = ax.axes.get_xaxis()
        yaxis = ax.axes.get_yaxis()
        if ymax > 1_000_000:
            yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))
        elif ymax > 1_000:
            yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000:.0f}k"))
        xaxis.set_major_locator(WeekdayLocator(6))
        xaxis.set_major_formatter(DateFormatter('%d %b'))
        plt.setp(xaxis.get_majorticklabels(), rotation=-90, horizontalalignment='center')
        xaxis.label.set_visible(False)

    if image_path:
        plt.savefig(image_path / f'{data_dt}.png', bbox_inches='tight')
        plt.close()


def main():

    parser = ArgumentParser()
    parser.add_argument('--group', choices=['first', 'second', 'everything'], default='everything')
    parser.add_argument('--area')
    parser.add_argument('-t', '--type',
                        choices=['first', 'second'])
    add_date_arg(parser, help='first release to use', default=earliest_vaccination)
    add_date_arg(parser, '--to-date', default=find_latest('vaccination_cum_*')[1])
    parser.add_argument('--duration', type=float, default=0.2)
    parser.add_argument('--raise-errors', action='store_true')
    args = parser.parse_args()

    if args.type:
        to_show = tuple_product_array(
            [' '.join(w.capitalize() for w in args.area.split('-'))],
            args.type.capitalize()
        )
        name = f'{args.area}_{args.type}'
    else:
        to_show = selection_mapping(args.from_date)[args.group]
        name = args.group

    dates = pd.date_range(args.from_date, args.to_date)
    parallel_render(f'animated_vaccinations_{name}',
                    partial(render_plots, to_show, args.to_date),
                    dates, duration=args.duration, raise_errors=args.raise_errors)


if __name__ == '__main__':
    main()

