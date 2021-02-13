import json
from datetime import timedelta, date, datetime
from functools import lru_cache, partial

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DayLocator, DateFormatter
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

import series as s
from constants import (
    base_path, specimen_date, area, cases, per100k, date_col, area_code, population,
    area_name, new_cases_by_specimen_date, pct_population, nation, region,
    ltla, utla, code, unique_people_tested_sum, national_lockdowns, msoa, release_timestamp
)
from download import find_latest, find_all
from geo import ltla_geoms
from plotting import stacked_bar_plot, per1k_formatter


def read_csv(data_path, start=None, end=None, metrics=None, index_col=None):
    kw = {}
    if metrics:
        kw['usecols'] = [date_col] + metrics
    data = pd.read_csv(data_path, index_col=index_col, parse_dates=[date_col], **kw)
    data.sort_index(inplace=True)
    if start or end:
        data = data.loc[start:end]
    return data


area_type_filters = {
    nation: ['Nation', 'nation'],
    region: ['Region', 'region'],
    ltla: ['Lower tier local authority', 'ltla'],
    utla: ['Upper tier local authority', 'utla'],
}


def available_dates(area_type=ltla, earliest=None):
    dates = set()
    for pattern in f'{area_type}_*.csv', 'coronavirus-cases_*.csv':
        for dt, _ in find_all(pattern, date_index=-1, earliest=earliest):
            dates.add(dt)
    return sorted(dates, reverse=True)


class NoData(ValueError): pass


def best_data(dt='*', area_type=ltla, areas=None, earliest=None, days=None):
    if area_type == msoa:
        assert dt == '*'
        data_path = base_path / 'msoa_composite.csv'
        data = read_csv(data_path)
        data_date = pd.to_datetime(data.iloc[-1][release_timestamp])
    else:
        try:
            data_path, data_date = find_latest(f'{area_type}_{dt}.csv')
        except FileNotFoundError:
            area_type_filter = area_type_filters.get(area_type)
            if area_type_filter is None:
                raise
            data_path, data_date = find_latest(f'coronavirus-cases_{dt}.csv')
            data = pd.read_csv(data_path, parse_dates=[specimen_date])
            data = data[data['Area type'].isin(area_type_filter)]
            data.rename(inplace=True, errors='raise', columns={
                area: area_name,
                code: area_code,
                specimen_date: date_col,
                cases: new_cases_by_specimen_date,
            })
        else:
            data = read_csv(data_path)

    if days:
        earliest = datetime.combine(data_date - timedelta(days=days), datetime.min.time())
    if earliest:
        data = data[data[date_col] >= pd.to_datetime(earliest)]

    if areas:
        data = data[data[area_code].isin(areas)]

    if data.empty:
        raise NoData(f'No {area_type} for {areas} available in {data_path}')

    return data, data_date


def cases_from(data):
    data = data.pivot_table(
        values=new_cases_by_specimen_date, index=[date_col], columns=area_name
    ).fillna(0)
    return data


def cases_data(area_type, areas, earliest_data, dt):
    all_data_, data_date_ = best_data(dt, area_type, areas, earliest_data)
    return cases_from(all_data_), data_date_


def tests_from(data):
    data = data.merge(load_population(), on=area_code, how='left')
    agg = data.groupby(date_col).agg(
        {unique_people_tested_sum: 'sum', population: 'sum'}
    )
    return 100 * agg[unique_people_tested_sum] / agg[population]


def plot_diff(ax, for_date, data, previous_date, previous_data,
              diff_ylims=None, diff_log_scale=None, earliest=None, colormap='viridis'):
    diff = data.sub(previous_data, fill_value=0)
    total_diff = diff.sum().sum()
    stacked_bar_plot(ax, diff, colormap)
    ax.set_title(f'Change between reports on {previous_date} and {for_date}: {total_diff:,.0f}')
    fix_x_axis(ax, diff, earliest)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.grid(True)
    if diff_ylims:
        ax.set_ylim(diff_ylims)
    if diff_log_scale:
        ax.set_yscale('symlog')
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.axhline(y=0, color='k')


def fix_x_axis(ax, data, earliest=None, number_to_show=50):
    ax.axes.set_axisbelow(True)
    ax.xaxis.set_tick_params(rotation=-90)
    ax.xaxis.label.set_visible(False)
    interval = max(1, round(data.shape[0]/number_to_show))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_locator(DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(DateFormatter('%d %b %Y'))
    ax.set_xlim(
        (pd.to_datetime(earliest) or data.index.min())-timedelta(days=0.5),
        data.index.max()+timedelta(days=0.5)
    )


def plot_stacked_bars(
        ax, data, average_days, average_end, title, testing_data,
        ylim, tested_ylim, earliest, colormap='viridis'
):

    handles = stacked_bar_plot(ax, data, colormap)

    if average_end is not None:
        average_label = f'{average_days} day average'
        mean = data.loc[:average_end].sum(axis=1).rolling(average_days).mean()
        handles.extend(
            ax.plot(mean.index, mean, color='k', label=average_label)
        )
        if not mean.empty:
            latest_average = mean.iloc[-1]
            handles.append(ax.axhline(y=latest_average, color='red', linestyle='dotted',
                                    label=f'Latest {average_label}: {latest_average:,.0f}'))

    if testing_data is not None:
        tested_ax = legend_ax = ax.twinx()
        tested_label = '% Population tested'
        if unique_people_tested_sum in testing_data:
            tested = tests_from(testing_data)
            if average_end is not None:
                tested = tested[:average_end]
            tested_color = 'darkblue'
            handles.extend(
                tested_ax.plot(tested.index, tested, color=tested_color,
                               label=tested_label, linestyle='dotted')
            )
        tested_ax.set_ylabel(f'{tested_label} in preceding 7 days',
                             rotation=-90, labelpad=14)
        tested_ax.set_ylim(0, tested_ylim)
        tested_ax.yaxis.tick_left()
        tested_ax.yaxis.set_label_position("left")
        tested_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}%'))
    else:
        legend_ax = ax

    fix_x_axis(ax, data, earliest)

    ax.set_ylabel(cases)
    if ylim:
        ax.set_ylim((0, ylim))
    ax.yaxis.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    for i, lockdown in enumerate(national_lockdowns):
        h = ax.axvspan(*lockdown, color='black', alpha=0.05,
                       zorder=-1000, label=f'National Lockdown')
        if not i:
            handles.append(h)

    legend_ax.legend(handles=handles, loc='upper left', framealpha=1)

    if title:
        ax.set_title(title)


def current_and_previous_data(get_data, start='*', diff_days=1):
    data, data_date = get_data(start)
    previous_date = data_date - timedelta(days=diff_days)
    previous_data = previous_data_date = None
    while previous_data is None and previous_date > date(2020, 2, 1):
        try:
            previous_data, previous_data_date = get_data(previous_date)
        except (FileNotFoundError, NoData):
            previous_date -= timedelta(days=1)

    if previous_data is None:
        previous_data = data
        previous_data_date = data_date
    return data, data_date, previous_data, previous_data_date


def plot_with_diff(data_date, get_data=cases_data, uncertain_days=5,
                   diff_days=1, diff_ylims=None, diff_log_scale=False,
                   image_path=None, title=None, to_date=None, ylim=None,
                   earliest='2020-10-01', area_type=ltla, areas=None, tested_ylim=None,
                   average_days=7, show_testing=True, colormap='viridis'):

    if earliest is None:
        earliest_data = None
    else:
        earliest_data = pd.to_datetime(earliest) - timedelta(days=average_days)

    get_data_for_areas = partial(get_data, area_type, areas, earliest_data)

    results = current_and_previous_data(get_data_for_areas, data_date, diff_days)
    data, data_date, previous_data, previous_data_date = results
    if show_testing:
        testing_data = best_data(data_date, area_type, areas, earliest_data)[0]
    else:
        testing_data = None

    if uncertain_days is None:
        average_end = None
    else:
        average_end = data.index.max()-timedelta(days=uncertain_days)

    end_dates = [previous_data.index.max(), data.index.max()]
    if to_date:
        end_dates.append(to_date)

    labels = pd.date_range(start=min(previous_data.index.min(), data.index.min()),
                           end=max(end_dates))
    data = data.reindex(labels, fill_value=0)
    previous_data = previous_data.reindex(labels, fill_value=0)

    fig, (bars_ax, diff_ax) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10),
                                           gridspec_kw={'height_ratios': [12, 2]})
    fig.set_facecolor('white')
    fig.subplots_adjust(hspace=0.45)

    with pd.plotting.plot_params.use("x_compat", True):
        plot_diff(
            diff_ax, data_date, data, previous_data_date, previous_data, diff_ylims, diff_log_scale,
            earliest, colormap
        )
        plot_stacked_bars(
            bars_ax, data, average_days, average_end, title, testing_data,
            ylim, tested_ylim,
            earliest, colormap
        )

    if image_path:
        plt.savefig(image_path / f'{data_date}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


@lru_cache
def load_population():
    # from http://coronavirus.data.gov.uk/downloads/data/population.json
    population = json.load((base_path / 'population.json').open())['general']
    population = pd.DataFrame({'population': population})
    population.index.name = area_code
    return population


def with_population(df,
                    source_cols=(new_cases_by_specimen_date,),
                    dest_cols=(per100k,),
                    factors=(100_000,)):
    df = df.reset_index().merge(load_population(), on=area_code, how='left')
    for source, dest, factor in zip(source_cols, dest_cols, factors):
        df[dest] = factor * df[source] / df[population]
    return df


def recent_phe_data_summed(latest_date, days=7):
    recent, _ = best_data(latest_date, days=days)
    recent_grouped = recent.groupby([area_code, area_name]).agg(
        {new_cases_by_specimen_date: 'sum', date_col: 'max'}
    )
    recent_grouped.rename(columns={date_col: specimen_date}, inplace=True)

    recent_pct = with_population(recent_grouped)
    recent_pct.set_index(area_code, inplace=True)
    recent_pct[pct_population] = recent_pct[per100k] / 1000
    recent_pct['recent_days'] = days

    return recent_pct


def map_data(for_date):

    recent_pct = recent_phe_data_summed(for_date)

    geoms = ltla_geoms()
    phe_recent_geo = pd.merge(
        geoms, recent_pct, how='outer', left_on='code', right_on=area_code
    )

    phe_recent_date = phe_recent_geo[specimen_date].max()

    phe_recent_title = (
        'PHE lab-confirmed cases summed over last '
        f"{int(phe_recent_geo['recent_days'].iloc[0])} days to {phe_recent_date:%d %b %Y}"
    )

    return phe_recent_date, phe_recent_geo, phe_recent_title


def summary_data(series, start=None, end=None, data_date=None):
    if data_date is None:
        data_path, data_date = find_latest('england_*.csv')
    else:
        data_path = base_path / f'england_{pd.to_datetime(data_date).date()}.csv'
    data = read_csv(
        data_path, start, end, [s_.metric for s_ in series], index_col=[date_col]
    ) / 7
    return data, data_date


def plot_summary(ax=None, data_date=None, frame_date=None, earliest_date=None, to_date=None,
                 left_series=(s.unique_people_tested_sum,),
                 left_formatter=per1k_formatter,
                 right_series=(s.new_cases_sum, s.new_admissions_sum, s.new_deaths_sum),
                 right_formatter=per1k_formatter,
                 title=True, figsize=(16, 5)):
    all_series = list(left_series)+list(right_series)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor('white')

    left_ax = ax
    right_ax = ax = left_ax.twinx()

    data, data_date = summary_data(all_series, earliest_date, to_date, data_date)

    possible_max = [dt for dt in (frame_date, to_date) if dt is not None]
    if possible_max:
        max_date = max(*possible_max) + timedelta(days=1)
        if max_date and max_date > data.index.max():
            data = data.reindex(pd.date_range(data.index.min(), max_date))

    for series_ax, series, formatter in (
            (left_ax, left_series, left_formatter),
            (right_ax, right_series, right_formatter),

    ):
        data.plot(ax=series_ax,
                  y=[s_.metric for s_ in series],
                  color=[s_.color for s_ in series], legend=False)

        series_ax.tick_params(axis='y', labelcolor=series[-1].color)
        series_ax.yaxis.set_major_formatter(formatter)
        series_ax.set_ylim(ymin=0)

    for lockdown in national_lockdowns:
        lockdown_obj = ax.axvspan(*lockdown, facecolor='black', alpha=0.2, zorder=0)

    lines = left_ax.get_lines() + right_ax.get_lines() + [lockdown_obj]
    ax.legend(lines, [s_.label for s_ in all_series]+['lockdown'],
              loc='upper left', framealpha=1)
    if title:
        ax.set_title(f'7 day moving average of PHE data for England as of {data_date:%d %b %Y}')
    if frame_date:
        ax.axvline(frame_date, color='red')
    ax.minorticks_off()

    xaxis = left_ax.get_xaxis()
    xaxis.label.set_visible(False)
