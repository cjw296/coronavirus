import json
from datetime import timedelta, datetime, date
from functools import lru_cache, partial
from typing import Sequence, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullLocator

import series as s
from constants import (
    base_path, specimen_date, area, per100k, date_col, area_code, population,
    area_name, new_cases_by_specimen_date, pct_population, nation, region,
    ltla, utla, code, national_lockdowns, msoa, release_timestamp
)
from download import find_latest, find_all
from geo import ltla_geoms
from plotting import per1k_formatter


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


def available_dates(metric, area_type=ltla, earliest=None) -> List[date]:
    dates = set()
    patterns = [f'{area_type}_*.csv']
    if metric == new_cases_by_specimen_date:
        patterns.append('coronavirus-cases_*.csv')
    for pattern in patterns:
        for dt, _ in find_all(pattern, date_index=-1, earliest=earliest):
            dates.add(dt)
    if not dates:
        raise ValueError(f'nothing matching {patterns}')
    return sorted(dates, reverse=True)


class NoData(ValueError): pass


def best_data(dt='*', area_type=ltla, areas=None, earliest=None, days=None,
              metric=new_cases_by_specimen_date, file_prefix: str = None):
    if file_prefix is None:
        file_prefix = area_type
    if area_type == msoa:
        assert dt == '*'
        data_path = base_path / 'msoa_composite.csv'
        data = read_csv(data_path)
        data_date = pd.to_datetime(data.iloc[-1][release_timestamp])
    else:
        try:
            data_path, data_date = find_latest(f'{file_prefix}_{dt}.csv')
        except FileNotFoundError:
            if metric != new_cases_by_specimen_date:
                raise
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
                'Daily lab-confirmed cases': new_cases_by_specimen_date,
            })
        else:
            data = read_csv(data_path)

    if days:
        earliest = datetime.combine(data_date - timedelta(days=days), datetime.min.time())
    if earliest:
        data = data[data[date_col] >= pd.to_datetime(earliest)]

    if areas:
        data = data[data[area_code].isin(areas)]

    if metric not in data or data.empty:
        raise NoData(f'No {file_prefix} for {areas} available in {data_path}')

    return data, data_date


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
        'PHE cases by specimen date summed over last '
        f"{int(phe_recent_geo['recent_days'].iloc[0])} days to {phe_recent_date:%d %b %Y}"
    )

    return phe_recent_date, phe_recent_geo, phe_recent_title


def summary_data(series, data_date=None, start=None, end=None, nation='england'):
    if data_date in (None, '*'):
        data_path, data_date = find_latest(f'{nation}_*.csv')
    else:
        data_path = base_path / f'{nation}_{pd.to_datetime(data_date).date()}.csv'
    data = read_csv(
        data_path, start, end, [s_.metric for s_ in series], index_col=[date_col]
    ) / 7
    return data, data_date


def plot_summary(ax=None, data_date=None, frame_date=None,
                 earliest_date='2020-03-01', to_date=None,
                 left_series: Sequence[s.Series] = (),
                 left_formatter=per1k_formatter,
                 left_ymax: float = None,
                 right_series: Sequence[s.Series] = (),
                 right_formatter=per1k_formatter,
                 right_ymax: float = None,
                 title=True, figsize=(16, 5), x_labels=True,
                 show_latest=False,
                 log=False,
                 nation='england'):
    all_series = list(left_series)+list(right_series)
    if not all_series:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor('white')

    left_ax = ax
    right_ax = ax = left_ax.twinx()

    data, data_date = summary_data(all_series, data_date, earliest_date, to_date, nation)

    possible_max = [dt for dt in (frame_date, to_date) if dt is not None]
    if possible_max:
        max_date = max(possible_max) + timedelta(days=1)
        if max_date and max_date > data.index.max():
            data = data.reindex(pd.date_range(data.index.min(), max_date))

    max_values = []

    for series_ax, series, formatter, ymax in (
            (left_ax, left_series, left_formatter, left_ymax),
            (right_ax, right_series, right_formatter, right_ymax),

    ):
        if not series:
            series_ax.yaxis.set_major_locator(NullLocator())
            continue
        data.plot(ax=series_ax,
                  y=[s_.metric for s_ in series],
                  color=[s_.color for s_ in series], legend=False)

        if show_latest:
            for s_ in series:
                max_values.append((series_ax, s_, data[s_.metric].dropna()[-1]))

        series_ax.tick_params(axis='y', labelcolor=series[-1].color)
        series_ax.yaxis.set_major_formatter(formatter)
        series_ax.set_ylim(ymin=10 if log else 0, ymax=ymax)
        if log:
            series_ax.set_yscale('symlog')

    if not right_series:
        right_ax.xaxis.set_major_locator(left_ax.xaxis.get_major_locator())
        right_ax.xaxis.set_major_formatter(left_ax.xaxis.get_major_formatter())

    for lockdown in national_lockdowns:
        lockdown_obj = ax.axvspan(*lockdown, facecolor='black', alpha=0.2, zorder=0)

    lines = left_ax.get_lines() + right_ax.get_lines() + [lockdown_obj]
    if show_latest:
        labels = []
        for series_ax, s_, value in max_values:
            labels.append(f'{s_.label}: {value:,.0f}')
            if show_latest == 'lines':
                series_ax.axhline(y=value, color=s_.color, linestyle='dotted')
    else:
        labels = [s_.label for s_ in all_series]

    ax.legend(lines, labels+['lockdown'], loc='upper left', framealpha=1)
    if title:
        ax.set_title('7 day moving average of PHE data for '
                     f'{nation.capitalize()} as of {data_date:%d %b %Y}')
    if frame_date:
        ax.axvline(frame_date, color='red')
    ax.minorticks_off()

    xaxis = left_ax.get_xaxis()
    xaxis.label.set_visible(False)
    if not x_labels:
        xaxis.set_ticklabels([])


def latest_from(data):
    dates = {}
    values = {}
    for metric, series in data.items():
        dates[metric] = latest_date = series.last_valid_index()
        values[metric] = series.loc[latest_date]
    return dates, values


def latest_changes(*series, start='*'):
    data, _, previous_data, _ = current_and_previous_data(partial(summary_data, series), start)
    current_dates, current_values = latest_from(data)
    previous_dates, previous_values = latest_from(previous_data)
    print('Latest for England:')
    for s in series:
        current = current_values[s.metric]
        dtc = current_dates[s.metric]
        dtp = previous_dates[s.metric]
        diff = current_values[s.metric] - previous_values[s.metric]
        print(f'{current:,.0f} {s.label} (7 day average) as of {dtc:%a %d %b}, '
              f'{diff:+,.1f} since {dtp:%a %d %b}')
