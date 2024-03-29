import json
import re
from collections import defaultdict
from datetime import timedelta, datetime, date
from functools import lru_cache, partial, reduce
from typing import Sequence, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.ticker import NullLocator

import series as s
from constants import (
    base_path, specimen_date, area, per100k, date_col, area_code, population,
    area_name, new_cases_by_specimen_date, pct_population, nation, region,
    ltla, utla, code, national_lockdowns, msoa, release_timestamp, nations, in_hospital,
    new_admissions
)
from download import find_latest, find_all
from geo import ltla_geoms
from plotting import (
    per1k_formatter, male_colour, female_colour, stacked_area_plot,
    per0k_formatter, xaxis_months
)


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


@lru_cache
def best_data(dt='*', area_type=ltla, areas=None, earliest=None, days=None,
              metric=new_cases_by_specimen_date, file_prefix: str = None,
              metrics=(), date_index=False):
    metrics = list(metrics) if metrics else [metric]
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
            if metric != [new_cases_by_specimen_date]:
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

    missing = []
    if data.empty:
        missing = metrics
    else:
        for metric in metrics:
            series = data.get(metric)
            if series is None or series.empty:
                missing.append(metric)
    if missing:
        missing = ', '.join(missing)
        raise NoData(f'No {missing} for {file_prefix} in {areas} available in {data_path}')

    if date_index:
        data = data.set_index(date_col).sort_index()

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


def recent_cases_summed(days: int, exclude_days: int):
    # used for maps below and "top 10" stuff in workbook.
    recent, _ = best_data(days=days+exclude_days)
    recent = recent[recent[date_col] <= (recent[date_col].max()-pd.Timedelta(days=exclude_days))]

    recent_grouped = recent.groupby([area_code, area_name]).agg(
        {new_cases_by_specimen_date: 'sum', date_col: 'max'}
    )
    recent_grouped.rename(columns={date_col: specimen_date}, inplace=True)

    recent_pct = with_population(recent_grouped)
    recent_pct.set_index(area_code, inplace=True)
    recent_pct[pct_population] = recent_pct[per100k] / 1000
    recent_pct['recent_days'] = days

    return recent_pct


def summed_map_data(days, exclude_days):

    recent_pct = recent_cases_summed(days, exclude_days)

    geoms = ltla_geoms()
    summed_with_geoms = pd.merge(
        geoms, recent_pct, how='outer', left_on='code', right_on=area_code
    )

    summed_date = summed_with_geoms[specimen_date].max()

    return summed_date, summed_with_geoms


def nation_data(series, data_date=None, start=None, end=None, nation_name='England'):
    nation_name = nation_name.lower()
    if data_date in (None, '*'):
        data_path, data_date = find_latest(f'{nation_name}_*.csv')
    else:
        data_path = base_path / f'{nation_name}_{pd.to_datetime(data_date).date()}.csv'
    data = read_csv(data_path, start, end, [s_.metric for s_ in series], index_col=[date_col])
    return data, data_date


def summary_data(series, data_date=None, start=None, end=None, nation_name='england'):
    data, data_date = nation_data(series, data_date, start, end, nation_name)
    return data / 7, data_date


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
                     f'{nation} as of {data_date:%d %b %Y}')
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


def parse_bands(text):
    pref, low, sep, high = re.match(r'(0?)(\d+)((?:_to)?_)(\d+)', text).groups()
    return bool(pref), int(low), sep, int(high)


def load_demographic_data(prefix, nation_name, value, band_size=None, start=None, end=None):
    path, data_date = find_latest(f'{prefix}_*')
    raw = pd.read_csv(path, parse_dates=[date_col])
    data = raw[raw[area_name] == nation_name].pivot(index='date', columns='age', values=value)

    band_pref, band_low, band_sep, band_high = parse_bands(data.columns[0])
    existing_band_size = band_high - band_low + 1

    if band_size is None:
        band_size = existing_band_size
    elif band_size < existing_band_size or band_size % existing_band_size:
        raise ValueError(f'band_size must be a multiple of {existing_band_size}')

    new_bands = defaultdict(list)

    new_lower = existing_lower = 0

    while True:
        existing_upper = existing_lower + existing_band_size - 1
        new_upper = new_lower + band_size - 1
        try:
            if band_pref:
                key = f'{existing_lower:02d}{band_sep}{existing_upper:02d}'
            else:
                key = f'{existing_lower}{band_sep}{existing_upper}'
            existing = data[key]
        except KeyError:
            existing = data[f'{existing_lower}+']
            new_bands[f'{new_lower}+'].append(existing.loc[start:end])
            done = True
        else:
            new_bands[f'{new_lower} to {new_upper}'].append(existing.loc[start:end])
            done = False
        existing_lower += existing_band_size
        if existing_lower > new_upper:
            new_lower += band_size
        if done:
            break

    return (
        pd.DataFrame({key: reduce(pd.Series.add, series) for (key, series) in new_bands.items()}),
        data_date
    )


def diff(data: pd.DataFrame, *, days: int):
    return (data - data.shift(days)).iloc[days:]


genders = 'male', 'female'


def demographic_stream_plot(
        title, nation='England', variable='rate', band_size=10, start=None, end=None,
        order=2, log=False, figsize=(16, 9), uncertain_days: int = 5
):
    if start is None:
        start = date.today() - timedelta(days=30)
        date_formatter = DateFormatter('%d %b')
    else:
        date_formatter = DateFormatter('%b %y')
    data = {}
    dates = set()
    for gender in genders:
        data[gender], data_date = load_demographic_data(
            f'case_demographics_{gender}', nation, variable, band_size
        )
        dates.add(data_date)
    assert len(dates) == 1

    for _ in range(order):
        for gender, gender_data in tuple(data.items()):
            data[gender] = gender_data.diff().rolling(7).mean()

    for gender, gender_data in tuple(data.items()):
        data[gender] = gender_data.loc[start:end]

    columns = data['male'].columns
    fig, axes = plt.subplots(ncols=len(columns), figsize=figsize,
                             constrained_layout=True, sharex=True, sharey=True, dpi=150)
    if log:
        colours = ['black' for _ in genders]
    else:
        colours = [male_colour, female_colour]
    for i, bucket in enumerate(columns):
        ax = axes[i]
        ax.set_title(bucket)
        series = [data[gender][bucket] for gender in genders]
        stacked_area_plot(ax, series, colours, genders, uncertain_days, vertical=True)
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(date_formatter)
    fig.suptitle(f'{title} for {nation}', fontsize=16)
    if log:
        ax.set_xscale('symlog')
    else:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2)
    ax.margins(x=0, y=0)
    ax.autoscale_view()

    fig.text(0, -0.03,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
             color='darkgrey',
             zorder=1000)


def hospital_plot(nation_name=None, start=None, end=None, figsize=(16, 10), figs=None, ymin=0):
    if nation_name:
        nation_names = (nation_name,)
        figs = 1, 1
    else:
        nation_names = nations
        figs = figs or (2, 2)
    fig, axes = plt.subplots(*figs, figsize=figsize, dpi=150, sharex=True, constrained_layout=True)
    fig.set_facecolor('white')
    for ax, nation_name in zip(
            axes.reshape(-1) if len(nation_names) > 1 else [axes],
            nation_names
    ):
        data, data_date = nation_data([s.new_admissions, s.in_hospital], '*', start, end,
                                      nation_name)
        nation_ymin = data[in_hospital].min()-data[new_admissions].max() if ymin is None else ymin
        ax.fill_between(data.index, nation_ymin, data[in_hospital], label=s.in_hospital.title,
                        color=s.in_hospital.color)
        ax.fill_between(data.index, data[in_hospital] - data[new_admissions], data[in_hospital],
                        label=s.new_admissions.title, color=s.new_admissions.color)
        ax.set_ylim(nation_ymin, None)
        ax.margins(x=0, y=0)
        ax.yaxis.set_major_formatter(per0k_formatter)
        xaxis_months(ax)
        ax.set_title(nation_name)
        ax.autoscale_view()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=2)
    fig.text(0, -0.06,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
             color='darkgrey',
             zorder=1000)
