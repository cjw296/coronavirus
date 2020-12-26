from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import requests
from matplotlib import ticker, transforms
from mpl_axes_aligner import align
from numpy import nan
from pandas import DataFrame, read_pickle, to_datetime, Series, DatetimeIndex

from constants import base_path
from download import find_latest
from geo import convert_df

api_key = 'iTsdIq-t8_cElnLjNmoRLA'
cases = 'corrected_covid_positive'
new_algo_date = date(2020, 7, 9)

def query(sql, **kw):
    response = requests.get('https://joinzoe.carto.com/api/v2/sql/', params={
        'q': sql,
        'api_key': api_key
    })
    data = response.json()
    if 'error' in data:
        return data['error']
    return DataFrame.from_records(data['rows'], **kw)


def pickle(df, name, for_date):
    path = base_path / f'zoe_{name}_{for_date:%Y-%m-%d}_{datetime.now():%Y-%m-%d-%H-%M}.pickle'
    df.to_pickle(path)
    return path


def data_for_date(dt, print_path=False, raise_if_missing=True):
    if isinstance(dt, datetime):
        dt = dt.date()
    uk_active_cases_glob = f'zoe_uk_active_cases_{dt}*'
    uk_active_cases_paths = sorted(base_path.glob(uk_active_cases_glob), reverse=True)
    if uk_active_cases_paths:
        uk_active_cases_path = uk_active_cases_paths[0]
        if print_path:
            print(uk_active_cases_path)
        return read_pickle(uk_active_cases_path)
    if raise_if_missing:
        raise ValueError(f"Nothing matching {uk_active_cases_glob}")


def find_previous(curr_date):
    prev_date = curr_date.date()
    while True:
        prev_date -= timedelta(days=1)
        prev_uk_active_cases = data_for_date(prev_date, print_path=True, raise_if_missing=False)
        if prev_uk_active_cases is not None:
            break
    return prev_date, prev_uk_active_cases


def plot_study(curr_date, prev_date, uk_active_cases, prev_uk_active_cases):

    prev_data_end = prev_date + timedelta(days=1)
    diff = uk_active_cases[cases].sub(prev_uk_active_cases[cases], fill_value=0)
    diff[prev_data_end:] = nan

    curr_est = uk_active_cases[cases].iloc[-1]

    month_ago = curr_date - timedelta(days=30)
    start = min(month_ago, diff.index.min() - timedelta(days=7))

    data_colour = '#113377'
    diff_colour = '#015c00'
    curr_colour = 'red'
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.set_facecolor('white')

    data_to_plot = uk_active_cases[cases][start:]
    data_to_plot.plot(ax=ax, grid=True, color=data_colour)
    ax.set_title(f'ZOE COVID Symptom Study as of {curr_date:%d %b %Y}')
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='y', labelcolor=data_colour)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('Predicted # of active cases', color=data_colour)

    ax.axhline(y=curr_est, color=curr_colour, zorder=-100)
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(1, curr_est + 1000, "{:,.0f}".format(curr_est), color=curr_colour, transform=trans,
            ha="right", va="bottom", zorder=1000)
    ax.patch.set_visible(False)

    diff_ax = ax.twinx()
    diff_ax.set_zorder(-1)
    diff_ax.tick_params(axis='y', labelcolor=diff_colour)
    diff_ax.set_ylabel(f'Change since {prev_date:%d %b %Y}', color=diff_colour)
    diff_to_plot = diff.loc[start:].reindex(data_to_plot.index)
    _ = diff_to_plot.plot(ax=diff_ax, drawstyle='steps', color=diff_colour, zorder=-1)

    align.yaxes(ax, curr_est, diff_ax, 0, 0.3)


def get_points(prefix, start_date):
    stop_path = sorted(base_path.glob(f'{prefix}_*'))[0]
    stop_date = to_datetime(stop_path.name.rsplit('_', 1)[0], format=f'{prefix}_%Y-%m-%d').date()
    point_date = start_date.date()
    while point_date > stop_date:
        point_glob = f'{prefix}_{point_date}*'
        point_paths = sorted(base_path.glob(point_glob), reverse=True)
        if point_paths:
            point_path = point_paths[0]
            point_df = read_pickle(point_path)
            point_data = point_df.iloc[-1]
            yield point_data
        point_date -= timedelta(days=1)


def plot_study_evolution(start_date, days=None):
    data_index = []
    data_values = []
    for point_data in get_points('zoe_uk_active_cases', start_date):
        data_index.append(point_data.name)
        data_values.append(point_data.corrected_covid_positive)
    for point_data in get_points('zoe_uk_time_series', start_date):
        data_index.append(point_data.name.tz_localize(None))
        data_values.append(point_data.corrected_covid_positive * 1000000)

    predictions_on_date = Series(data_values, DatetimeIndex(data_index)).sort_index()
    data_to_plot = data_for_date(start_date)[cases]

    if days is None:
        cut_off = max(new_algo_date, start_date - timedelta(days=60))
    else:
        cut_off = start_date - timedelta(days=days)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.set_facecolor('white')

    predictions_on_date[cut_off:].plot(label='Predication observed on date')
    data_to_plot[cut_off:].plot(label='Current prediction for date')
    plt.grid()

    ax.set_title(f'ZOE COVID Symptom Study')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('Predicted # of active cases')
    plt.legend()


def latest_map_data():
    path, dt = find_latest('zoe_prevalence_map_*.pickle', date_index=-2)
    df = read_pickle(path)
    gdf = convert_df(df, 'the_geom_webmercator')
    return dt, gdf
