import concurrent.futures
import json
from datetime import timedelta, date, datetime
from functools import lru_cache
from urllib.parse import parse_qs, urlparse

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.dates import DayLocator, DateFormatter
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, FuncFormatter
from tqdm.auto import tqdm

import series as s
from constants import (
    base_path, specimen_date, area, cases, per100k, release_timestamp, lockdown1,
    lockdown2, date_col, area_code, population,
    area_name, new_cases_by_specimen_date, pct_population, second_wave, nation, region,
    ltla, utla, code, unique_people_tested_sum
)
from download import find_latest
from plotting import stacked_bar_plot


def get(filters, structure, **params):
    _params={
        'filters':';'.join(f'{k}={v}' for (k, v) in filters.items()),
        'structure': json.dumps({element:element for element in structure}),
    }
    _params.update(params)
    response = requests.get('https://api.coronavirus.data.gov.uk/v1/data', timeout=20, params=_params)
    if response.status_code != 200:
        raise ValueError(f'{response.status_code}:{response.content}')
    return response.json()


def pickle(name, df):
    for_dates = df[release_timestamp].unique()
    assert len(for_dates) == 1, for_dates
    for_date, = for_dates
    path = base_path / f'phe_{name}_{for_date}_{datetime.now():%Y-%m-%d-%H-%M}.pickle'
    df.to_pickle(path)
    return path


def query(filters, structure, max_workers=None, **params):
    page = 1
    response = get(filters, structure, page=page, **params)
    result = response['data']
    max_page = int(parse_qs(urlparse(response['pagination']['last']).query)['page'][0])
    if max_page > 1:
        t = tqdm(total=max_page)
        t.update(1)
        todo = range(2, max_page+1)
        attempt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or max_page-1) as executor:
            while todo:
                attempt += 1
                bad = []
                t.set_postfix({'errors': len(bad), 'attempt': attempt})
                futures = {executor.submit(get, filters, structure, page=page, **params): page
                           for page in todo}
                for future in concurrent.futures.as_completed(futures):
                    page = futures[future]
                    try:
                        response = future.result()
                    except Exception as exc:
                        bad.append(page)
                        t.set_postfix({'errors': len(bad), 'attempt': attempt})
                    else:
                        result.extend(response['data'])
                        t.update(1)
                todo = bad
        t.close()
    return pd.DataFrame(result)


def download(name, area_type, *metrics, area_name=None, release=None, format='csv'):
    release = release or date.today()

    _params = {
        'areaType': area_type,
        'metric': metrics,
        'format': format,
        'release': str(release),
    }
    if area_name:
        _params['areaName'] = area_name
    response = requests.get(
        'https://api.coronavirus.data.gov.uk/v2/data', timeout=20, params=_params
    )
    if response.status_code != 200:
        raise ValueError(f'{response.status_code}:{response.content}')

    actual_release = datetime.strptime(
        response.headers['Content-Disposition'].rsplit('_')[-1], f'%Y-%m-%d.{format}"'
    ).date()
    if actual_release != release:
        raise ValueError(f'downloaded: {actual_release}, requested: {release}')
    path = (base_path / f'{name}_{actual_release}.csv')
    path.write_bytes(response.content)
    return path


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


def best_data(dt='*', area_type=ltla, areas=None, earliest=None, days=None):
    try:
        data_path, data_date = find_latest(f'{area_type}_{dt}.csv')
    except FileNotFoundError:
        data_path, data_date = find_latest(f'coronavirus-cases_{dt}.csv')
        data = pd.read_csv(data_path, parse_dates=[specimen_date])
        data = data[data['Area type'].isin(area_type_filters[area_type])]
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
        raise ValueError(f'No {area_type} for {areas} available in {data_path}')

    return data, data_date


def case_data(data):
    data = data.pivot_table(
        values=new_cases_by_specimen_date, index=[date_col], columns=area_name
    ).fillna(0)
    return data


def plot_diff(ax, for_date, data, previous_date, previous_data,
              diff_ylims=None, diff_log_scale=None):
    diff = data.sub(previous_data, fill_value=0)
    stacked_bar_plot(ax, diff, colormap='viridis')
    ax.set_title(f'Change between reports on {previous_date} and {for_date}')
    fix_x_axis(ax, diff)
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


def fix_x_axis(ax, data, number_to_show=50):
    ax.axes.set_axisbelow(True)
    ax.xaxis.set_tick_params(rotation=-90)
    ax.xaxis.label.set_visible(False)
    interval = max(1, round(data.shape[0]/number_to_show))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_locator(DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(DateFormatter('%d %b %Y'))
    ax.set_xlim(data.index.min()-timedelta(days=0.5), data.index.max()+timedelta(days=0.5))


def plot_stacked_bars(ax, data, average_end, title, ylim, all_data, tested_ylim=None):

    handles = stacked_bar_plot(ax, data, colormap='viridis')

    if average_end is not None:
        average_days = 7
        average_label = f'{average_days} day average'
        mean = data[:average_end].sum(axis=1).rolling(average_days).mean()
        handles.extend(
            ax.plot(mean.index, mean, color='k', label=average_label)
        )
        latest_average = mean.iloc[-1]
        handles.append(ax.axhline(y=latest_average, color='red', linestyle='dotted',
                                label=f'Latest {average_label}: {latest_average:,.0f}'))

    if unique_people_tested_sum in all_data:
        test_data = all_data.merge(load_population(), on=area_code, how='left')
        agg = test_data.groupby(date_col).agg(
            {unique_people_tested_sum: 'sum', population: 'sum'}
        )
        tested = (100 * agg[unique_people_tested_sum] / agg[population])
        if average_end is not None:
            tested = tested[:average_end]
        tested_color = 'darkblue'
        tested_ax = ax.twinx()
        tested_label = '% Population tested'
        handles.extend(
            tested_ax.plot(tested.index, tested, color=tested_color,
                           label=tested_label, linestyle=(0, (5, 5)))
        )
        tested_ax.set_ylabel(f'{tested_label} in preceding 7 days',
                             rotation=-90, labelpad=14)
        tested_ax.set_ylim(0, tested_ylim)
        tested_ax.yaxis.tick_left()
        tested_ax.yaxis.set_label_position("left")

        tested_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}%'))

    fix_x_axis(ax, data)

    ax.set_ylabel(cases)
    if ylim:
        ax.set_ylim((0, ylim))
    ax.yaxis.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    for i, lockdown in enumerate((lockdown1, lockdown2)):
        h = ax.axvspan(*lockdown, color='black', alpha=0.05,
                       zorder=-1000, label=f'National Lockdown')
        if not i:
            handles.append(h)

    ax.legend(handles=handles, loc='upper left', framealpha=1)

    if title:
        ax.set_title(title)


def plot_with_diff(data_date, uncertain_days,
                   diff_days=1, diff_ylims=None, diff_log_scale=False,
                   image_path=None, title=None, to_date=None, ylim=None,
                   earliest=None, area_type=None, areas=None):

    all_data, data_date = best_data(data_date, area_type, areas, earliest)

    previous_date = data_date - timedelta(days=diff_days)

    data = case_data(all_data)
    previous_data = None
    while previous_data is None and previous_date > date(2020, 1, 1):
        try:
            previous = best_data(previous_date, area_type, areas, earliest)
        except FileNotFoundError:
            previous_date -= timedelta(days=1)
        else:
            all_previous_data, previous_data_date = previous
            previous_data = case_data(all_previous_data)

    if previous_data is None:
        previous_data = data

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
        plot_diff(diff_ax, data_date, data, previous_date, previous_data, diff_ylims, diff_log_scale)
        plot_stacked_bars(bars_ax, data, average_end, title, ylim, all_data)

    if image_path:
        plt.savefig(image_path / f'{data_date}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_areas(for_date, areas=None, uncertain_days=5, diff_days=1, area_type=ltla,
               earliest=second_wave, **kw):
    plot_with_diff(
        for_date, uncertain_days, diff_days,
        earliest=earliest, areas=areas, area_type=area_type, **kw
    )


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


@lru_cache
def load_geoms():
    return geopandas.read_file(str(base_path / 'ltlas_v1.geojson')).to_crs("EPSG:3857")


def map_data(for_date):

    recent_pct = recent_phe_data_summed(for_date)

    geoms = load_geoms()
    phe_recent_geo = pd.merge(
        geoms, recent_pct, how='outer', left_on='lad19cd', right_on=area_code
    )

    phe_recent_date = phe_recent_geo[specimen_date].max()

    phe_recent_title = (
        'PHE lab-confirmed cases summed over last '
        f"{int(phe_recent_geo['recent_days'].iloc[0])} days to {phe_recent_date:%d %b %Y}"
    )

    return phe_recent_date, phe_recent_geo, phe_recent_title


def plot_summary(ax=None, data_date=None, frame_date=None, earliest_date=None, to_date=None,
                 series=(s.new_cases_sum, s.new_admissions_sum, s.new_deaths_sum),
                 tested_formatter=lambda y, pos: f"{y / 1_000_000:.1f}m", title=True):
    all_series = [s.unique_people_tested_sum] + list(series)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.set_facecolor('white')

    tests_ax = ax
    outcomes_ax = ax = tests_ax.twinx()

    if data_date is None:
        data_path, data_date = find_latest('england_*.csv')
    else:
        data_path = base_path / f'england_{data_date}.csv'

    data = read_csv(
        data_path, earliest_date, to_date, [s_.metric for s_ in all_series], index_col=[date_col]
    ) / 7
    if to_date and to_date > data.index.max():
        data = data.reindex(pd.date_range(data.index.min(), to_date))

    data.plot(ax=tests_ax,
              y=s.unique_people_tested_sum.metric,
              color=s.unique_people_tested_sum.color, legend=False)

    data.plot(ax=outcomes_ax,
              y=[s_.metric for s_ in series],
              color=[s_.color for s_ in series], legend=False)

    for lockdown in lockdown1, lockdown2:
        lockdown_obj = ax.axvspan(*lockdown, facecolor='black', alpha=0.2, zorder=0)

    lines = tests_ax.get_lines() + outcomes_ax.get_lines() + [lockdown_obj]
    ax.legend(lines, [s_.label for s_ in all_series]+['lockdown'],
              loc='upper left', framealpha=1)
    if title:
        ax.set_title(f'7 day moving average of PHE data for England as of {data_date:%d %b %Y}')
    if frame_date:
        ax.axvline(frame_date, color='red')
    ax.minorticks_off()

    outcomes_ax.tick_params(axis='y', labelcolor=s.new_admissions_sum.color)
    outcomes_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000:.0f}k"))
    tests_ax.tick_params(axis='y', labelcolor=s.unique_people_tested_sum.color)
    tests_ax.yaxis.set_major_formatter(FuncFormatter(tested_formatter))

    xaxis = tests_ax.get_xaxis()
    xaxis.label.set_visible(False)
