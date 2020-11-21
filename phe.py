import concurrent.futures
import json
from datetime import timedelta, date, datetime
from functools import lru_cache, partial
from urllib.parse import parse_qs, urlparse

import requests
from dateutil.parser import parse as parse_date
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, LogLocator, SymmetricalLogLocator
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from constants import (
    base_path, utla, specimen_date, area, cases, lockdown, relax_2, code, ltla,
    phe_vmax, per100k, release_timestamp, lockdown1, lockdown2
)
from plotting import geoplot_bokeh, save_to_disk


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


def data_for_date(dt, areas=None, area_types=utla):
    path = base_path / f'coronavirus-cases_{dt}.csv'
    df = pd.read_csv(path)
    area_filter = df['Area type'].isin(area_types)
    if areas is not None:
        area_filter &= df['Area code'].isin(areas)
    by_area = df[area_filter]
    if by_area.empty:
        raise ValueError(f'No {area_types} for {areas} in {path}')
    data = by_area[[specimen_date, area, cases]].pivot_table(
        values=cases, index=[specimen_date], columns=area
    ).fillna(0)
    labels = pd.date_range(start=data.index.min(), end=data.index.max())
    return data.reindex([str(date.date()) for date in labels], fill_value=0)


def plot_diff(ax, for_date, data, previous_date, previous_data,
              diff_ylims=None, diff_log_scale=None):
    diff = data.sub(previous_data, fill_value=0)
    diff.plot(
        ax=ax, kind='bar', stacked=True, width=1, rot=-90, colormap='viridis',
        title=f'Change between reports on {previous_date} and {for_date}', legend=False
    )
    fix_x_dates(ax)
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


def fix_x_dates(ax):
    labels = ax.xaxis.get_ticklabels()
    for i, label in enumerate(reversed(labels)):
        if i % 4:
            label.set_text('')
    ax.axes.set_axisbelow(True)
    ax.xaxis.set_ticklabels(labels)


def plot_stacked_bars(ax, data, average_end, title=None, ylim=None):
    data.plot(ax=ax, kind='bar', stacked=True, width=1, rot=-90, colormap='viridis', legend=False,
              title=title)
    ax.set_ylabel(cases)
    if ylim:
        ax.set_ylim((0, ylim))

    mean = None
    if average_end is not None:
        mean = data.sum(axis=1).rolling(7).mean()
        mean.loc[str(average_end):] = np.NaN
        mean.plot(ax=ax, color='k', label='7 day average', rot=-90)

    fix_x_dates(ax)
    ax.yaxis.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    for i, lockdown in enumerate((lockdown1, lockdown2), start=1):
        labels = (data.index.get_loc(str(d))
                  if str(d) in data.index
                  else data.index.get_loc(data.index.max())+1
                  for d in lockdown)
        ax.axvspan(*labels, color='black', alpha=0.05,
                   zorder=-1000, label=f'lockdown {i}')

    if mean is not None:
        latest_average = mean.loc[str(average_end-timedelta(days=1))]
        ax.axhline(y=latest_average, color='blue', linestyle=':',
                   label=f'Latest average: {latest_average:.0f}')

    ax.legend(loc='upper left')


def plot_with_diff(for_date, data_for_date, uncertain_days,
                   diff_days=1, diff_ylims=None, diff_log_scale=False,
                   image_path=None, title=None, to_date=None, ylim=None):
    previous_date = for_date - timedelta(days=diff_days)

    data = data_for_date(for_date)
    previous_data = None
    while previous_data is None and previous_date > date(2020, 1, 1):
        try:
            previous_data = data_for_date(previous_date)
        except FileNotFoundError:
            previous_date -= timedelta(days=1)

    if previous_data is None:
        previous_data = data

    if uncertain_days is None:
        average_end = None
    else:
        average_end = parse_date(data.index.max()).date()-timedelta(days=uncertain_days)
    end_dates = [previous_data.index.max(), data.index.max()]
    if to_date:
        end_dates.append(str(to_date))

    labels = [str(dt.date()) for dt in
              pd.date_range(start=min(previous_data.index.min(), data.index.min()),
                            end=max(end_dates))]
    data = data.reindex(labels, fill_value=0)
    previous_data = previous_data.reindex(labels, fill_value=0)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [12, 2]})
    fig.set_facecolor('white')
    fig.subplots_adjust(hspace=0.5)

    plot_diff(axes[1], for_date, data, previous_date, previous_data, diff_ylims, diff_log_scale)
    plot_stacked_bars(axes[0], data, average_end, title, ylim)
    if image_path:
        plt.savefig(image_path / f'{for_date}.png', dpi=90, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_areas(for_date, areas, uncertain_days, diff_days=1, area_types=utla):
    plot_with_diff(
        for_date,
        partial(data_for_date, areas=areas, area_types=area_types),
        uncertain_days, diff_days
    )


def recent_phe_data_summed(latest_date, by, days=7):
    fields = [code, area]
    earliest_date = str(latest_date-timedelta(days=days))
    df = pd.read_csv(base_path / f'coronavirus-cases_{latest_date}.csv',
                     parse_dates=[specimen_date])
    la_data = df[df['Area type'].isin(by)][[area, code, specimen_date, cases]]
    recent = la_data[la_data[specimen_date] >= earliest_date]
    recent_grouped = recent.groupby(list(fields)).agg({cases: 'sum', specimen_date: 'max'})
    recent_grouped.reset_index(level=1, inplace=True)

    population = load_population()
    recent_pct = pd.merge(recent_grouped, population, how='outer', on=code)
    fraction = recent_pct[cases] / recent_pct['population']
    recent_pct[per100k] = 100_000 * fraction
    recent_pct['% of population'] = 100 * fraction

    recent_pct['recent_days'] = days
    return recent_pct


@lru_cache
def load_population():
    population = pd.DataFrame({'population': json.load((base_path / 'population.json').open())})
    population.index.name = code
    return population


@lru_cache
def load_geoms():
    return geopandas.read_file(str(base_path / 'ltlas_v1.geojson')).to_crs("EPSG:3857")


def map_data(for_date):

    recent_pct = recent_phe_data_summed(for_date, by=ltla)

    geoms = load_geoms()
    phe_recent_geo = pd.merge(
        geoms, recent_pct, how='outer', left_on='lad19cd', right_on='Area code'
    )

    return phe_recent_geo[specimen_date].max(), phe_recent_geo


def plot_map(phe_recent_geo, phe_recent_title):
    data = phe_recent_geo[['geometry', 'lad19nm', cases, 'population', '% of population']]
    p = geoplot_bokeh(data[~data.geometry.isnull()], phe_recent_title, '% of population',
                      vmax=phe_vmax, tooltips=[
            ('Name', '@lad19nm'),
            ('Cases', '@{Daily lab-confirmed cases}{1}'),
            ('Population', '@{population}{1}'),
            ('Percentage', '@{% of population}{1.111}%'),
        ])
    save_to_disk(p, "phe.html", title=phe_recent_title)
