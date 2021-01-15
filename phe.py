import json
from datetime import timedelta, date, datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DayLocator, DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, FuncFormatter

import series as s
from constants import (
    base_path, specimen_date, area, cases, per100k, date_col, area_code, population,
    area_name, new_cases_by_specimen_date, pct_population, second_wave, nation, region,
    ltla, utla, code, unique_people_tested_sum, first_dose_weekly,
    first_vaccination, second_dose_weekly, national_lockdowns, area_type, complete_dose_daily_cum,
    first_dose_daily_cum, second_dose_daily_cum, repo_path
)
from download import find_latest, find_all
from geo import ltla_geoms
from plotting import stacked_bar_plot


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
        raise NoData(f'No {area_type} for {areas} available in {data_path}')

    return data, data_date


def cases_data(data):
    data = data.pivot_table(
        values=new_cases_by_specimen_date, index=[date_col], columns=area_name
    ).fillna(0)
    return data


def tests_data(data):
    data = data.merge(load_population(), on=area_code, how='left')
    agg = data.groupby(date_col).agg(
        {unique_people_tested_sum: 'sum', population: 'sum'}
    )
    return 100 * agg[unique_people_tested_sum] / agg[population]


def plot_diff(ax, for_date, data, previous_date, previous_data,
              diff_ylims=None, diff_log_scale=None, earliest=None):
    diff = data.sub(previous_data, fill_value=0)
    total_diff = diff.sum().sum()
    stacked_bar_plot(ax, diff, colormap='viridis')
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
        ax, data, all_data, average_days, average_end, title, show_testing,
        ylim, tested_ylim, earliest
):

    handles = stacked_bar_plot(ax, data, colormap='viridis')

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

    if show_testing:
        tested_ax = ax.twinx()
        tested_label = '% Population tested'
        if unique_people_tested_sum in all_data:
            tested = tests_data(all_data)
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

    ax.legend(handles=handles, loc='upper left', framealpha=1)

    if title:
        ax.set_title(title)


def plot_with_diff(data_date, uncertain_days,
                   diff_days=1, diff_ylims=None, diff_log_scale=False,
                   image_path=None, title=None, to_date=None, ylim=None,
                   earliest=None, area_type=ltla, areas=None, tested_ylim=None,
                   average_days=7, show_testing=True):

    if earliest is None:
        earliest_data = None
    else:
        earliest_data = pd.to_datetime(earliest) - timedelta(days=average_days)

    all_data, data_date = best_data(data_date, area_type, areas, earliest_data)

    previous_date = data_date - timedelta(days=diff_days)

    data = cases_data(all_data)
    previous_data = None
    while previous_data is None and previous_date > date(2020, 1, 1):
        try:
            previous = best_data(previous_date, area_type, areas, earliest_data)
        except (FileNotFoundError, NoData):
            previous_date -= timedelta(days=1)
        else:
            all_previous_data, previous_data_date = previous
            previous_data = cases_data(all_previous_data)

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
        plot_diff(
            diff_ax, data_date, data, previous_date, previous_data, diff_ylims, diff_log_scale,
            earliest
        )
        plot_stacked_bars(
            bars_ax, data, all_data, average_days, average_end, title, show_testing,
            ylim, tested_ylim,
            earliest
        )

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
        data_path = base_path / f'england_{pd.to_datetime(data_date).date()}.csv'

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

    for lockdown in national_lockdowns:
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


def latest_raw_vaccination_data():
    new_weekly_path, new_weekly_dt = find_latest('vaccination_????-*', date_index=-1)
    cum_path, cum_dt = find_latest('vaccination_cum_*', date_index=-1)
    assert cum_dt == new_weekly_dt, f'{cum_dt} != {new_weekly_dt}'
    new_weekly_df = read_csv(new_weekly_path)
    cum_df = read_csv(cum_path)
    raw = pd.merge(new_weekly_df, cum_df, how='outer',
                   on=[date_col, area_type, area_code, area_name]).sort_values(
        [date_col, area_code])
    # this isn't currently populated:
    assert raw[complete_dose_daily_cum].isnull().all()
    return raw, new_weekly_dt


def weekly_data(raw, nation_codes):
    weekly = raw[[date_col, area_code, first_dose_weekly, second_dose_weekly]].dropna()

    # data massaging:
    initial = pd.DataFrame(
        {date_col: pd.to_datetime(first_vaccination), first_dose_weekly: 0, second_dose_weekly: 0},
        index=nation_codes
    )
    initial.index.name = area_code

    to_fudge = weekly.set_index(date_col).sort_index().loc[:'2020-12-20'].reset_index()
    fudged = to_fudge.groupby(area_code).agg(
        {date_col: 'max', first_dose_weekly: 'sum', second_dose_weekly: 'sum'}
    )

    normal = weekly.set_index(date_col).sort_index().loc['2020-12-21':]
    data = pd.concat((df.reset_index() for df in (initial, fudged, normal)))
    data.rename(columns={first_dose_weekly: 'first_dose', second_dose_weekly: 'second_dose'},
                  errors='raise', inplace=True)
    return data


def daily_data(raw, weekly):
    initial_daily = weekly.groupby(area_code).agg({
        'first_dose': 'sum', 'second_dose': 'sum', date_col: 'max'
    }).reset_index()

    daily_rows = raw[[date_col, area_code, first_dose_daily_cum, second_dose_daily_cum]].dropna()
    daily_rows.rename(
        columns={first_dose_daily_cum: 'first_dose', second_dose_daily_cum: 'second_dose'},
        errors='raise', inplace=True)
    daily = pd.concat(
        [initial_daily, daily_rows[daily_rows[date_col] > initial_daily[date_col].max()]])

    return daily.set_index([date_col, area_code]).groupby(area_code).diff().dropna().reset_index()


def vaccination_dashboard():
    # input data:
    raw, data_date = latest_raw_vaccination_data()
    names_frame = raw[[area_code, area_name]].drop_duplicates()
    nation_codes = names_frame[area_code]
    nation_populations = load_population().loc[nation_codes]
    total_population = nation_populations.sum()[0]

    weekly = weekly_data(raw, nation_codes)
    daily = daily_data(raw, weekly)
    all_data = pd.concat([weekly, daily])

    # look out for weirdness
    assert (all_data['first_dose'] >= 0).all()
    assert (all_data['second_dose'] >= 0).all()

    all_data['start'] = all_data[date_col].shift(len(nation_codes))
    all_data['duration'] = all_data[date_col] - all_data['start']
    all_data = all_data.set_index([date_col, area_code])
    all_data['any'] = all_data.groupby(level=-1)['first_dose'].cumsum()
    all_data['full'] = all_data.groupby(level=-1)['second_dose'].cumsum()
    all_data['partial'] = all_data['any'] - all_data['full']
    data = pd.merge(all_data.reset_index(), names_frame, on=area_code)
    max_date = data[date_col].max()

    # data for plotting:
    latest = data[[area_name, area_code, 'full', 'any', 'partial']][
        data[date_col] == max_date].copy()
    latest = pd.merge(latest, nation_populations, on=area_code)

    latest['full_pct'] = 100 * latest['full'] / latest[population]
    latest['partial_pct'] = 100 * latest['partial'] / latest[population]
    latest['none_pct'] = 100 - latest['full_pct'] - latest['partial_pct']

    pie_data = latest.set_index(area_name).sort_index()
    pie_data = pie_data[['full_pct', 'partial_pct', 'none_pct']].transpose()
    pct_total = 100 * (
        data.pivot_table(values='any', index=[date_col], columns=area_name).fillna(0)
        / total_population
    )

    # plotting
    england_col = 0
    ni_col = 6
    scotland_col = 2
    wales_col = 3
    colors = [plt.cm.tab10(i) for i in [england_col, ni_col, scotland_col, wales_col]]

    fig = plt.figure(figsize=(16, 8.5), dpi=100)
    fig.set_facecolor('white')
    fig.suptitle(f'COVID-19 Vaccination Progress in the UK as of {max_date:%d %b %Y}', fontsize=14)

    gs = GridSpec(3, 4, height_ratios=[1, 1, 1])
    gs.update(top=0.95, bottom=0.15, right=0.95, left=0.02, wspace=0, hspace=0.2)

    for x, nation in enumerate(pie_data):
        ax = plt.subplot(gs[0, x])
        ax.add_patch(plt.Circle((0, 0), radius=1, color='k', fill=False))
        pie_data.plot(ax=ax, y=nation,
                      kind='pie', labels=None, legend=False, startangle=-90, counterclock=False,
                      colors=['green', 'lightgreen', 'white'],
                      )
        pct = pie_data[nation].loc['full_pct'] + pie_data[nation].loc['partial_pct']
        ax.text(0, 0.5, f"{pct:.1f}%", ha='center', va='top', weight='bold', fontsize=14)

    ax = plt.subplot(gs[1, :])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.grid(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}%'))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('Percentage of UK population partially or fully vaccinated')

    labels = [f"{nation}: {latest[latest[area_name] == nation]['any'].item():,.0f} people" for
              nation in pct_total.columns]
    ax.stackplot(pct_total.index, pct_total.values.transpose(), colors=colors, labels=labels)
    ax.legend(loc='upper left')

    ax = plt.subplot(gs[2, :], sharex=ax)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))
    ax.set_xlim(first_vaccination, max_date)
    ax.set_xticks(all_data.index.get_level_values(0).unique(), minor=True)
    major_ticks = list(weekly[date_col].unique())
    if max_date > major_ticks[-1]+np.timedelta64(1,'D'):
        major_ticks.append(max_date.to_datetime64())
    ax.set_xticks(major_ticks)
    ax.xaxis.set_major_formatter(DateFormatter('%d %b %y'))
    ax.xaxis.label.set_visible(False)
    ax.set_title('Total injections per week')

    bottom = None
    for nation_name, color in zip(pct_total, colors):
        nation_data = data[data[area_name] == nation_name].iloc[1:].set_index(date_col)
        if bottom is None:
            bottom = pd.Series(0, nation_data.index)
        heights = (nation_data['first_dose'] + nation_data['second_dose']) * (
                    7 / nation_data['duration'].dt.days)
        ax.bar(
            nation_data.index - nation_data['duration'],
            bottom=bottom,
            height=heights,
            width=nation_data['duration'].dt.days,
            align='edge',
            color=color,
        )
        bottom += heights

    fig.text(0.5, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
            ha='center')

    # return latest data so it gets displayed
    plt.savefig(repo_path / f'vaccination.png', bbox_inches='tight')
    return latest
