from datetime import timedelta
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter, FuncFormatter, MaxNLocator, FixedLocator

from constants import date_col, area_type, area_code, area_name, complete_dose_daily_cum, \
    first_dose_weekly, second_dose_weekly, first_vaccination, first_dose_daily_cum, \
    second_dose_daily_cum, population, repo_path, second_dose_daily_new, \
    complete_dose_daily_new
from download import find_latest
from phe import read_csv, load_population, current_and_previous_data
from plotting import nation_tab10_cm_indices
from series import Series

day = pd.to_timedelta(1, 'D')

first_dose = 'first_dose'
second_dose = 'second_dose'

any_cov = 'any'
full_cov = 'full'
partial_cov = 'partial'


def raw_vaccination_data(dt='*', sanity_checks: bool = True):
    if dt == '*':
        dt = '????-*'
    else:
        dt = pd.to_datetime(dt).date()
    data_path, data_date = find_latest(f'vaccination_{dt}.csv')
    raw = read_csv(data_path)
    raw.sort_values([date_col, area_code], inplace=True)

    if sanity_checks:
        complete = raw[[complete_dose_daily_cum, second_dose_daily_cum,
                        complete_dose_daily_new, second_dose_daily_new]].dropna(how='any')
        cum_equal = (complete[complete_dose_daily_cum] == complete[second_dose_daily_cum]).all()
        new_equal = ((complete[complete_dose_daily_new] == complete[second_dose_daily_new]).all())
        assert raw[complete_dose_daily_cum].isnull().all() or (cum_equal and new_equal)

    return raw, data_date


def initial_data(raw):
    initial = pd.DataFrame(
        {date_col: pd.to_datetime(first_vaccination), any_cov: 0, full_cov: 0},
        index=raw[area_code].unique()
    )
    initial.index.name = area_code
    return initial.reset_index().set_index([date_col, area_code])


def weekly_data():
    raw_path, _ = find_latest('vaccination_old_style_2021-04-08.csv')
    raw = read_csv(raw_path)
    raw.sort_values([date_col, area_code], inplace=True)
    weekly = raw[[date_col, area_code, first_dose_weekly, second_dose_weekly]].dropna()
    weekly.rename(
        columns={first_dose_weekly: any_cov, second_dose_weekly: full_cov},
        errors='raise', inplace=True
    )
    return weekly.set_index([date_col, area_code]).groupby(level=-1).cumsum()


def daily_data(raw):
    daily = raw[[
        date_col, area_code, first_dose_daily_cum, second_dose_daily_cum
    ]].dropna().set_index([date_col, area_code])
    daily.rename(
        columns={first_dose_daily_cum: any_cov, second_dose_daily_cum: full_cov},
        errors='raise', inplace=True
    )
    return daily


def combined_data(raw):
    daily = daily_data(raw)
    weekly = pd.concat([initial_data(raw), weekly_data()])
    weekly_grouped = weekly.reset_index(area_code).groupby(area_code)
    interpolated = weekly_grouped[[any_cov, full_cov]].resample('D').interpolate()
    sorted_weekly = interpolated.reset_index().set_index([date_col, area_code]).sort_index()
    return pd.concat([
        sorted_weekly.loc[:daily.index.min()[0] - day],
        daily
    ])


def fix_mistakes(data):
    # If we see drops, then use the latest figure and roll it backwards
    return data.sort_index(ascending=False).groupby(area_code).cummin().sort_index()


def with_derived(data):
    data = data.copy()
    data[first_dose] = data.groupby(level=-1)[any_cov].diff().fillna(0)
    data[second_dose] = data.groupby(level=-1)[full_cov].diff().fillna(0)
    data[partial_cov] = data[any_cov] - data[full_cov]
    return data


def vaccination_dashboard(savefig=True, show_partial=True, dt='*'):
    # input data:
    raw, data_date = raw_vaccination_data(dt)
    names_frame = raw[[area_code, area_name]].drop_duplicates()
    nation_codes = names_frame[area_code]
    nation_populations = load_population().loc[nation_codes]
    total_population = nation_populations.sum()[0]

    all_data = with_derived(fix_mistakes(combined_data(raw)))
    data = pd.merge(all_data.reset_index(), names_frame, on=area_code)
    max_date = data[date_col].max()

    # data for plotting:
    is_latest_date = data[date_col] == max_date
    latest = data[[area_name, area_code, 'full', 'any', 'partial']][is_latest_date].copy()
    latest = pd.merge(latest, nation_populations, on=area_code)

    latest['full_pct'] = 100 * latest['full'] / latest[population]
    latest['partial_pct'] = 100 * latest['partial'] / latest[population]
    latest['none_pct'] = 100 - latest['full_pct'] - latest['partial_pct']

    pie_data = latest.set_index(area_name).sort_index()
    pie_data = pie_data[['full_pct', 'partial_pct', 'none_pct']].transpose()
    totals = data.pivot_table(values=['partial', 'full'], index=[date_col],
                              columns=area_name).fillna(0)
    totals = totals.swaplevel(axis='columns').sort_index(axis='columns')

    # plotting
    colors = list(chain(*((plt.cm.tab20(i * 2), plt.cm.tab20(i * 2 + (1 if show_partial else 0)))
                          for i in nation_tab10_cm_indices)))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.set_facecolor('white')
    fig.suptitle(
        f'COVID-19 Vaccination Progress in the Total UK Population as of {max_date:%d %b %Y}',
        fontsize=14
    )

    gs = GridSpec(3, 4, height_ratios=[0.8, 1, 1])
    gs.update(top=0.95, bottom=0.15, right=0.95, left=0.02, wspace=0, hspace=0.2)

    # pie charts:
    for x, nation in enumerate(pie_data):
        ax: Axes = plt.subplot(gs[0, x])
        ax.add_patch(plt.Circle((0, 0), radius=1, color='k', fill=False))
        pie_data.plot(ax=ax, y=nation,
                      kind='pie', labels=None, legend=False, startangle=-90, counterclock=False,
                      colors=['green', 'lightgreen', 'white'],
                      )
        pct = pie_data[nation].loc['full_pct'] + pie_data[nation].loc['partial_pct']
        ax.text(0, 0.5, f"{pct:.1f}%", ha='center', va='top', weight='bold', fontsize=14)

    # stack plot for cumulative
    ax: Axes = plt.subplot(gs[1, :])
    ax.yaxis.grid(zorder=-10)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('UK total population partially or fully vaccinated')

    labels = []
    for nation, level in totals.columns:
        if level == 'full':
            labels.append(f"{nation}: {totals[nation].iloc[-1].sum():,.0f} people")
        else:
            labels.append(None)
    ax.stackplot(totals.index, totals.values.transpose(), colors=colors, labels=labels, zorder=10)
    ax.legend(loc='upper left', framealpha=1)

    # make sure the current highest always has a tick:
    current = latest['any'].sum()
    ticks = [t for t in ax.get_yticks() if t <= current]
    if current < ticks[-1] + (ticks[-1] - ticks[-2]) / 2:
        ticks.pop()
    ticks.append(current)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))

    pct = ax.twinx()
    pct.set_ylim(*(l / total_population for l in ax.get_ylim()))
    pct.yaxis.set_major_locator(FixedLocator([t / total_population for t in ax.get_yticks()]))
    pct.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y * 100:,.0f}%"))

    # bar charts for rates
    ax: Axes = plt.subplot(gs[2, :], sharex=ax)
    ax.yaxis.grid(zorder=-10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[10], prune='lower'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y * 7 / 1_000_000:.1f}m"))
    ax.set_ylabel('weekly')
    ax.set_xlim(first_vaccination, max_date)
    ax.set_xticks(all_data.index.levels[0], minor=True)
    ax.xaxis.set_major_formatter(DateFormatter('%d %b'))
    ax.xaxis.label.set_visible(False)
    ax.set_title('Rate of injections by publish date')

    bottom = None
    for (nation_name, level), color in zip(totals, colors):
        nation_data = data[data[area_name] == nation_name].iloc[1:].set_index(date_col)
        if bottom is None:
            bottom = pd.Series(0, nation_data.index)
        nation_level_data = nation_data['first_dose' if level == 'partial' else 'second_dose']
        heights = nation_level_data
        ax.bar(
            nation_data.index-day,
            bottom=bottom,
            height=heights,
            width=day,
            align='edge',
            color=color,
            zorder=10,
        )
        bottom += heights

    window_days = 7
    daily_totals = all_data[[first_dose, second_dose]].groupby(date_col).sum()
    daily_average = (daily_totals[first_dose]+daily_totals[second_dose]).rolling(window_days).mean()
    lines = ax.plot(daily_average.index, daily_average, color='black', zorder=1000, )

    latest_average = daily_average.iloc[-1]
    ax.legend(lines, [f'Latest {window_days} day average: '
                      f'{7*latest_average/1_000_000:0.1f}m per week, '
                      f'{latest_average/1_000:.0f}k per day'])

    daily = ax.twinx()
    daily.set_ylim(*(l for l in ax.get_ylim()))
    daily.yaxis.set_major_locator(FixedLocator([t for t in ax.get_yticks()]))
    daily.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))
    daily.set_ylabel('daily')

    fig.text(0.5, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
             ha='center')

    # return latest data so it gets displayed
    if savefig:
        plt.savefig(repo_path / f'vaccination.png', bbox_inches='tight')
    return latest


def process(raw, index=None):
    filtered = raw.drop(columns=[area_type, area_code]).set_index(
        [area_name, date_col]).sort_index()
    if index is not None:
        filtered = filtered.reindex(index)
    return filtered.fillna(0)


def vaccination_changes(dt='*', exclude_okay=False):
    raw_data = partial(raw_vaccination_data, sanity_checks=False)
    result = current_and_previous_data(raw_data, start=dt)
    raw2, current_date, raw1, previous_date = result
    processed2 = process(raw2)
    diff = (processed2 - process(raw1, processed2.index)).fillna(0)
    for type_, okay_date in (
            ('publish', previous_date),
    ):

        type_diff = diff.filter(like=f'By{type_.capitalize()}Date').copy()

        if exclude_okay:
            okay_date = pd.to_datetime(okay_date)
            idx = type_diff.index.get_level_values(date_col)
            type_diff.loc[idx == okay_date] = type_diff.loc[idx == okay_date].clip(upper=0)

        type_diff = type_diff.loc[(type_diff != 0).any(axis=1), (type_diff != 0).any(axis=0)]

        if not type_diff.empty:
            type_diff.rename(inplace=True, columns=Series.column_names())
            type_diff.index.names = ['', '']
            type_diff.index = type_diff.index.set_levels(
                [type_diff.index.levels[0], type_diff.index.levels[1].strftime('%d %b %y')]
            )
            styled = type_diff.style
            styled.set_caption(
                f'Changes to "by {type_} date" data between '
                f'{previous_date:%d %b %Y} and {current_date:%d %b %Y}'
            )
            styled.set_table_styles([
                {'selector': 'caption', 'props': [
                    ("text-align", "center"),
                    ("font-size", "100%"),
                    ("color", 'darkred'),
                ]},
                {'selector': 'th.row_heading.level0', 'props': [
                    ("vertical-align", "top"),
                    ("background-color", "white"),
                ]},
            ])
            styled.applymap(lambda v: f'background-color: red' if v < 0 else '')
            styled.format("{:+,.0f}")

            def lighten(o):
                if o.name[-1] == okay_date.strftime('%d %b %y'):
                    return ['color: lightgreen' if v > 0 else '' for v in o]
                else:
                    return [''] * o.shape[0]

            styled.apply(lighten, axis='columns')
            styled.applymap(
                lambda v: f'color: lightgrey; background-color: inherit' if v == 0 else ''
            )
            return styled
