from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter, FuncFormatter, MaxNLocator, FixedLocator

from constants import date_col, area_type, area_code, area_name, complete_dose_daily_cum, \
    first_dose_weekly, second_dose_weekly, first_vaccination, first_dose_daily_cum, \
    second_dose_daily_cum, population, repo_path, second_dose_daily_new, \
    complete_dose_daily_new
from download import find_latest
from phe import read_csv, load_population, current_and_previous_data
from series import Series


def raw_vaccination_data(dt='*'):
    if dt == '*':
        dt = '????-*'
    else:
        dt = pd.to_datetime(dt).date()
    new_weekly_path, new_weekly_dt = find_latest(f'vaccination_{dt}.csv', date_index=-1)
    cum_path, cum_dt = find_latest(f'vaccination_cum_{dt}.csv', date_index=-1)
    assert cum_dt == new_weekly_dt, f'{cum_dt} != {new_weekly_dt}'
    new_weekly_df = read_csv(new_weekly_path)
    cum_df = read_csv(cum_path)
    raw = pd.merge(new_weekly_df, cum_df, how='outer',
                   on=[date_col, area_type, area_code, area_name])
    raw.sort_values([date_col, area_code], inplace=True)
    complete = raw[[complete_dose_daily_cum, second_dose_daily_cum,
                    complete_dose_daily_new, second_dose_daily_new]].dropna(how='any')
    cum_equal = (complete[complete_dose_daily_cum] == complete[second_dose_daily_cum]).all()
    new_equal = ((complete[complete_dose_daily_new] == complete[second_dose_daily_new]).all())
    assert raw[complete_dose_daily_cum].isnull().all() or (cum_equal and new_equal)

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
    initial = weekly.groupby(area_code).agg({
        'first_dose': 'sum', 'second_dose': 'sum', date_col: 'max'
    }).reset_index().set_index([date_col, area_code])
    weekly_date = initial.index.max()[0]

    daily_rows = raw[[
        date_col, area_code, first_dose_daily_cum, second_dose_daily_cum
    ]].dropna().set_index([date_col, area_code])
    daily_rows.rename(
        columns={first_dose_daily_cum: 'first_dose', second_dose_daily_cum: 'second_dose'},
        errors='raise', inplace=True
    )

    initial_date = weekly_date + timedelta(days=1)
    initial_from_daily = daily_rows.loc[initial_date]
    initial_from_weekly = initial.loc[weekly_date]
    initial_from_weekly.where(
        initial_from_daily > initial_from_weekly, initial_from_daily, inplace=True
    )

    daily = pd.concat([initial, daily_rows[initial_date:]])
    return daily.groupby(area_code).diff().dropna().reset_index()


def vaccination_dashboard():
    # input data:
    raw, data_date = raw_vaccination_data()
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
    is_latest_date = data[date_col] == max_date
    latest = data[[area_name, area_code, 'full', 'any', 'partial']][is_latest_date].copy()
    latest = pd.merge(latest, nation_populations, on=area_code)

    latest['full_pct'] = 100 * latest['full'] / latest[population]
    latest['partial_pct'] = 100 * latest['partial'] / latest[population]
    latest['none_pct'] = 100 - latest['full_pct'] - latest['partial_pct']

    pie_data = latest.set_index(area_name).sort_index()
    pie_data = pie_data[['full_pct', 'partial_pct', 'none_pct']].transpose()
    totals = data.pivot_table(values='any', index=[date_col], columns=area_name).fillna(0)

    # plotting
    england_col = 0
    ni_col = 6
    scotland_col = 2
    wales_col = 3
    colors = [plt.cm.tab10(i) for i in [england_col, ni_col, scotland_col, wales_col]]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.set_facecolor('white')
    fig.suptitle(f'COVID-19 Vaccination Progress in the UK as of {max_date:%d %b %Y}', fontsize=14)

    gs = GridSpec(3, 4, height_ratios=[0.8, 1, 1])
    gs.update(top=0.95, bottom=0.15, right=0.95, left=0.02, wspace=0, hspace=0.2)

    # pie charts:
    for x, nation in enumerate(pie_data):
        ax = plt.subplot(gs[0, x])
        ax.add_patch(plt.Circle((0, 0), radius=1, color='k', fill=False))
        pie_data.plot(ax=ax, y=nation,
                      kind='pie', labels=None, legend=False, startangle=-90, counterclock=False,
                      colors=['green', 'lightgreen', 'white'],
                      )
        pct = pie_data[nation].loc['full_pct'] + pie_data[nation].loc['partial_pct']
        ax.text(0, 0.5, f"{pct:.1f}%", ha='center', va='top', weight='bold', fontsize=14)

    # stack plot for cumulative
    ax = plt.subplot(gs[1, :])
    ax.yaxis.grid(zorder=-10)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('UK population partially or fully vaccinated')

    labels = [f"{nation}: {totals[nation].iloc[-1]:,.0f} people" for nation in totals.columns]
    ax.stackplot(totals.index, totals.values.transpose(), colors=colors, labels=labels, zorder=10)
    ax.legend(loc='upper left', framealpha=1)

    # make sure the current highest always has a tick:
    current = latest['any'].sum()
    ticks = [t for t in ax.get_yticks() if t <= current]
    ticks.append(current)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))

    pct = ax.twinx()
    pct.set_ylim(*(l/total_population for l in ax.get_ylim()))
    pct.yaxis.set_major_locator(FixedLocator([t/total_population for t in ax.get_yticks()]))
    pct.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:,.0f}%"))

    # bar charts for rates
    ax = plt.subplot(gs[2, :], sharex=ax)
    ax.yaxis.grid(zorder=-10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))
    ax.set_ylabel('weekly')
    ax.set_xlim(first_vaccination, max_date)
    ax.set_xticks(all_data.index.levels[0], minor=True)
    major_ticks = list(weekly[date_col].unique())
    if max_date > major_ticks[-1]+np.timedelta64(1,'D'):
        major_ticks.append(max_date.to_datetime64())
    ax.set_xticks(major_ticks)
    ax.xaxis.set_major_formatter(DateFormatter('%d %b %y'))
    ax.xaxis.label.set_visible(False)
    ax.axvline(weekly[date_col].max(), linestyle='dashed', color='lightgrey', zorder=-10)
    ax.set_title('Rate of injections (weeky by injection date, daily by publish date)')

    bottom = None
    for nation_name, color in zip(totals, colors):
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
            zorder=10,
        )
        bottom += heights
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[10], prune='lower'))

    daily = ax.twinx()
    daily.set_ylim(*(l/7 for l in ax.get_ylim()))
    daily.yaxis.set_major_locator(FixedLocator([t/7 for t in ax.get_yticks()]))
    daily.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m"))
    daily.set_ylabel('daily')

    fig.text(0.5, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
            ha='center')

    # return latest data so it gets displayed
    plt.savefig(repo_path / f'vaccination.png', bbox_inches='tight')
    return latest


def process(raw, index=None):
    filtered = raw.drop(columns=[area_type, area_code]).set_index(
        [area_name, date_col]).sort_index()
    if index is not None:
        filtered = filtered.reindex(index)
    return filtered.fillna(0)


def vaccination_changes(dt='*', exclude_okay=False):
    result = current_and_previous_data(raw_vaccination_data, start=dt)
    raw2, current_date, raw1, previous_date = result
    processed2 = process(raw2)
    diff = (processed2 - process(raw1, processed2.index)).fillna(0)
    for type_ in 'vaccination', 'publish':

        type_diff = diff.filter(like=f'By{type_.capitalize()}Date')
        type_diff = type_diff.loc[:, (type_diff != 0).any(axis=0)]
        type_diff = type_diff.loc[(type_diff != 0).any(axis=1), :]

        if type_ == 'publish':
            ok_date = previous_date
        else:
            ok_date = current_date - timedelta(days=4)

        if exclude_okay and not type_diff.empty:
            type_diff.drop(ok_date, level=-1, inplace=True)

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
                if o.name[-1] == ok_date.strftime('%d %b %y'):
                    color = 'color: lightgreen'
                else:
                    color = ''
                return [color] * o.shape[0]

            styled.apply(lighten, axis='columns')
            styled.applymap(
                lambda v: f'color: lightgrey; background-color: inherit' if v == 0 else ''
            )
            return styled
