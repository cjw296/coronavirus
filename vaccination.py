from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter, FuncFormatter

from constants import date_col, area_type, area_code, area_name, complete_dose_daily_cum, \
    first_dose_weekly, second_dose_weekly, first_vaccination, first_dose_daily_cum, \
    second_dose_daily_cum, population, repo_path, first_dose_daily_new, second_dose_daily_new
from download import find_latest
from phe import read_csv, load_population, current_and_previous_data


def raw_vaccination_data(dt='*'):
    if dt == '*':
        dt = '????-*'
    new_weekly_path, new_weekly_dt = find_latest(f'vaccination_{dt}.csv', date_index=-1)
    cum_path, cum_dt = find_latest(f'vaccination_cum_{dt}.csv', date_index=-1)
    assert cum_dt == new_weekly_dt, f'{cum_dt} != {new_weekly_dt}'
    new_weekly_df = read_csv(new_weekly_path)
    cum_df = read_csv(cum_path)
    raw = pd.merge(new_weekly_df, cum_df, how='outer',
                   on=[date_col, area_type, area_code, area_name])
    raw.sort_values([date_col, area_code], inplace=True)
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
    ax.set_title('Rate of injections per week')

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

    # ax.set_ylim(0, 2_500_000)

    fig.text(0.5, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}',
            ha='center')

    # return latest data so it gets displayed
    plt.savefig(repo_path / f'vaccination.png', bbox_inches='tight')
    return latest


def vaccination_corrections():

    calc_first_dose = 'calcPeopleVaccinatedFirstDoseByPublishDate'
    calc_second_dose = 'calcPeopleVaccinatedSecondDoseByPublishDate'

    def pivoted(data):
        return data.pivot_table(
            values=[first_dose_daily_new, second_dose_daily_new,
                    first_dose_daily_cum, second_dose_daily_cum],
            index=date_col, columns=area_name
        )

    result = current_and_previous_data(raw_vaccination_data)
    all_current_data, current_date, all_previous_data, previous_date = result

    pivoted_current = pivoted(all_current_data)
    corrections = pivoted_current - pivoted(all_previous_data)

    current_cum = pivoted_current[[first_dose_daily_cum, second_dose_daily_cum]]
    calculated = current_cum - current_cum.shift(1)
    calculated.rename(
        columns={first_dose_daily_cum: calc_first_dose, second_dose_daily_cum: calc_second_dose},
        inplace=True
    )

    observed = pivoted_current[[first_dose_daily_new, second_dose_daily_new]].rename(
        columns={first_dose_daily_new: calc_first_dose, second_dose_daily_new: calc_second_dose},
    )

    corrections = pd.concat([corrections, observed-calculated], axis=1)

    corrections.dropna(inplace=True)
    corrections.columns.names = ['', '']
    corrections.columns = corrections.columns.swaplevel()
    corrections.rename(inplace=True, level=1, columns={
        first_dose_daily_cum: 'First Dose (Total)',
        first_dose_daily_new: 'First Dose (New)',
    })
    corrections.sort_index(axis=1, inplace=True)
    corrections.index.name = ''
    corrections.index = corrections.index.strftime("%d %b %Y")
    corrections = corrections.loc[:, (corrections != 0).any(axis=0)]
    corrections = corrections.loc[(corrections != 0).any(axis=1), :]
    if corrections.empty:
        return
    styled = corrections.style
    styled.set_caption(
        "Corrections between https://coronavirus.data.gov.uk/ release on "
        f"{previous_date:%d %b %Y} and {current_date:%d %b %Y}"
    )
    styled.set_table_styles([
        {'selector': 'caption', 'props': [
            ("text-align", "center"),
            ("font-size", "100%"),
            ("color", 'darkred'),
        ]},
        {'selector': 'th.col_heading.level0', 'props': [
            ("text-align", "center"),
            ("font-weight", "normal"),
            ("font-style", "italic"),
            ("padding-bottom", '0.1em')
        ]},
        {'selector': 'td', 'props': [
            ("text-align", "center")
        ]},
    ])
    styled.format("{:,.0f}")
    return styled
