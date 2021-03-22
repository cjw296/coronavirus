from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DAYS_PER_MONTH

from constants import date_col, in_hospital
from download import find_latest
from phe import read_csv
from plotting import per1k_formatter
from zoe import find_previous as load_zoe


def load_prevalence(source, dates, data_date):
    data_path, data_date = find_latest(f'{source}_*.csv', on_or_before=data_date)
    data = pd.read_csv(data_path, parse_dates=[dates], index_col=[dates])
    return data.sort_index(), data_date


def load_data(data_date=None):
    if data_date:
        data_date = pd.to_datetime(data_date)
    react, react_date = load_prevalence('react_england', 'mid', data_date)
    ons_daily, ons_date = load_prevalence('ons_daily_england', 'Date', data_date)
    ons_weekly, _ = load_prevalence('ons_weekly_england', 'mid', data_date)
    ons_weekly = ons_weekly.loc[:ons_daily.index.min()]
    ons = pd.concat([ons_weekly, ons_daily])
    zoe_date, zoe = load_zoe((data_date or pd.to_datetime('now')) + pd.to_timedelta(1, 'D'),
                             print_path=False)

    hospital_data_path, _ = find_latest('england_*.csv')
    hospital = read_csv(hospital_data_path, index_col=date_col)[in_hospital].dropna()
    if data_date:
        hospital = hospital.loc[:data_date]
    hospital_date = hospital.index.max()

    return (react, ons, zoe, hospital), max(react_date, ons_date, zoe_date, hospital_date)


def plot(ax, data, column, label, with_errors=True):
    ax.plot(data.index, data[column], label=label, linestyle='dotted')
    if with_errors:
        ax.fill_between(data.index, data[f'{column}-lower-95'], data[f'{column}-upper-95'],
                        alpha=0.2,
                        label=f'{label} 95% Confidence Interval')


def plot_prevalence(data_date=None, *,
                    earliest=None, latest=None,
                    max_cases=None, max_hospital=None,
                    image_path=None, dpi=150):

    (react, ons, zoe, hospital), as_of = load_data(data_date)

    fig, (axc, axh) = plt.subplots(2, 1, figsize=(16, 9), dpi=dpi, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.set_facecolor('white')

    latest_hospital_date = hospital.index.max()
    latest_hospital_value = hospital.loc[hospital.index.max()]
    axh.plot(hospital.index, hospital)
    axh.set_title('Patients in hospital with COVID-19: '
                  f'{latest_hospital_value:,.0f} on {latest_hospital_date:%d %b %Y}')
    axh.set_ylim((0, max_hospital))

    plot(axc, react, 'people', label='REACT')
    plot(axc, ons, 'number', label='ONS')
    plot(axc, zoe, 'corrected_covid_positive', label='ZOE*', with_errors=False)

    axc.legend(loc='upper left', framealpha=1)
    axc.set_title(f'Modelled number of people with COVID-19 in England as of {as_of:%d %b %Y}')
    axc.set_ylim((0, max_cases))

    for ax in axh, axc:
        yaxis = ax.yaxis
        yaxis.tick_right()
        yaxis.set_label_position("right")
        yaxis.set_major_formatter(per1k_formatter)

        xaxis = ax.xaxis
        xaxis.label.set_visible(False)
        xaxis.set_tick_params(labelbottom=True)
        formatter = xaxis.get_major_formatter()
        formatter.scaled[1] = '%d %b'
        formatter.scaled[DAYS_PER_MONTH] = '%b %y'

        ax.margins(x=0.01)

        if latest:
            latest = pd.to_datetime(latest)
        else:
            latest = as_of
        ax.set_xlim((ons.index.min() if earliest is None else pd.to_datetime(earliest),
                     latest + timedelta(days=1)))

    fig.text(0.1, 0.04,
             '* ZOE is a UK-wide study. '
             '\nPlots by @chriswithers13, data from gov.uk and the various studies.',
             color='darkgrey')

    if image_path:
        plt.savefig(image_path / f'{data_date.date()}.png', bbox_inches='tight', dpi=dpi)
        plt.close()