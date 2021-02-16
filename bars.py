import re
from dataclasses import dataclass, replace
from datetime import timedelta, date
from typing import List, Union, Optional

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from constants import (
    unique_people_tested_sum, cases, national_lockdowns, ltla, my_areas,
    oxford_areas, london_areas, region, new_cases_by_specimen_date, area_name, date_col, nation,
    scotland, northern_ireland, wales, area_code, population, new_deaths_by_death_date,
    new_admissions, overview, england
)
from phe import best_data, current_and_previous_data, load_population
from plotting import stacked_bar_plot
from series import Series


def plot_diff(ax, for_date, data, previous_date, previous_data,
              diff_ylims=None, diff_log_scale=None, earliest=None, colormap='viridis'):
    diff = data.sub(previous_data, fill_value=0)
    total_diff = diff.sum().sum()
    stacked_bar_plot(ax, diff, colormap)
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
        ax, data, label, average_days, average_end, title, testing_data,
        ylim, tested_ylim, earliest, colormap='viridis',
        legend_loc='upper left', legend_ncol=1
):

    handles = stacked_bar_plot(ax, data, colormap)

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

    if testing_data is not None:
        tested_ax = legend_ax = ax.twinx()
        tested_label = '% population tested'
        if average_end is not None:
            testing_data = testing_data[:average_end]
        tested_color = 'darkblue'
        handles.extend(
            tested_ax.plot(testing_data.index, testing_data, color=tested_color,
                           label=tested_label, linestyle='dotted')
        )
        tested_ax.set_ylabel(f'{tested_label} in preceding 7 days',
                             rotation=-90, labelpad=14)
        tested_ax.set_ylim(0, tested_ylim)
        tested_ax.yaxis.tick_left()
        tested_ax.yaxis.set_label_position("left")
        tested_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}%'))
    else:
        legend_ax = ax

    fix_x_axis(ax, data, earliest)

    ax.set_ylabel(label)
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

    legend_ax.legend(handles=handles, loc=legend_loc, framealpha=1, ncol=legend_ncol)

    if title:
        ax.set_title(title)


def plot_with_diff(
        data_date,
        config: Union['Bars', str] = None,
        image_path=None,
        **overrides
):
    config = Bars.get(config, **overrides)
    results = current_and_previous_data(config.data_for, data_date, config.diff_days)
    data, data_date, previous_data, previous_data_date = results

    if config.uncertain_days is None or not config.average_days:
        average_end = None
    else:
        average_end = data.index.max()-timedelta(days=config.uncertain_days)

    end_dates = [previous_data.index.max(), data.index.max()]
    if config.to_date:
        end_dates.append(config.to_date)

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
            diff_ax, data_date, data, previous_data_date, previous_data,
            config.diff_ylims, config.diff_log_scale, config.earliest, config.colormap
        )
        plot_stacked_bars(
            bars_ax, data, config.series.title,
            config.average_days, average_end, config.title,
            config.testing_data_for(data_date), config.ylim, config.tested_ylim,
            config.earliest, config.colormap, config.legend_loc, config.legend_ncol
        )

    if image_path:
        plt.savefig(image_path / f'{data_date}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


@dataclass
class Bars:
    metric: str = new_cases_by_specimen_date
    columns_from: str = area_name
    uncertain_days: int = 5
    average_days: Optional[int] = 7
    show_testing: bool = True
    diff_days: int = 1
    diff_ylims: List[float] = None
    diff_log_scale: bool = False
    ylim: float = None
    tested_ylim: float = None
    earliest: Union[str, date, pd.Timestamp] = '2020-10-01'
    area_type: str = ltla
    areas: List[str] = None
    colormap: str = 'viridis'
    to_date: Union[date, pd.Timestamp] = None
    title_template: str = 'Evolution of PHE {config.series.label} reporting'
    show_title: bool = False
    legend_loc: str = 'upper left'
    legend_ncol: int = 1
    data_is_cumulative: bool = False

    @classmethod
    def get(cls, name_or_instance: Union['Bars', str] = None, **overrides):
        if isinstance(name_or_instance, str):
            bars = BARS[name_or_instance]
        elif name_or_instance:
            bars = name_or_instance
        else:
            return cls(**overrides)
        if overrides:
            return replace(bars, **overrides)
        return bars

    @property
    def series(self):
        return Series.lookup(self.metric)

    @property
    def title(self):
        if self.show_title:
            return self.title_template.format(config=self)

    @property
    def earliest_data(self):
        if self.earliest is None:
            return None
        else:
            return pd.to_datetime(self.earliest) - timedelta(days=self.average_days or 0)

    @property
    def data_file_stem(self):
        return self.area_type

    def data_for(self, dt):
        data, data_date = best_data(dt, self.data_file_stem, self.areas, self.earliest_data,
                                    metric=self.metric)
        data = data.pivot_table(
            values=self.metric, index=[date_col], columns=self.columns_from
        ).fillna(0)
        if self.data_is_cumulative:
            data = data.diff().iloc[1:]
        return data, data_date

    def testing_data_for(self, dt):
        if self.show_testing:
            data = best_data(dt, self.area_type, self.areas, self.earliest_data)[0]
            if unique_people_tested_sum in data:
                data = data.merge(load_population(), on=area_code, how='left')
                agg = data.groupby(date_col).agg(
                    {unique_people_tested_sum: 'sum', population: 'sum'}
                )
                return 100 * agg[unique_people_tested_sum] / agg[population]
            else:
                return pd.Series(0, index=[pd.to_datetime(dt)])


@dataclass()
class DemographicBars(Bars):

    columns_from: str = 'age'
    bands: List[str] = None
    show_testing: bool = False
    reverse_bands: bool = False

    all_detail = ['00_04', '05_09', '10_14', '15_19', '20_24', '25_29', '30_34',
                  '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69',
                  '70_74', '75_79', '80_84', '85_89', '90+']

    split_60 = ['00_59', '60+']

    detail_above_60 = ['00_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85_89', '90+']

    @property
    def data_file_stem(self):
        return f'{self.metric[:-1]}_demographics_{self.area_type}'

    def data_for(self, dt):
        data, data_date = super().data_for(dt)

        def tidy(name):
            return re.sub(r'0(\d)', r'\1', name).replace('_', '-')

        bands = self.bands or self.all_detail
        if self.reverse_bands:
            bands = reversed(bands)
        columns = list(bands)
        return data[columns].rename(columns=tidy), data_date


death_demographics = DemographicBars(
    'deaths',
    area_type=nation,
    areas=[england],
    title_template='Evolution of COVID-10 {config.series.title} in England by age',
    legend_loc='upper center',
    legend_ncol=2,
    uncertain_days=16,
    diff_log_scale=True,
    diff_ylims=[-10, 1000],
)


BARS = dict(
    cases_my_areas=Bars(
        diff_ylims=[-2, 350],
        areas=my_areas,
    ),
    cases_oxford=Bars(
        areas=oxford_areas,
    ),
    cases_wiltshire=Bars(
        areas=['E06000054'],

    ),
    cases_london=Bars(
        diff_ylims=[-10, 1800],
        areas=london_areas,
    ),
    cases_regions=Bars(
        diff_ylims=[-100, 25_000],
        area_type=region,
    ),
    cases_devolved=Bars(
        area_type=nation,
        areas=[scotland, northern_ireland, wales],
        diff_ylims=[-100, 3_000],
        show_testing=False
    ),
    cases_demographics=DemographicBars(
        'cases',
        area_type=overview,
        title_template='Evolution of COVID-10 {config.series.title} in the UK by age',
        legend_ncol=2,
        diff_log_scale=True,
        uncertain_days=0,
        diff_ylims=[-50, 80_000],
    ),
    admissions_nations=Bars(
        metric=new_admissions,
        title_template='Evolution of PHE new hospital admissions reporting',
        colormap='summer',
        area_type=nation,
        show_testing=False,
        diff_ylims=[-100, 3_500],
        legend_loc='upper center',
    ),
    deaths_regions=Bars(
        metric=new_deaths_by_death_date,
        title_template='Evolution of PHE deaths reporting in England',
        colormap='cividis',
        area_type=region,
        show_testing=False,
        diff_ylims=[-10, 300],
        legend_loc='upper center',
        uncertain_days=21,
    ),
    deaths_demographics_for_comparison=death_demographics,
    deaths_demographics=replace(
        death_demographics,
        colormap='cividis',
        bands=DemographicBars.detail_above_60,
        reverse_bands=True,
        legend_loc='upper center',
        uncertain_days=16,
        diff_log_scale=True,
        diff_ylims=[-10, 1000],
    ),
)
