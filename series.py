from dataclasses import dataclass
import constants as c


@dataclass
class Series:
    metric: str
    label: str = None
    title: str = None
    color: str = 'black'

    _lookup = {}

    def __post_init__(self):
        if self.label is None:
            self.label = self.metric
        if self.title is None:
            self.title = self.label
        self._lookup[self.metric] = self

    @classmethod
    def lookup(cls, metric):
        series = cls._lookup.get(metric)
        if series is None:
            return Series(metric)
        return series

    @classmethod
    def column_names(cls):
        return {series.metric: series.label for series in cls._lookup.values()}


new_cases = Series(
    metric=c.new_cases_by_specimen_date,
    label='cases',
    title='new cases by specimen date',
    color='red',
)


new_cases_sum = Series(
    metric=c.new_cases_sum,
    label='cases',
    color='red',
)


new_cases_rate = Series(
    metric=c.new_cases_rate,
    label='cases',
    title='7 day rolling average of new cases by specimen date per 100,000 people',
    color='red',
)


new_admissions_sum = Series(
    metric=c.new_admissions_sum,
    label='hospitalised',
    color='darkblue',
)


new_deaths_sum = Series(
    metric=c.new_deaths_sum,
    label='died',
    color='black',
)


unique_people_tested_sum = Series(
    metric=c.unique_people_tested_sum,
    label='tested',
    color='darkgreen',
)

first_dose_weekly = Series(
    metric=c.first_dose_weekly,
    label='First Dose (New)',
)

second_dose_weekly = Series(
    metric=c.second_dose_weekly,
    label='Second Dose (New)',
)

first_dose_daily_new = Series(
    metric=c.first_dose_daily_new,
    label='First Dose (New)',
)

second_dose_daily_new = Series(
    metric=c.second_dose_daily_new,
    label='Second Dose (New)',
)

complete_dose_daily_new = Series(
    metric=c.complete_dose_daily_new,
    label='Complete (New)',
)

first_dose_weekly_cum = Series(
    metric=c.first_dose_weekly_cum,
    label='First Dose (Total)',
)

second_dose_weekly_cum = Series(
    metric=c.second_dose_weekly_cum,
    label='Second Dose (Total)',
)

first_dose_daily_cum = Series(
    metric=c.first_dose_daily_cum,
    label='First Dose (Total)'
)

second_dose_daily_cum = Series(
    metric=c.second_dose_daily_new,
    label='Second Dose (Total)',
)

complete_dose_daily_cum = Series(
    metric=c.complete_dose_daily_cum,
    label='Complete (Total)',
)
