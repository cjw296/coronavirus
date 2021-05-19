from dataclasses import dataclass, replace
import constants as c


@dataclass
class Series:
    metric: str
    label: str = None
    title: str = None
    color: str = 'black'
    cmap: str = 'viridis'
    file_prefix: str = None

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


SUM_TEMPLATE = '7 day rolling sum of {title}'
RATE_TEMPLATE = '7 day rolling average of {title} per 100,000 people'


def derived(base: Series, metric, title):
    return replace(base, metric=metric, title=title.format(title=base.title))


def sum_of(base: Series):
    return derived(base, base.metric+'RollingSum', SUM_TEMPLATE)


def rate_of(base: Series):
    return derived(base, base.metric+'RollingRate', RATE_TEMPLATE)


unique_people_tested_sum = Series(
    metric=c.unique_people_tested_sum,
    title=SUM_TEMPLATE.format(title='unique people tested by specimen date'),
    label='tested',
    color='darkgreen',
    cmap='Greens',
)


partially_vaccinated = Series(
    metric=c.first_dose_vaccinated_cum,
    title='people at least partially vaccinated',
    label='vaccinated',
    color='darkgreen',
    cmap='Greens',
)


new_virus_tests = Series(
    metric=c.new_virus_tests,
    title='tests performed by specimen date',
    label='tests',
    color='purple',
)

new_virus_tests_sum = sum_of(new_virus_tests)

unique_cases_positivity_sum = Series(
    metric=c.unique_cases_positivity_sum,
    title='unique case positivity by specimen date',
    label='positivity',
    color='darkorange',
    cmap='Oranges',
)


new_cases = Series(
    metric=c.new_cases_by_specimen_date,
    title='new cases by specimen date',
    label='cases',
    color='red',
    cmap='Reds',
)
new_cases_sum = sum_of(new_cases)
new_cases_rate = rate_of(new_cases)


new_admissions = Series(
    metric=c.new_admissions,
    title='new hospital admissions',
    label='hospitalised',
    color='darkblue',
    cmap='Blues',
)
new_admissions_sum = sum_of(new_admissions)


new_deaths = Series(
    metric=c.new_deaths_by_death_date,
    title='new deaths within 28 days of a positive test',
    label='died',
    color='black',
    cmap='Greys',
)
new_deaths_sum = sum_of(new_deaths)

first_dose_publish_new = Series(
    metric=c.first_dose_publish_new,
    label='First Dose (New)',
)

second_dose_publish_new = Series(
    metric=c.second_dose_publish_new,
    label='Second Dose (New)',
)

complete_dose_publish_new = Series(
    metric=c.complete_dose_publish_new,
    label='Complete (New)',
)

first_dose_publish_cum = Series(
    metric=c.first_dose_publish_cum,
    label='First Dose (Total)'
)

second_dose_publish_cum = Series(
    metric=c.second_dose_publish_cum,
    label='Second Dose (Total)',
)

complete_dose_publish_cum = Series(
    metric=c.complete_dose_publish_cum,
    label='Complete (Total)',
)
