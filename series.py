from dataclasses import dataclass


@dataclass
class Series:
    metric: str
    label: str = None
    color: str = 'black'

    _lookup = {}

    def __post_init__(self):
        if self.label is None:
            self.label = self.metric
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


new_cases_sum = Series(
    metric='newCasesBySpecimenDateRollingSum',
    label='cases',
    color='red',
)


new_admissions_sum = Series(
    metric='newAdmissionsRollingSum',
    label='hospitalised',
    color='darkblue',
)


new_deaths_sum = Series(
    metric='newDeaths28DaysByDeathDateRollingSum',
    label='died',
    color='black',
)


unique_people_tested_sum = Series(
    metric='uniquePeopleTestedBySpecimenDateRollingSum',
    label='tested',
    color='darkgreen',
)

first_dose_weekly = Series(
    metric='weeklyPeopleVaccinatedFirstDoseByVaccinationDate',
    label='First Dose (New)',
)

second_dose_weekly = Series(
    metric='weeklyPeopleVaccinatedSecondDoseByVaccinationDate',
    label='Second Dose (New)',
)

first_dose_daily_new = Series(
    metric="newPeopleVaccinatedFirstDoseByPublishDate",
    label='First Dose (New)',
)

second_dose_daily_new = Series(
    metric="newPeopleVaccinatedSecondDoseByPublishDate",
    label='Second Dose (New)',
)

complete_dose_daily_new = Series(
    metric="newPeopleVaccinatedCompleteByPublishDate",
    label='Complete (New)',
)

first_dose_weekly_cum = Series(
    metric='cumPeopleVaccinatedFirstDoseByVaccinationDate',
    label='First Dose (Total)',
)

second_dose_weekly_cum = Series(
    metric='cumPeopleVaccinatedSecondDoseByVaccinationDate',
    label='Second Dose (Total)',
)

first_dose_daily_cum = Series(
    metric="cumPeopleVaccinatedFirstDoseByPublishDate",
    label='First Dose (Total)'
)

second_dose_daily_cum = Series(
    metric="cumPeopleVaccinatedSecondDoseByPublishDate",
    label='Second Dose (Total)',
)

complete_dose_daily_cum = Series(
    metric="cumPeopleVaccinatedCompleteByPublishDate",
    label='Complete (Total)',
)
