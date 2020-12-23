from dataclasses import dataclass


@dataclass
class Series:
    metric: str
    label: str
    color: str


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

