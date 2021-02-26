from datetime import date
from pathlib import Path

overview = 'overview'
nation = 'nation'
region = 'region'
utla = 'utla'
ltla = 'ltla'
msoa = 'msoa'

earliest_available_download = date(2020, 8, 12)
earliest_msoa = date(2020, 12, 18)
earliest_testing = date(2020, 12, 14)
earliest_vaccination = date(2021, 1, 12)

cases_url = 'https://coronavirus.data.gov.uk/downloads/{data_type}/coronavirus-cases_latest.{data_type}'

base_path = Path('~/coronavirus/data').expanduser()
output_path = Path('~/Downloads').expanduser()
repo_path = Path(__file__).absolute().parent


base = str(base_path)
area = 'Area name'
code = 'Area code'
specimen_date = 'Specimen date'
date_col = 'date'
cases = 'Daily lab-confirmed cases'
people_tested = 'Daily number of people tested'
per100k = 'per 100,000 people'
metric = 'metric'  # generic column name for dealing with metrics in abstract
population = 'population'
pct_population = '% of population'

area_code = 'areaCode'
area_name = 'areaName'
area_type = 'areaType'
new_cases_by_specimen_date = 'newCasesBySpecimenDate'
new_admissions = "newAdmissions"
new_deaths_by_death_date = 'newDeaths28DaysByDeathDate'
new_tests_by_publish_date = 'newTestsByPublishDate'
new_virus_tests = 'newVirusTests'
release_timestamp = 'releaseTimestamp'

new_admissions_sum = 'newAdmissionsRollingSum'
new_cases_sum = 'newCasesBySpecimenDateRollingSum'
new_deaths_sum = 'newDeaths28DaysByDeathDateRollingSum'
unique_people_tested_sum = 'uniquePeopleTestedBySpecimenDateRollingSum'
unique_cases_positivity_sum = 'uniqueCasePositivityBySpecimenDateRollingSum'
case_demographics = 'newCasesBySpecimenDateAgeDemographics'
death_demographics = 'newDeaths28DaysByDeathDateAgeDemographics'
admission_demographics = 'cumAdmissionsByAge'

new_cases_rate = 'newCasesBySpecimenDateRollingRate'
new_cases_change = 'newCasesBySpecimenDateChange'

in_hospital = 'hospitalCases'

first_dose_weekly = 'weeklyPeopleVaccinatedFirstDoseByVaccinationDate'
second_dose_weekly = 'weeklyPeopleVaccinatedSecondDoseByVaccinationDate'
first_dose_daily_new = "newPeopleVaccinatedFirstDoseByPublishDate"
second_dose_daily_new = "newPeopleVaccinatedSecondDoseByPublishDate"
complete_dose_daily_new = "newPeopleVaccinatedCompleteByPublishDate"

first_dose_weekly_cum = 'cumPeopleVaccinatedFirstDoseByVaccinationDate'
second_dose_weekly_cum = 'cumPeopleVaccinatedSecondDoseByVaccinationDate'
first_dose_daily_cum = "cumPeopleVaccinatedFirstDoseByPublishDate"
second_dose_daily_cum = "cumPeopleVaccinatedSecondDoseByPublishDate"
complete_dose_daily_cum = "cumPeopleVaccinatedCompleteByPublishDate"

england_metrics = [
    new_admissions_sum,
    new_cases_sum,
    new_deaths_sum,
    unique_people_tested_sum,
    in_hospital,
]

vaccination_new_and_weekly = [
    first_dose_daily_new,
    second_dose_daily_new,
    complete_dose_daily_new,
    first_dose_weekly,
    second_dose_weekly,
]

vaccination_cumulative = [
    first_dose_daily_cum,
    second_dose_daily_cum,
    complete_dose_daily_cum,
    first_dose_weekly_cum,
    second_dose_weekly_cum,
]

standard_metrics = [
    new_cases_by_specimen_date,
    new_deaths_by_death_date,
    unique_people_tested_sum,
    unique_cases_positivity_sum,
    new_cases_rate
]

lockdown = date(2020, 3, 23)
testing = date(2020, 4, 30)
relax_1 = date(2020, 5, 11)
relax_2 = date(2020, 7, 4)
second_wave = date(2020, 8, 1)
data_start = date(2020, 3, 15)
first_vaccination = date(2020, 12, 8)

lockdown1 = (lockdown, relax_2)
lockdown2 = (date(2020, 11, 5), date(2020, 12, 2))
lockdown3 = (date(2021, 1, 5), date(2021, 3, 8))
national_lockdowns = lockdown1, lockdown2, lockdown3


brighton = 'E06000043'
west_sussex = 'E10000032'
east_sussex = 'E10000011'
areas = [brighton, west_sussex, east_sussex]

hammersmith = 'E09000013'
kensington = 'E09000020'
ealing = 'E09000009'
hounslow = 'E09000018'
richmond = 'E09000027'
wandsworth = 'E09000032'
brent = 'E09000005'

wirral = 'E08000015'
cheshire = 'E06000050'
liverpool = 'E08000012'
sefton = 'E08000014'
knowsley = 'E08000011'
st_helens = 'E08000013'

bristol = 'E06000023'
south_gloucestershire = 'E06000025'
bath = 'E06000022'
north_somerset = 'E06000024'

hampshire = 'E10000014'
# areas = [hampshire]

bedford = 'E06000055'
central_bedfordshire = 'E06000056'
luton = 'E06000032'
hertfordshire = 'E10000015'

# areas = [bedford, central_bedfordshire, luton, hertfordshire]

northampton = 'E10000021'
milton_keynes =  'E06000042'

wokingham = 'E06000041'
reading = 'E06000038'
west_berks = 'E06000037'
bracknell = 'E06000036'

oxford = 'E07000178'
south_oxfordshire = 'E07000179'
west_oxfordshire = 'E07000181'
vale_of_white_horse = 'E07000180'
cherwell = 'E07000177'

oxford_areas = [oxford, south_oxfordshire, west_oxfordshire, vale_of_white_horse, cherwell]

dorset = 'E06000059'

southwark = 'E09000028'
lambeth = 'E09000022'
westminster = 'E09000033'
cambden = 'E09000007'
islington = 'E09000019'
hackney = 'E09000012'
tower_hamlets = 'E09000030'
lewisham = 'E09000023'
city_of_london = 'E09000001'
kingston='E09000021'
surrey='E10000030'
newham = 'E09000025'

bromley = 'E09000006'
croydon = 'E09000008'

my_areas = [wokingham, reading, west_berks, bracknell]
london_areas = [
    southwark, lambeth, wandsworth, hammersmith, kensington,
    westminster, cambden, islington, hackney, tower_hamlets, lewisham, city_of_london, newham
]

england = 'E92000001'
wales = 'W92000004'
scotland = 'S92000003'
northern_ireland = 'N92000002'
