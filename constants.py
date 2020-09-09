from datetime import date
from pathlib import Path

nation = ['Nation', 'nation']
region = ['Region', 'region']
ltla = ['Lower tier local authority', 'ltla']
utla = ['Upper tier local authority', 'utla']

cases_url = 'https://coronavirus.data.gov.uk/downloads/{data_type}/coronavirus-cases_latest.{data_type}'

base_path = Path('~/Downloads').expanduser()
base = str(base_path)
area = 'Area name'
code = 'Area code'
specimen_date = 'Specimen date'
cases = 'Daily lab-confirmed cases'
people_tested = 'Daily number of people tested'

lockdown = date(2020, 3, 23)
testing = date(2020, 4, 30)
relax_1 = date(2020, 5, 11)
relax_2 = date(2020, 7, 4)

phe_vmax = 0.1

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
oxfordshire = 'E10000025'

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
