from dateutil.parser import parse as parse_date
import requests

from constants import base_path


def download(url, path):
    response = requests.get(url)
    assert response.status_code==200
    with path.open('wb') as target:
        target.write(response.content)


def find_latest(glob, date_index=-1):
    path = sorted(base_path.glob(glob), reverse=True)[0]
    dt = parse_date(str(path.stem).rsplit('_')[date_index]).date()
    return path, dt
