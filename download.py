from dateutil.parser import parse as parse_date
import requests

from constants import base_path


def download(url, path):
    response = requests.get(url)
    assert response.status_code==200
    with path.open('wb') as target:
        target.write(response.content)


def find_all(glob, date_index=-1, earliest=None):
    possible = []
    for path in base_path.glob(glob):
        dt = parse_date(str(path.stem).rsplit('_')[date_index]).date()
        if earliest is None or dt >= earliest:
            possible.append((dt, path))
    return possible


def find_latest(glob, date_index=-1):
    possible = find_all(glob, date_index)
    if not possible:
        raise FileNotFoundError(glob)
    dt, path = sorted(possible, reverse=True)[0]
    return path, dt
