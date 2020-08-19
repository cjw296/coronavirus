import requests


def download(url, path):
    response = requests.get(url)
    assert response.status_code==200
    with path.open('wb') as target:
        target.write(response.content)
