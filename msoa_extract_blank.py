from argparse import ArgumentParser
from csv import DictWriter
from pathlib import Path

from constants import base_path
from msoa_composite import tqdm_dict_reader


PREFIX = 'regionCode,regionName,UtlaCode,UtlaName,LtlaCode,LtlaName,areaType,areaCode,areaName'
FIELDS = PREFIX.split(',')


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', default=base_path / 'msoa_2021-01-01.csv', type=Path)
    parser.add_argument('--target', default=base_path / 'msoa_template.csv', type=Path)
    args = parser.parse_args()

    msoa_rows = {}
    for row in tqdm_dict_reader(args.source):
        area_code = row['areaCode']
        if area_code not in msoa_rows:
            msoa_rows[area_code] = {name: row[name] for name in FIELDS}

    assert len(msoa_rows) == 6791

    with open(args.target, 'w') as target:
        writer = DictWriter(target, FIELDS)
        writer.writeheader()
        for _, row in sorted(msoa_rows.items()):
            writer.writerow(row)


if __name__ == '__main__':
    main()
