from datetime import date

from dateutil.parser import parse as parse_date

from constants import (
    data_start, second_wave, lockdown2, earliest_available_download,
    earliest_testing
)

special_dates = {text: dt for dt, text in (
    (data_start, 'start'),
    (second_wave, 'second-wave'),
    (lockdown2[0], 'lockdown-2-start'),
    (lockdown2[1], 'lockdown-2-end'),
    (earliest_available_download, 'earliest-download'),
    (date(2020, 12, 18), 'earliest-msoa'),
    (earliest_testing, 'earliest-testing'),
    (date.today(), 'today'),
)}


def date_lookup(text):
    dt = special_dates.get(text)
    if dt is None:
        dt = parse_date(text).date()
    return dt


def add_date_arg(parser, name='--from-date', **kw):
    help_text = kw.pop('help', '')
    if help_text:
        help_text += ': '
    parser.add_argument(name, type=date_lookup, help=help_text+', '.join(special_dates), **kw)