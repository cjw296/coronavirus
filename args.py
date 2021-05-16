from datetime import date, timedelta

from dateutil.parser import parse as parse_date
import pandas as pd

from constants import (
    data_start, second_wave, lockdown2, earliest_available_download,
    earliest_testing, earliest_msoa, lockdown3
)

special_dates = {text: dt for dt, text in (
    (data_start, 'start'),
    (second_wave, 'second-wave'),
    (lockdown2[0], 'lockdown-2-start'),
    (lockdown2[1], 'lockdown-2-end'),
    (lockdown3[0], 'lockdown-3-start'),
    (lockdown3[1], 'lockdown-3-end'),
    (earliest_available_download, 'earliest-download'),
    (earliest_msoa, 'earliest-msoa'),
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


def add_parallel_args(parser, default_duration=1/24, default_output='mp4', from_date=True):
    if from_date:
        add_date_arg(parser, default=second_wave if isinstance(from_date, bool) else from_date)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--exclude-days', type=int)
    add_date_arg(group, '--to-date')

    parser.add_argument('--output', default=default_output)
    parser.add_argument('--ignore-errors', dest='raise_errors', action='store_false')

    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--duration', type=float, default=default_duration,
                        help='fast=0.05, slow=0.3')

    add_date_arg(parser, '--single')


def parallel_params(args, item_is_timestamp=True):
    return dict(
        duration=args.duration,
        outputs=args.output,
        raise_errors=args.raise_errors,
        max_workers=args.max_workers,
        item=pd.to_datetime(args.single) if args.single and item_is_timestamp else args.single
    )


def parallel_to_date(args, max_date: date, default_exclude=None) -> date:
    if args.to_date:
        return args.to_date
    exclude_days = default_exclude if args.exclude_days is None else default_exclude
    if exclude_days is not None:
        return max_date - timedelta(days=exclude_days)
    return max_date
