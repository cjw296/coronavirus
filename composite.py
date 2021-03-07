import subprocess
import sys
from argparse import ArgumentParser
from datetime import date
from functools import partial
from itertools import chain
from multiprocessing import cpu_count
from typing import List, Tuple, Union, Sequence

import imageio
import pandas as pd
from matplotlib.colors import to_rgba
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.fx.margin import margin
from tqdm.auto import tqdm

from constants import (
    output_path, ltla, data_start, nhs_region, msoa, new_admissions_sum,
    new_cases_sum, new_deaths_sum, unique_people_tested_sum
)

Date = Union[date, str]


def rgb_color(name):
    return tuple(e*255 for e in to_rgba(name)[:3])


def humanize(dt):
    return dt.strftime('%d %b %y')


white = rgb_color('white')
fps = 24


def run(cmd):
    cmd = [str(c.date()) if isinstance(c, pd.Timestamp) else str(c) for c in cmd]
    print('Running: ', subprocess.list2cmdline(['python']+cmd[1:]))
    subprocess.check_call(cmd)


class Part:

    prefix = ''

    def __init__(self, name: str =None):
        self._name = name
        self.frames = {}

    @property
    def name(self):
        return self._name

    @property
    def dirname(self):
        return f'{self.prefix}{self.name}'

    def build(self):
        pass

    def discover_frames(self):
        for path in (output_path / self.dirname).glob('*.png'):
            self.frames[pd.to_datetime(path.stem)] = path
        if not self.frames:
            raise ValueError(f'No frames for {self.name}')
        return sorted(self.frames)

    def clip(self, frames, frame_duration):
        clips = []
        for f in tqdm(frames, desc=self.name):
            path = self.frames[f]
            clips.append(ImageClip(imageio.imread(path), duration=frame_duration))
        return concatenate_videoclips(clips, method="chain")


class TextPart(Part):

    prefix = 'animated_text_'

    def __init__(self, name, template, dates=None, fontsize=20, color="black", **margin):
        super().__init__(name)
        self.dynamic = '{' in template
        self.template = template
        self.dates = dates or pd.date_range(data_start, date.today())
        self.margin = margin or {'margin': 5}
        self.fontsize = fontsize
        self.color = color

    def build(self):
        cmd = [sys.executable, 'animated_text.py', self.name, self.template,
               '--fontsize', self.fontsize, '--color', self.color,
               '--from', self.dates[0], '--to', self.dates[-1]]
        if not self.dynamic:
            cmd.extend(('--single', self.dates[0]))
        run(cmd)

    def discover_frames(self):
        if self.dynamic:
            return super().discover_frames()
        return self.dates

    def clip(self, dates, frame_duration):
        if self.dynamic:
            clip = super().clip(dates, frame_duration)
        else:
            clip = ImageClip(imageio.imread(next((output_path / self.dirname).glob('*.png'))),
                             duration=frame_duration * len(dates))
        if self.margin:
            params = self.margin.copy()
            params['mar'] = params.pop('margin')
            params['color'] = white
            clip = margin(clip, **params)
        return clip


class MapPart(Part):

    prefix = 'animated_map_'

    def __init__(self, map_name: str, title: str = None, area_type: str = None, view: str = None,
                 start: Date = None, end: Date = None, dpi: int = None):
        super().__init__()
        self.area_type = area_type
        self.map = map_name
        self.view = view
        self.title = title
        self.start = start
        self.end = end
        self.dpi = dpi

    @property
    def name(self):
        return f'{self.area_type}_{self.map}_{self.view}'

    def build(self):
        cmd = [sys.executable, 'animated_map.py', self.area_type, self.map, '--view', self.view,
               '--bare', '--output', 'none']
        if self.title:
            cmd.extend(('--title', self.title))
        if self.start:
            cmd.extend(('--from', self.start))
        if self.end:
            cmd.extend(('--to-date', self.end))
        if self.dpi:
            cmd.extend(('--dpi', self.dpi))
        run(cmd)


class SummaryPart(Part):

    def __init__(self,
                 left_series: Sequence[str], right_series: Sequence[str],
                 left_formatter: str = None, right_formatter: str = None,
                 start: Date = None, end: Date = None,
                 width: int = 15, height: float = 2,
                 dpi: int = None):
        super().__init__(f'animated_summary_{width}_{height}')
        self.left_series = left_series
        self.right_series = right_series
        self.left_formatter = left_formatter
        self.right_formatter = right_formatter
        self.start = start
        self.end = end
        self.width = width
        self.height = height
        self.dpi = dpi

    def build(self):
        cmd = [sys.executable, 'animated_summary.py',
               ','.join(self.left_series), ','.join(self.right_series),
               '--width', self.width, '--height', self.height]
        if self.left_formatter:
            cmd.extend(('--lf', self.left_formatter))
        if self.right_formatter:
            cmd.extend(('--rf', self.right_formatter))
        if self.start:
            cmd.extend(('--from', self.start))
        if self.end:
            cmd.extend(('--to', self.end))
        if self.dpi:
            cmd.extend(('--dpi', self.dpi))
        run(cmd)


class Composition:

    def __init__(self, *rows: List[Part], **attrs):
        self.parts: List[Part] = list(chain(*rows))
        self.rows: Tuple[List[Part]] = rows
        for part in self.parts:
            for attr, value in attrs.items():
                if hasattr(part, attr):
                    current = getattr(part, attr)
                    if current is None:
                        setattr(part, attr, value)


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--threads', type=int, default=cpu_count())
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--duration', type=float, default=0.06,
                        help='fast=0.05, slow=0.3')
    args = parser.parse_args()

    composition = compositions[args.name]
    if args.build:
        for part in composition.parts:
            part.build()

    start = pd.Timestamp.min
    end = pd.Timestamp.max
    for part in composition.parts:
        part_dates = part.discover_frames()
        part_start = part_dates[0]
        part_end = part_dates[-1]
        start = max(start, part_start)
        end = min(end, part_end)
        print(f'{part.name}: {humanize(part_start)} to {humanize(part_end)}')
    print(f'final: {humanize(start)} to {humanize(end)}')
    dates = pd.date_range(start, end)

    final = clips_array(
        [[clips_array([[p.clip(dates, args.duration) for p in row]], bg_color=white)]
         for row in composition.rows],
        bg_color=white
    )
    final = final.set_duration(args.duration * len(dates))

    # 4k = 3840
    # 1080p = 1920
    final.write_videofile(str(output_path / f"{args.name}.mp4"),
                          fps=24, threads=args.threads, bitrate='10M')


footer = [TextPart(
    'footer',
    "@chriswithers13 - data from https://coronavirus.data.gov.uk/",
    fontsize=14, color="darkgrey"
)]


def cases_admissions_deaths_summary(width=28, height=4):
    return [SummaryPart(
        left_series=[new_cases_sum],
        right_series=[new_admissions_sum, new_deaths_sum],
        right_formatter='0k',
        width=width, height=height
    )]


def cases_tests_composition(view, *, start, end, summary_width=28, summary_height=4):
    return Composition(
        [
            MapPart('cases', "Confirmed Cases", area_type=msoa),
            MapPart('tested', "People Tested", area_type=ltla),
        ],
        cases_admissions_deaths_summary(summary_width, summary_height),
        footer,
        start=start, end=end, view=view, dpi=90
    )


compositions = {
    'tested-positivity-cases': Composition(
        [
            MapPart('tested', "Population Tested"),
            MapPart('positivity', "Test Positivity"),
            MapPart('cases-red', "Confirmed Case Rate"),
        ],
        cases_admissions_deaths_summary(),
        footer,
        area_type=ltla, view='england', start=data_start
    ),
    'cases-admissions-deaths': Composition(
        [
            MapPart('cases-red', "New Cases"),
            MapPart('admissions', "Hospital Admissions", area_type=nhs_region),
            MapPart('deaths', "Deaths"),
        ],
        cases_admissions_deaths_summary(),
        footer,
        area_type=ltla, view='england', start='2020-03-19',
    ),
    'colwall': cases_tests_composition(
        'colwall', start='2020-07-01', end='2020-08-15'
    ),
    'cases-area-type': Composition(
        [TextPart('header', 'PHE News Cases by Specimen Date as of {date:%d %b %y}',
                  fontsize=40)],
        [
            MapPart('cases-7', area_type=ltla),
            MapPart('cases', area_type=msoa),
        ],
        [SummaryPart(left_series=[unique_people_tested_sum],
                     left_formatter='1m',
                     right_series=[new_admissions_sum, new_deaths_sum])],
        footer,
        start=data_start, view='england', dpi=150
    ),
    'hammersmith': cases_tests_composition(
        'hammersmith', start='2020-10-01', end='2020-11-15',
        summary_width=24
    ),
    'leicester': cases_tests_composition(
        'leicester', start='2020-05-15', end='2020-08-30'
    ),
    'luton': cases_tests_composition(
        'luton', start='2020-07-10', end='2020-09-01'
    ),
    'rutland': cases_tests_composition(
        'rutland', start='2021-01-15', end='2021-03-01'
    ),
}


if __name__ == '__main__':
    main()
