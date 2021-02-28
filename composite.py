import subprocess
import sys
from argparse import ArgumentParser
from datetime import date
from functools import partial
from itertools import chain
from multiprocessing import cpu_count
from typing import List, Tuple

import imageio
import pandas as pd
from matplotlib.colors import to_rgba
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.fx.margin import margin
from tqdm.auto import tqdm

from constants import output_path, ltla, data_start


def rgb_color(name):
    return tuple(e*255 for e in to_rgba(name)[:3])


def humanize(dt):
    return dt.strftime('%d %b %y')


white = rgb_color('white')
fps = 24


def run(cmd):
    cmd = [str(c.date()) if isinstance(c, pd.Timestamp) else str(c) for c in cmd]
    print('Running: ', subprocess.list2cmdline(['python']+cmd[1:]))
    subprocess.run(cmd)


class Part:

    prefix = ''

    def __init__(self, name):
        self.name = name
        self.dirname = f'{self.prefix}{name}'
        self.frames = {}

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

    def __init__(self, area_type: str, view: str, map_name: str, title: str = None,
                 start: date = None, end: date = None):
        super().__init__(f'{area_type}_{map_name}_{view}')
        self.area_type = area_type
        self.map = map_name
        self.view = view
        self.title = title
        self.start = start
        self.end = end

    def build(self):
        cmd = [sys.executable, 'animated_map.py', self.area_type, self.map, '--view', self.view,
               '--bare', '--output', 'none']
        if self.title:
            cmd.extend(('--title', self.title))
        if self.start:
            cmd.extend(('--from', self.start))
        if self.end:
            cmd.extend(('--to', self.end))
        run(cmd)


class Composition:

    def __init__(self, *rows: List[Part]):
        self.parts: List[Part] = list(chain(*rows))
        self.rows: Tuple[List[Part]] = rows


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--threads', type=int, default=cpu_count())
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--duration', type=float, default=1/fps,
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


all_ltla_england = partial(MapPart, ltla, 'england', start=data_start)


compositions = {
        'tested-positivity-cases': Composition(
            [TextPart('title', "PHE data for {date:%d %b %Y}")],
            [
                all_ltla_england('tested', "Population Tested"),
                all_ltla_england('positivity', "Test Positivity"),
                all_ltla_england('cases-red', "Confirmed Case Rate"),
            ],
            [TextPart(
                'footer',
                "@chriswithers13 - data from https://coronavirus.data.gov.uk/",
                fontsize=14, color="darkgrey"
            )]
        ),
}


if __name__ == '__main__':
    main()
