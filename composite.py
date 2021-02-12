from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import chain
from multiprocessing import cpu_count
from typing import List

import imageio
import pandas as pd
from matplotlib.colors import to_rgba
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.fx.margin import margin
from tqdm.auto import tqdm

from constants import output_path, ltla


def rgb_color(name):
    return tuple(e*255 for e in to_rgba(name)[:3])


white = rgb_color('white')
fps = 24


class Part:

    def __init__(self, name):
        self.name = name
        self.frames = {}
        for path in (output_path / name).glob('*'):
            self.frames[pd.to_datetime(path.stem)] = path
        if not self.frames:
            raise ValueError(f'No frames for {name}')

    def dates(self):
        return sorted(self.frames)

    def clip(self, dates, frame_duration):
        clips = []
        for d in tqdm(dates, desc=self.name):
            path = self.frames[d]
            clips.append(ImageClip(imageio.imread(path), duration=frame_duration))
        return concatenate_videoclips(clips, method="chain")


class TextPart:

    def __init__(self, template, fontsize, color="black", **margin):
        self.name = template
        self.dynamic = '{' in template
        self.template = template
        self.params = dict(font="DejaVu Sans", fontsize=fontsize, color=color, bg_color='white')
        self.margin = margin

    def _clip(self, text, duration):
        clip = TextClip(text, **self.params)
        return clip.set_duration(duration)

    def clip(self, dates, frame_duration):
        if self.dynamic:
            clip = concatenate_videoclips(
                list(tqdm((self._clip(self.template.format(date=d), frame_duration) for d in dates),
                     desc=self.name, total=len(dates))),
                method="chain"
            )
        else:
            clip = self._clip(self.template, frame_duration * len(dates))
        if self.margin:
            params = self.margin.copy()
            params['mar'] = params.pop('margin')
            params['color'] = white
            clip = margin(clip, **params)
        return clip


def maps(*metrics: str, area_type: str = ltla, area: str = 'england') -> List[Part]:
    return [Part(f'animated_map_{area_type}_{m}_{area}') for m in metrics]


@dataclass
class Composition:
    title: TextPart
    parts: List[List[Part]]
    date_lock: bool = True

    def all_parts(self):
        yield [self.title]
        for row in self.parts:
            yield row
        yield [TextPart(
            "@chriswithers13 - data from https://coronavirus.data.gov.uk/",
            fontsize=14, color="darkgrey", margin=5
        )]


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--threads', type=int, default=cpu_count())
    parser.add_argument('--duration', type=float, default=1/fps,
                        help='fast=0.05, slow=0.3')
    args = parser.parse_args()

    composition = compositions[args.name]

    rows = list(composition.all_parts())

    start = pd.Timestamp.min
    end = pd.Timestamp.max
    for part in chain(*rows):
        get_dates = getattr(part, 'dates', None)
        if get_dates:
            part_dates = get_dates()
            part_start = part_dates[0]
            part_end = part_dates[-1]
            start = max(start, part_start)
            end = min(end, part_end)
            print(f'{part.name}: {part_start}-{part_end}')
    print(f'final: {start}-{end}')
    dates = pd.date_range(start, end)

    print(rows)
    final = clips_array(
        [[clips_array([[p.clip(dates, args.duration) for p in row]], bg_color=white)]
         for row in rows],
        bg_color=white
    )
    final = final.set_duration(args.duration * len(dates))

    # 4k = 3840
    # 1080p = 1920
    final.write_videofile(str(output_path / "composite.mp4"),
                          fps=24, threads=args.threads, bitrate='10M')


compositions = {
        'tested-positivity-cases': Composition(
            TextPart("PHE data for {date: %d %b %Y}", fontsize=20, margin=5),
            [maps('tested', 'positivity')+maps('cases-red', area_type='msoa')],
        ),
}


if __name__ == '__main__':
    main()
