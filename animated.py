import concurrent.futures
from argparse import ArgumentParser
from datetime import timedelta
from functools import partial
from os import cpu_count
from shutil import rmtree
from typing import Union

import imageio
import numpy as np
import pandas as pd
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pygifsicle import optimize
from tqdm import tqdm

from args import add_date_arg
from constants import output_path, second_wave


def parallel_render(name, render: partial, items, duration: Union[float, list],
                    outputs: str = 'gif'):

    # do this up front to catch typos cheaply:
    outputs = [output_types[output] for output in outputs.split(',')]

    image_path = output_path / name
    if image_path.exists():
        rmtree(image_path)
    image_path.mkdir()
    render.keywords['image_path'] = image_path

    with concurrent.futures.ProcessPoolExecutor() as executor:
        tuple(tqdm(executor.map(render, items), total=len(items), desc='rendering'))

    image_paths = sorted(image_path.iterdir())
    if not isinstance(duration, list):
        duration = [duration] * len(image_paths)

    # still last frame
    image_paths.append(image_paths[-1])
    duration.append(3)

    data = list(tqdm((imageio.imread(filename) for filename in image_paths),
                     total=len(image_paths), desc='loading'))

    for output in outputs:
        output(name, data, duration)


def output_gif(name, data, durations):
    # another still last frame to help twitter's broken gif playback
    data.append(data[-1])
    durations.append(3)

    print('saving...')
    gif_path = output_path / (name+'.gif')
    imageio.mimsave(gif_path, data, duration=durations)

    print('shrinking...')
    optimize(str(gif_path))


def output_mp4(name, data, durations):
    # load the images
    clips = [ImageClip(data, duration=d) for (data, d) in zip(data, durations)]
    # save the mp4
    movie = concatenate_videoclips(clips, method="chain")
    movie.write_videofile(str(output_path / (name + '.mp4')),
                          fps=24,
                          threads=cpu_count(),
                          bitrate='10M')

# output for twitter:
# ffmpeg -i in.mp4 -filter:v "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p" out.mp4


output_types = {
    'mp4': output_mp4,
    'gif': output_gif,
}


def slowing_durations(dates, normal=0.05, slow=0.3, period=30):
    durations = np.full((len(dates)), normal)
    durations[-period:] = np.geomspace(normal, slow, period)
    return list(durations)


def map_main(name, read_map_data, render_dt, default_exclude=0):
    parser = ArgumentParser()
    add_date_arg(parser, default=second_wave)
    parser.add_argument('--exclude-days', default=default_exclude, type=int)
    parser.add_argument('--output', default='mp4')
    args = parser.parse_args()

    df, data_date = read_map_data()

    to_date = df.index.max().date() - timedelta(days=args.exclude_days)
    earliest_date = df.index.min().date()
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date, earliest_date, to_date)

    parallel_render(name, render, dates, slowing_durations(dates), args.output)
