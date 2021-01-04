import concurrent.futures
from argparse import ArgumentParser
from datetime import timedelta
from functools import partial
from os import cpu_count
from shutil import rmtree
from typing import Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pygifsicle import optimize
from tqdm import tqdm

from args import add_date_arg
from constants import output_path, second_wave
from phe import plot_summary
import series as s


def safe_render(render, image_path, raise_errors, item):
    try:
        render(item, image_path=image_path)
    except Exception as e:
        print(f'Could not render for {item}: {type(e)}: {e}')
        if raise_errors:
            raise


def parallel_render(name, render: partial, items, duration: Union[float, list],
                    outputs: str = 'gif', raise_errors: bool = True):

    # do this up front to catch typos cheaply:
    outputs = [output_types[output] for output in outputs.split(',')]

    image_path = output_path / name
    if image_path.exists():
        rmtree(image_path)
    image_path.mkdir()
    renderer = partial(safe_render, render, image_path, raise_errors)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        tuple(tqdm(executor.map(renderer, items), total=len(items), desc='rendering'))

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


def render_dt(data_date, earliest_date, to_date, dpi, render_map, frame_date, image_path):
    fig, (map_ax, lines_ax) = plt.subplots(
        figsize=(10, 15), nrows=2, gridspec_kw={'height_ratios': [9, 1], 'hspace': 0}
    )
    render_map(map_ax, frame_date)
    plot_summary(lines_ax, data_date, frame_date, earliest_date, to_date,
                 series=(s.new_admissions_sum, s.new_deaths_sum), title=False)
    fig.text(0.25, 0.08,
             f'@chriswithers13 - '
             f'data from https://coronavirus.data.gov.uk/ retrieved on {data_date:%d %b %Y}')
    plt.savefig(image_path / f'{frame_date.date()}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def slowing_durations(dates, normal=0.05, slow=0.3, period=30):
    durations = np.full((len(dates)), normal)
    durations[-period:] = np.geomspace(normal, slow, period)
    return list(durations)


def map_main(name, read_map_data, render_map, default_exclude=0, dpi=90):
    parser = ArgumentParser()
    add_date_arg(parser, default=second_wave)
    parser.add_argument('--exclude-days', default=default_exclude, type=int)
    parser.add_argument('--output', default='mp4')
    parser.add_argument('--raise-errors', action='store_true')
    args = parser.parse_args()

    df, data_date = read_map_data()

    to_date = df.index.max().date() - timedelta(days=args.exclude_days)
    earliest_date = df.index.min().date()
    dates = pd.date_range(args.from_date, to_date)

    render = partial(render_dt, data_date, earliest_date, to_date, dpi, render_map)

    parallel_render(name, render, dates, slowing_durations(dates),
                    args.output, raise_errors=args.raise_errors)
