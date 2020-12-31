import concurrent.futures
from functools import partial
from os import cpu_count
from shutil import rmtree
from typing import Union

import imageio
from dateutil.parser import parse as parse_date
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pygifsicle import optimize
from tqdm import tqdm

from constants import relax_2, second_wave, data_start, output_path, lockdown2


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

special_dates = {text: dt for dt, text in (
    (data_start, 'start'),
    (second_wave, 'second-wave'),
    (lockdown2[0], 'lockdown-2-start'),
    (lockdown2[1], 'lockdown-2-end'),
)}


def date_lookup(text):
    dt = special_dates.get(text)
    if dt is None:
        dt = parse_date(text).date()
    return dt


def add_date_arg(parser, name='--from-date', help='data release date', default=second_wave):

    parser.add_argument(name, default=default, type=date_lookup,
                        help=f"{help}: {', '.join(special_dates)}")
