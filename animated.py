import concurrent.futures
from functools import partial
from os import cpu_count
from shutil import rmtree
from typing import Union

import imageio
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pygifsicle import optimize
from tqdm import tqdm

from constants import base_path


def parallel_render(name, render: partial, items, duration: Union[float, list],
                    output: Union[str, callable] = 'gif'):

    image_path = base_path / name
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

    if not callable(output):
        output = output_types[output]
    output(name, image_paths, duration)


def output_gif(name, image_paths, durations):
    # another still last frame to help twitter's broken gif playback
    image_paths.append(image_paths[-1])
    durations.append(3)

    data = list(tqdm((imageio.imread(filename) for filename in image_paths),
                     total=len(image_paths), desc='loading'))

    print('saving...')
    gif_path = base_path / (name+'.gif')
    imageio.mimsave(gif_path, data, duration=durations)

    print('shrinking...')
    optimize(str(gif_path))


def output_mp4(name, image_paths, durations):
    # load the images
    clips = list(tqdm((ImageClip(str(p), duration=d) for (p, d) in zip(image_paths, durations)),
                      total=len(image_paths), desc='loading'))
    # save the mp4
    movie = concatenate_videoclips(clips, method="chain")
    movie.write_videofile(str(base_path / (name + '.mp4')),
                          fps=24,
                          threads=cpu_count(),
                          bitrate='10M')

# output for twitter:
# ffmpeg -i in.mp4 -filter:v "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p" out.mp4


output_types = {
    'mp4': output_mp4,
    'gif': output_gif,
}
