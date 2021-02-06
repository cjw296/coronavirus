import concurrent.futures
from functools import partial
from os import cpu_count
from shutil import rmtree
from typing import Union

import imageio
import numpy as np
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pygifsicle import optimize
from tqdm import tqdm

from constants import output_path


def safe_render(render, image_path, raise_errors, item):
    try:
        render(item, image_path=image_path)
    except Exception as e:
        print(f'Could not render for {item}: {type(e)}: {e}')
        if raise_errors:
            raise


def parallel_render(name, render: partial, items, duration: Union[float, list],
                    outputs: str = 'gif', raise_errors: bool = True, max_workers: int = None,
                    item=None):

    # do this up front to catch typos cheaply:
    outputs = [output_types[output] for output in outputs.split(',')]

    image_path = output_path / name
    if image_path.exists():
        rmtree(image_path)
    image_path.mkdir()
    renderer = partial(safe_render, render, image_path, raise_errors)

    if item is not None:
        renderer(item)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        tuple(tqdm(executor.map(renderer, items), total=len(items), desc='rendering'))

    if None in outputs:
        # just rendering the frames to be composited later
        return
    
    image_paths = sorted(image_path.iterdir())
    image_count = len(image_paths)
    if not isinstance(duration, list):
        durations = [duration] * image_count
    else:
        # make sure if there are too many durations, we ignore the ones at the start.
        durations = duration[-image_count:]

    # still last frame
    image_paths.append(image_paths[-1])
    durations.append(3)

    data = list(tqdm((imageio.imread(filename) for filename in image_paths),
                     total=len(image_paths), desc='loading'))

    for output in outputs:
        output(name, data, durations, max_workers)


def output_gif(name, data, durations, _):
    # another still last frame to help twitter's broken gif playback
    data.append(data[-1])
    durations.append(3)

    print('saving...')
    gif_path = output_path / (name+'.gif')
    imageio.mimsave(gif_path, data, duration=durations)

    print('shrinking...')
    optimize(str(gif_path))


def output_mp4(name, data, durations, max_workers=None):
    # sanity check the images
    sizes = {frame.shape for frame in data}
    assert len(sizes) == 1, sizes
    # turn the image into clips
    clips = [ImageClip(data, duration=d) for (data, d) in zip(data, durations)]
    # save the mp4
    movie = concatenate_videoclips(clips, method="chain")
    movie.write_videofile(str(output_path / (name + '.mp4')),
                          fps=24,
                          threads=max_workers or cpu_count(),
                          bitrate='10M')

# output for twitter:
# ffmpeg -i in.mp4 -filter:v "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p" out.mp4


output_types = {
    'mp4': output_mp4,
    'gif': output_gif,
    'none': None,
}


def round_nearest(a, nearest):
    return (a/nearest).round(0) * nearest


def slowing_durations(dates, normal=0.05, slow=0.3, period=30):
    durations = np.full((len(dates)), normal)
    period = min(len(durations), period)
    durations[-period:] = np.geomspace(normal, slow, period)
    return list(durations)


