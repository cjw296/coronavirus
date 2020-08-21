from functools import partial
from shutil import rmtree
import concurrent.futures

from pygifsicle import optimize
from tqdm import tqdm
import imageio

from constants import base_path


def parallel_render(name, render: partial, items, duration):
    
    image_path = base_path / name
    if image_path.exists():
        rmtree(image_path)
    image_path.mkdir()
    render.keywords['image_path'] = image_path

    with concurrent.futures.ProcessPoolExecutor() as executor:
        tuple(tqdm(executor.map(render, items), total=len(items), desc='rendering'))

    data = list(tqdm((imageio.imread(filename) for filename in sorted(image_path.iterdir())),
                     total=len(items), desc='loading'))
    data.append(data[-1])

    print('saving...')
    gif_path = base_path / (name+'.gif')
    imageio.mimsave(gif_path, data, duration=duration)

    print('shrinking...')
    optimize(str(gif_path))
