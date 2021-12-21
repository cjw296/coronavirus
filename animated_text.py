from argparse import ArgumentParser
from datetime import date
from functools import partial
from pathlib import Path


from animated import parallel_render
from args import add_parallel_args, parallel_params, parallel_to_date
from moviepy.video.VideoClip import TextClip
import pandas as pd


def render_text(template, fontsize, color, frame_date, image_path: Path):
    dt = frame_date.date()

    text = template.format(date=frame_date)
    text_path = image_path / f'{dt}.txt'
    text_path.write_text(text)

    png_path = image_path / f'{dt}.png'

    TextClip(
        text, font="DejaVu Sans", fontsize=fontsize, color=color, bg_color='white',
        tempfilename=str(png_path), remove_temp=False,
    )

    text_path.unlink()


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('text')
    parser.add_argument('--fontsize', type=int, default=20)
    parser.add_argument('--color', default='black')
    add_parallel_args(parser, default_output='none')
    args = parser.parse_args()

    render = partial(render_text, args.text, args.fontsize, args.color)

    to_date = parallel_to_date(args, date.today())
    dates = pd.date_range(args.from_date, to_date)

    params = parallel_params(args, dates)
    parallel_render(f'animated_text_{args.name}', render, dates, **params)


if __name__ == '__main__':
    main()
