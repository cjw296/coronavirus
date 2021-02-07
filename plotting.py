import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.io import reset_output, output_notebook, show, output_file, save
from bokeh.layouts import row
from bokeh.models import GeoJSONDataSource, HoverTool
from bokeh.palettes import Reds
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.transform import linear_cmap
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter

from constants import phe_vmax, new_cases_by_specimen_date, population, pct_population, base_path
from geo import views, old_ltla_geoms


def show_area(ax, view=views['uk']):
    ax.set_axis_off()
    minx, miny, maxx, maxy = view.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def geoplot_matplotlib(df, ax, column, title, label, vmax=None, missing_kwds=None):
    df.plot(ax=ax,
        column=column,
        k=10,
        cmap='Reds',
        legend=True,
        legend_kwds={'fraction': 0.02, 'anchor': (0, 0), 'label': label},
        missing_kwds=missing_kwds,
        vmax=vmax,
    )
    show_area(ax)
    ax.set_title(title)


def save_to_disk(p, filename, title, show_inline=True):
    if show_inline:
        reset_output()
        output_notebook(resources=INLINE)
        show(p)
    reset_output()
    output_file(filename, title=title)
    save(p)
    reset_output()
    output_notebook(resources=INLINE)


def geoplot_bokeh(data, title, column, tooltips, x_range=None, y_range=None, vmax=None):
    p = figure(title=title,
               plot_height=800,
               plot_width=600,
               toolbar_location='below',
               tools="pan, wheel_zoom, box_zoom, reset",
               active_scroll='wheel_zoom',
               match_aspect=True,
               x_range=x_range,
               y_range=y_range)

    if vmax is None:
        vmax = data[column].max()
    areas = p.patches(
        'xs', 'ys', source=GeoJSONDataSource(geojson=data.to_json()),
        fill_color=linear_cmap(column, tuple(reversed(Reds[256])), 0, vmax, nan_color='gray'),
        line_color='gray',
        line_width=0,
        fill_alpha=1
    )

    p.add_tools(HoverTool(
        renderers=[areas],
        tooltips=tooltips,
    ))

    return p


def matplotlib_zoe_vs_phe_map(zoe_df, zoe_date, phe_recent_geo, phe_recent_title):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

    geoplot_matplotlib(zoe_df, axes[0],
                       column='percentage',
                       title=f'ZOE COVID Symptom Study data for {zoe_date:%d %b %Y}',
                       label='Estimated Symptomatic Percentage')

    geoplot_matplotlib(phe_recent_geo, axes[1],
                       column=pct_population,
                       title=phe_recent_title,
                       label='lab-confirmed cases as % of population',
                       vmax=phe_vmax,
                       missing_kwds={'color': 'lightgrey'})

    fig.set_facecolor('white')
    plt.show()


def bokeh_zoe_vs_phe_map(zoe_df, zoe_date, phe_recent_geo, phe_recent_title):
    zoe_new_lad16 = pd.merge(
        old_ltla_geoms(),
        zoe_df[['lad16cd', 'lad16nm', 'percentage', 'percentage_string']],
        how='left',
        left_on='code', right_on='lad16cd',
    )
    zoe_title = f'ZOE COVID Symptom Study data for {zoe_date:%d %b %Y}'
    zoe = geoplot_bokeh(zoe_new_lad16.to_crs('EPSG:3857'), zoe_title, 'percentage', tooltips=[
        ('Name','@lad16nm'),
        ('Percentage','@{percentage}{1.111}%'),
    ])

    phe_data = phe_recent_geo[[
        'geometry', 'name', 'code', new_cases_by_specimen_date, population, pct_population
    ]]
    phe = geoplot_bokeh(phe_data, phe_recent_title, pct_population,
                  x_range=zoe.x_range, y_range=zoe.y_range, vmax=phe_vmax, tooltips=[
            ('Name','@name'),
            ('Code','@code'),
            ('Cases', '@{'+new_cases_by_specimen_date+'}{1}'),
            ('Population', '@{population}{1}'),
            ('Percentage','@{'+pct_population+'}{1.111}%'),
    ])

    p = row(zoe, phe)
    save_to_disk(p, "zoe_phe.html", title='ZOE modelled estimates versus PHE lab confirmed cases', show_inline=False)


def stacked_bar_plot(ax, data, colormap):
    pos_prior = neg_prior = pd.Series(0, data.index)
    colormap = get_cmap(colormap)
    ncolors = data.shape[1]
    colors = [colormap(num) for num in np.linspace(0, 1, num=ncolors)]
    handles = []
    for i, (name, series) in enumerate(data.iteritems()):
        mask = series > 0
        bottom = np.where(mask, pos_prior, neg_prior)
        handles.append(
            ax.bar(data.index, series, width=1, bottom=bottom, label=name, color=colors[i])
        )
        pos_prior = pos_prior + np.where(mask, series, 0)
        neg_prior = neg_prior + np.where(mask, 0, series)
    return handles


per1m_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000_000:.1f}m")
per1k_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000:,.0f}k")
per0k_formatter = FuncFormatter(lambda y, pos: f"{y / 1_000:,.1f}k")
