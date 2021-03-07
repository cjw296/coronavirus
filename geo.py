from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import Sequence, Union

import geopandas
import pandas as pd
import shapely
from geopandas import GeoDataFrame

from constants import repo_path, ltla, msoa, nhs_region


@lru_cache
def geoportal_geoms(name, code_column=None, name_column=None, suffix='-shp'):
    # geoms that comes from https://geoportal.statistics.gov.uk/
    geoms = geopandas.read_file(repo_path / 'geodata' / f"{name}{suffix}" / f"{name}.shp")
    geoms.to_crs("EPSG:3857", inplace=True)
    if code_column and name_column:
        geoms.rename(columns={code_column: 'code', name_column: 'name'},
                     errors='raise', inplace=True)
    return geoms


def old_ltla_geoms():
    return geoportal_geoms(
        "Local_Authority_Districts__December_2016__Boundaries_UK",
        code_column='lad16cd',
        name_column='lad16nm',
    )


def ltla_geoms_500():
    return geoportal_geoms(
        "Local_Authority_Districts__April_2019__UK_BUC_v2",
        code_column='LAD19CD',
        name_column='LAD19NM',
    )


def ltla_geoms_20():
    return geoportal_geoms(
        "Local_Authority_Districts__April_2019__UK_BGC_v2",
        code_column='LAD19CD',
        name_column='LAD19NM',
    )


def utla_geoms_20():
    return geoportal_geoms(
        "Counties_and_Unitary_Authorities__December_2017___EW_BGC",
        code_column='CTYUA17CD',
        name_column='CTYUA17NM',
    )


def ltla_geoms_full():
    return geoportal_geoms(
        'Local_Authority_Districts__April_2019__UK_BFC_v2',
        code_column='LAD19CD',
        name_column='LAD19NM',
    )


ltla_geoms = ltla_geoms_500


@lru_cache
def msoa_geoms_with_names(source):
    # from https://visual.parliament.uk/msoanames
    names = pd.read_csv(repo_path / 'geodata' / 'MSOA-Names-1.10.csv',
                        usecols=['msoa11cd', 'msoa11hclnm'])
    geoms = geoportal_geoms(source)
    with_names = pd.merge(geoms, names, left_on='MSOA11CD', right_on='msoa11cd')
    with_names.rename(
        columns={'MSOA11CD': 'code', 'msoa11hclnm': 'name'},
        errors='raise', inplace=True
    )
    return with_names


def msoa_geoms_200():
    return msoa_geoms_with_names(
        "Middle_Layer_Super_Output_Areas__December_2011__EW_BSC_V2"
    )


def msoa_geoms_20():
    return msoa_geoms_with_names(
        "Middle_Layer_Super_Output_Areas__December_2011__EW_BGC_V2"
    )


def msoa_geoms_full():
    return msoa_geoms_with_names(
        "Middle_Layer_Super_Output_Areas__December_2011__Boundaries_Full_Extent__BFE__EW_V3"
    )


def town_and_city_geoms():
    return geoportal_geoms(
        "Major_Towns_and_Cities__December_2015__Boundaries",
        code_column='TCITY15CD',
        name_column='TCITY15NM',
    )


def nhs_regions_geoms():
    return geoportal_geoms(
        "NHS_England_Regions_(April_2020)_Boundaries_EN_BFC",
        code_column='nhser20cd',
        name_column='nhser20nm',
        suffix='',
    )


area_type_to_geoms = {
    ltla: ltla_geoms,
    msoa: msoa_geoms_20,
    nhs_region: nhs_regions_geoms,
}


def convert_df(df, geom_col):
    geoms = df[geom_col]
    df[geom_col] = geoms = geoms.apply(lambda x: shapely.wkb.loads(str(x), hex=True))
    crs = "epsg:{}".format(shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom))
    return GeoDataFrame(df, crs=crs, geometry=geom_col)


def above(geometry, factor=0.05):
    x = geometry.centroid.coords[0][0]
    minx, miny, maxx, maxy = geometry.bounds
    y = maxy + (maxy - miny) * factor
    return x, y


def below(geometry, factor=0.125):
    x = geometry.centroid.coords[0][0]
    minx, miny, maxx, maxy = geometry.bounds
    y = miny - (maxy - miny) * factor
    return x, y


def center(geometry):
    return geometry.centroid.coords[0]


class Places:

    def __init__(self, *names,
                 colour='black',
                 outline_width=0.5,
                 label_location=above,
                 geom_source=town_and_city_geoms,
                 fontsize='x-large',
                 fontweight=1000):
        self.geom_source = geom_source
        self.names = names
        self.colour = colour
        self.outline_width = outline_width
        self.label_location = label_location
        self.fontsize = fontsize
        self.fontweight = fontweight

    @lru_cache()
    def frame(self):
        geoms = self.geom_source()
        filtered = geoms[geoms['name'].isin(self.names)]
        found_names = set(filtered['name'])
        expected_names = set(self.names)
        if found_names != expected_names:
            missing = ','.join(repr(n) for n in expected_names - found_names)
            raise ValueError(f'{self.geom_source} does not contain {missing}')
        return filtered


class PlacesFrom:

    def __init__(self, plus=(), attr='show'):
        self.attr = attr
        self.plus = plus

    def __call__(self, view):
        value = getattr(view, self.attr)
        if not isinstance(value, (list, tuple)):
            value = [value]
        value.extend(self.plus)
        return value


places_from_show = PlacesFrom(attr='show')
places_from_outline = PlacesFrom(attr='outline')


@dataclass
class View:

    minx: float = None
    maxx: float = None
    miny: float = None
    maxy: float = None

    legend_fraction: float = 0.02
    grid_hspace = 0.05
    summary_height = 1.5

    show: Places = None
    outline: Union[Sequence[Places], callable] = ()
    label: Union[Sequence[Places], callable] = ()

    margin_pct: int = 10

    def check(self):
        # make sure all the places are valid, cache them if we're lucky!
        for places in [self.show], self.outline, self.label:
            for place in places:
                if place is not None:
                    place.frame()

    @cached_property
    def total_bounds(self):
        if self.minx is None:
            show = self.show.frame()
            minx, miny, maxx, maxy = show.geometry.total_bounds
            factor = self.margin_pct / 100
            x_margin = (maxx-minx)*factor
            minx = minx - x_margin
            maxx = maxx + x_margin
            y_margin = (maxy-miny)*factor
            miny = miny - y_margin
            maxy = maxy + y_margin
            return minx, miny, maxx, maxy
        else:
            return self.minx, self.miny, self.maxx, self.maxy

    def layout(self, summary_height=None):
        summary_height = self.summary_height if summary_height is None else summary_height
        minx, miny, maxx, maxy = self.total_bounds
        plot_height = maxy - miny
        plot_width = maxx - minx
        aspect = plot_width / plot_height
        if aspect > 1:
            # reading-london
            width = 15
            map_width = width * (1-self.legend_fraction)
            map_height = map_width / aspect
            height = map_height + summary_height + self.grid_hspace
        else:
            # uk / england
            height = 15
            map_height = height - summary_height - self.grid_hspace
            map_width = aspect * height
            width = map_width / (1-self.legend_fraction)

        ratio = map_height / summary_height if summary_height else 0
        return width, height, ratio

    def __post_init__(self):
        for attr in 'outline', 'label':
            value = getattr(self, attr)
            if callable(value):
                setattr(self, attr, value(view=self))


views = {
    'uk': View(-900_000, 200_000, 6_460_000, 8_000_000),
    'england': View(-640_000, 200_000, 6_460_000, 7_520_000),
    'bristol': View(
        show=Places('Bristol'),
        outline=PlacesFrom(attr='show', plus=[
            Places('Staple Hill North',
                   colour='white',
                   outline_width=1,
                   geom_source=msoa_geoms_20)
        ]),
        label=places_from_show,
        margin_pct=40
    ),
    'colwall': View(
        show=Places('Colwall, Cradley & Wellington Heath',
                        geom_source=msoa_geoms_20),
        outline=[Places('Colwall, Cradley & Wellington Heath',
                        geom_source=msoa_geoms_20, outline_width=1),
                 Places("Herefordshire, County of",
                        geom_source=ltla_geoms_20)],
        label=[Places('Colwall, Cradley & Wellington Heath',
                        geom_source=msoa_geoms_20)],
        margin_pct=300,
    ),
    'leicester': View(
        show=Places('Leicester'),
        outline=places_from_show,
        label=places_from_show,
        margin_pct=100
    ),
    'liverpool': View(
        show=Places('Liverpool', 'Wirral', 'Sefton', 'Warrington',
                    geom_source=ltla_geoms_20),
        outline=[Places('Liverpool')],
        label=places_from_outline,
    ),
    'london': View(show=Places('London'),
                   outline=places_from_show,
                   label=places_from_show,
                   margin_pct=30),
    'luton': View(
        show=Places('Northampton', 'Luton',
                        geom_source=ltla_geoms_20),
        outline=places_from_show,
        label=places_from_show,
    ),
    'reading': View(
        show=Places('Reading'),
        outline=[Places('Reading'),
                 Places('Earley',
                        'Lower Earley North',
                        'Lower Earley South',
                        colour='white',
                        outline_width=1,
                        geom_source=msoa_geoms_20)],
        label=places_from_show,
        margin_pct=40
    ),
    'poole': View(
        show=Places("Bournemouth, Christchurch and Poole", geom_source=ltla_geoms_20),
        outline=[Places(
            'Ferndown Town',
            'Poole Town',
            outline_width=1,
            geom_source=msoa_geoms_20
        )],
        label=[Places('Bournemouth', label_location=center)],
    ),
    'thames-valley': View(
        show=Places('Reading', 'London', 'Oxford'),
        outline=places_from_show,
        label=places_from_show,
    ),
    'west-midlands': View(
        show=Places('Sandwell',
                    'Wolverhampton',
                    'Walsall',
                    'Birmingham',
                    'Dudley',
                    'Telford and Wrekin',
                    'Coventry',
                    'Staffordshire',
                    'Shropshire',
                    'Stoke-on-Trent',
                    'Warwickshire',
                    'Solihull',
                    "Herefordshire, County of",
                    geom_source=utla_geoms_20),
        outline=PlacesFrom(attr='show', plus=[
            Places("Birmingham", 'Stoke-on-Trent', 'Telford',
                   colour='white')
        ]),
        label=[
            Places('Staffordshire', 'Shropshire',
                   label_location=above, geom_source=utla_geoms_20),
            Places('Herefordshire, County of', 'Warwickshire',
                   label_location=below, geom_source=utla_geoms_20),
            Places("Birmingham", 'Stoke-on-Trent', 'Telford',
                   label_location=below,
                   fontsize='medium',
                   colour='white')
        ],
        margin_pct=30,
    ),
    'wokingham': View(show=Places('Wokingham', geom_source=ltla_geoms_20),
                      outline=places_from_show, label=places_from_show,
                      margin_pct=100),
}
