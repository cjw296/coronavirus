from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Sequence

import geopandas
import pandas as pd
import shapely
from geopandas import GeoDataFrame

from constants import repo_path


@lru_cache
def geoportal_geoms(name, code_column=None, name_column=None):
    # geoms that comes from https://geoportal.statistics.gov.uk/
    geoms = geopandas.read_file(repo_path / 'geo' / f"{name}-shp" / f"{name}.shp")
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
    names = pd.read_csv(repo_path / 'geo' / 'MSOA-Names-1.10.csv',
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


def convert_df(df, geom_col):
    geoms = df[geom_col]
    df[geom_col] = geoms = geoms.apply(lambda x: shapely.wkb.loads(str(x), hex=True))
    crs = "epsg:{}".format(shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom))
    return GeoDataFrame(df, crs=crs, geometry=geom_col)


def above(geometry):
    x = geometry.centroid.coords[0][0]
    minx, miny, maxx, maxy = geometry.bounds
    y = maxy + (maxy - miny) * 0.05
    return x, y


def center(geometry):
    return geometry.centroid.coords[0]


class Places:

    def __init__(self, *names,
                 outline_colour='black',
                 outline_width=0.5,
                 label_location=above,
                 geom_source=town_and_city_geoms):
        self.geom_source = geom_source
        self.names = names
        self.outline_colour = outline_colour
        self.outline_width = outline_width
        self.label_location = label_location

    def frame(self):
        geoms = self.geom_source()
        return geoms[geoms['name'].isin(self.names)]


PlacesFrom = Enum('PlacesFrom', 'show outline')


@dataclass
class View:

    minx: float = None
    maxx: float = None
    miny: float = None
    maxy: float = None

    width: float = 10
    height: float = 15
    ratio: int = 9

    show: Places = ()
    outline: Sequence[Places] = ()
    label: Sequence[Places] = ()

    margin_pct = 10

    def __post_init__(self):
        if self.minx is None:
            show = self.show.frame()
            minx, miny, maxx, maxy = show.geometry.total_bounds
            factor = self.margin_pct / 100
            x_margin = (maxx-minx)*factor
            self.minx = minx - x_margin
            self.maxx = maxx + x_margin
            y_margin = (maxy-miny)*factor
            self.miny = miny - y_margin
            self.maxy = maxy + y_margin
        for attr in 'outline', 'label':
            value = getattr(self, attr)
            if value is PlacesFrom.show:
                value = [getattr(self, 'show')]
            elif value is PlacesFrom.outline:
                value = getattr(self, 'outline')
            setattr(self, attr, value)


views = {
    'uk': View(-900_000, 200_000, 6_460_000, 8_000_000),
    'england': View(-640_000, 200_000, 6_460_000, 7_520_000),
    'reading': View(
        show=Places('Reading'),
        outline=[Places('Reading'),
                 Places('Earley',
                        'Lower Earley North',
                        'Lower Earley South',
                        outline_colour='white',
                        outline_width=1,
                        geom_source=msoa_geoms_20)],
        label=PlacesFrom.show,
        width=10, height=10, ratio=7
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
        width=15, height=10, ratio=4
    ),
    'reading-london': View(
        show=Places('Reading', 'London'),
        outline=PlacesFrom.show,
        label=PlacesFrom.show,
        width=15, height=10, ratio=3
    ),
    'london': View(show=Places('London'), height=11, ratio=3),
}
