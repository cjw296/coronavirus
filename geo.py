from dataclasses import dataclass
from functools import lru_cache

import geopandas
import shapely
from geopandas import GeoDataFrame

from constants import base_path


@lru_cache
def geoportal_geoms(name, code_column, name_column):
    # geoms that comes from https://geoportal.statistics.gov.uk/
    geoms = geopandas.read_file(base_path / f"{name}-shp" / f"{name}.shp")
    geoms.to_crs("EPSG:3857", inplace=True)
    geoms.rename(columns={code_column: 'code', name_column: 'name'},
                 errors='raise', inplace=True)
    return geoms


@lru_cache
def old_ltla_geoms():
    return geoportal_geoms(
        "Local_Authority_Districts__December_2016__Boundaries_UK",
        code_column='lad16cd',
        name_column='lad16nm',
    )


@lru_cache
def ltla_geoms():
    return geoportal_geoms(
        "Local_Authority_Districts__April_2019__UK_BUC_v2",
        code_column='LAD19CD',
        name_column='LAD19NM',
    )


def msoa_geoms():
    return geoportal_geoms(
        "Middle_Layer_Super_Output_Areas__December_2011__EW_BSC_V2",
        code_column='MSOA11CD',
        name_column='MSOA11NM',
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


@dataclass
class View:
    minx: int
    maxx: int
    miny: int
    maxy: int


views = {
    'uk': View(-900_000, 200_000, 6_460_000, 8_000_000),
    'england': View(-640_000, 200_000, 6_460_000, 7_520_000),
}
