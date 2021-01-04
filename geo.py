from dataclasses import dataclass

import shapely
from geopandas import GeoDataFrame


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
