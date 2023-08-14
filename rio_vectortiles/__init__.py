"""rio_vectortiles"""
import mercantile
from shapely import geometry
import rasterio
from vtzero.tile import Tile, Layer, Polygon
import gzip
from io import BytesIO
from affine import Affine
import numpy as np
from rasterio.warp import reproject
from rasterio.io import MemoryFile
from rasterio.features import shapes, sieve
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import warnings
from itertools import groupby
from rio_vectortiles.split import katana

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning) #type: ignore


def decompress_tile(tile_data):
    """Util to decompress data to bytes"""
    with BytesIO(tile_data) as src:
        with gzip.open(src, "rb") as gz:
            return gz.read()


def filtered_coords(coords):
    """ Filter out duplicate coords"""
    fixed_coords = []
    previous_coord =  None
    for coord in coords:
        coord = (int(coord[0]), int(coord[1]))
        if previous_coord is not None and coord[0] == previous_coord[0] and coord[1] == previous_coord[1]:
            continue
        fixed_coords.append(coord)
        previous_coord = coord
    return fixed_coords

def read_transform_tile(
    tile,
    src_path=None,
    output_kwargs={},
    extent_func=None,
    interval=1,
    layer_name="raster",
    filters=[],
    filter_on_val=[],
):
    """Warp to dimensions and vectorize

    Parameters
    ----------
    tile: mercantile.Tile
        the tile to create
    src_path: str
        path to the raster to transform
    output_kwargs: dict
        base creation options for the intermediate raster
    extent_func: func()
        a function needing a single parameter {z}
    interval: number
        interval to vectorize on
    layer_name: str
        name of the created raster layer
    interval: float
        interval to vectorize on

    Returns
    -------
    vector_tile: bytes
        gzipped-compressed vector tile
    tile: mercantile.Tile
        the passed-through tile object
    """
    xy_bounds = mercantile.xy_bounds(*tile)
    if extent_func is None:
        raise ValueError("extent_func must be provided")
    extent = extent_func(tile.z)
    dst_transform = from_bounds(*xy_bounds, extent, extent)
    dst_kwargs = {
        **output_kwargs,
        **{"transform": dst_transform, "width": extent, "height": extent},
    }

    with rasterio.open(src_path) as src:
        src_band = rasterio.band(src, bidx=1)
        vtile = Tile()
        layer = Layer(vtile, layer_name.encode(), version=2, extent=extent)
        sieve_value = 2
        with MemoryFile() as mem:
            with mem.open(**dst_kwargs) as dst:
                dst_band = rasterio.band(dst, bidx=1)
                reproject(src_band, dst_band, resampling=Resampling.mode)
                dst.transform = Affine.identity()
                if interval is None:
                    data = sieve(dst_band, sieve_value)
                else:
                    data = dst.read(1)
                    data = (data // interval * interval).astype(np.int32)

                if filter_on_val:
                    data = data.astype(np.int32)
                    mask = np.isin(data, filter_on_val)
                    data[~mask] = -1


                vectorizer = shapes(data)

                grouped_vectors = groupby(
                    sorted(vectorizer, key=lambda x: x[1]), key=lambda x: x[1]
                )

                for v, geoms in grouped_vectors:
                    if filter_on_val is not None and v == -1:
                        continue

                    geoms = geometry.MultiPolygon(
                        [geometry.shape(g) for g, _ in geoms]
                    ).buffer(0)
                    if geoms.geom_type == "Polygon":
                        iter_polys = [geoms]
                    else:
                        iter_polys = geoms.geoms
                    for possibly_large_geom in iter_polys:
                        # while the vertices limit in mapbox gl is 65535, we need to
                        # account for the fact the vertices are added as part of rendering
                        # so we need to be well under the limit to ensure all features are rendered properly
                        # How did we get to 5000? Kept descending until we got to a number that worked.
                        # This would be dataset dependent, but not sure how we'd programmatically determine this
                        # because it depends no the gl rendering process.
                        limited_coord_geoms = katana(possibly_large_geom, 5000)
                        for geom in limited_coord_geoms:
                            fixed_exterior_coords = filtered_coords(geom.exterior.coords)
                            if len(fixed_exterior_coords) < 3:
                                print("""Skipping feature part with less than 3 vertices""", len(fixed_exterior_coords))
                                continue
                            feature = Polygon(layer)
                            feature.add_ring(len(fixed_exterior_coords))

                            for coord in fixed_exterior_coords:
                                coord = (int(coord[0]), int(coord[1]))
                                feature.set_point(*coord)
                            for part in geom.interiors:
                                fixed_part_coords = filtered_coords(part.coords)
                                if len(fixed_part_coords) < 3:
                                    print("""Skipping feature part with less than 3 vertices""", len(fixed_part_coords))
                                    continue

                                feature.add_ring(len(fixed_part_coords))
                                for coord in fixed_part_coords:
                                    coord = (int(coord[0]), int(coord[1]))
                                    feature.set_point(*coord)

                            feature.add_property(b"val", v)
                            feature.commit()
    with BytesIO() as dst:
        with gzip.open(dst, mode="wb") as gz:
            gz.write(vtile.serialize())
        dst.seek(0)
        return dst.read(), tile
