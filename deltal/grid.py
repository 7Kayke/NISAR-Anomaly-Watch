"""Grade comum: todas as fontes são reamostradas para a mesma grade UTM."""
from dataclasses import dataclass

from pyproj import CRS, Transformer
from rasterio.transform import from_origin


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Grid:
    bbox: tuple          # (oeste, sul, leste, norte) em graus
    resolution: float    # metros
    epsg: int
    transform: object
    width: int
    height: int

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def pixel_area_m2(self) -> float:
        return self.resolution ** 2


def make_grid(bbox, resolution=20.0) -> Grid:
    w, s, e, n = bbox
    epsg = utm_epsg((w + e) / 2, (s + n) / 2)
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    xs, ys = zip(*[tr.transform(x, y) for x, y in [(w, s), (w, n), (e, s), (e, n)]])
    x0 = (min(xs) // resolution) * resolution
    y1 = -((-max(ys)) // resolution) * resolution
    width = int(-(-(max(xs) - x0) // resolution))
    height = int(-(-(y1 - min(ys)) // resolution))
    return Grid(bbox, resolution, epsg, from_origin(x0, y1, resolution, resolution), width, height)
