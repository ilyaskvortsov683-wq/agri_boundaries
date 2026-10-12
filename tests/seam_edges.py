"""Сколько ребер полигонов лежит на линиях швов тайлов: python3 tests/seam_edges.py fields.gpkg tif TILE [layer]"""
import sys, numpy as np, geopandas as gpd, shapely, rasterio, warnings
warnings.filterwarnings('ignore')
f, tif, TILE = sys.argv[1], sys.argv[2], int(sys.argv[3]); lay = sys.argv[4] if len(sys.argv) > 4 else 'fields'
s = rasterio.open(tif); T = s.transform; H, W = s.height, s.width
g = gpd.read_file(f, layer=lay)
inv = ~T
tot = 0; n_edges = 0; byk = 0
bnd = shapely.union_all(g.geometry.boundary.values)
for c in range(TILE, W, TILE):
    x = T.c + c * T.a; line = shapely.LineString([(x, T.f), (x, T.f + T.e * H)])
    inter = bnd.intersection(line.buffer(abs(T.a) * 0.02))
    tot += inter.length / abs(T.a) if not inter.is_empty else 0
for r in range(TILE, H, TILE):
    y = T.f + r * T.e; line = shapely.LineString([(T.c, y), (T.c + W * T.a, y)])
    inter = bnd.intersection(line.buffer(abs(T.e) * 0.02))
    tot += inter.length / abs(T.e) if not inter.is_empty else 0
seam_len = (len(range(TILE, W, TILE)) * H + len(range(TILE, H, TILE)) * W)
print(f'длина ребер полигонов на швах: {tot:.0f} px из {seam_len} px шовных линий ({100*tot/seam_len:.1f} %)')
