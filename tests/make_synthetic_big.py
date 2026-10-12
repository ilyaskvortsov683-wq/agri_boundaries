"""Многотайловая синтетическая сцена с известным эталоном: python3 tests/make_synthetic_big.py out.tif gt.geojson [W H seed]
Поля — случайное разбиение на прямоугольники ~100–300 га; цвета (max,min,diff) розовые/тёмные; соседи часто одного тона,
границы между ними: 60 % едва заметные (min +10, max −8, 3 px), 25 % лесополосы (min +60, 3 px), 15 % без границы вообще.
Параметры растра как у TAT: EPSG:4326, пиксель 8.98315e-5°, 4 канала (4-й — маска)."""
import sys, json
import numpy as np
import rasterio
from rasterio.transform import from_origin

out, gt = sys.argv[1], sys.argv[2]
W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (6000, 4000)
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(seed)
PX = 8.98315e-05
T = from_origin(50.5, 55.55, PX, PX)

rects = []


def split(x0, y0, x1, y1, depth=0):
    w, h = x1 - x0, y1 - y0
    if (w < 420 and h < 420 and rng.random() < 0.8) or (w < 220 and h < 220):
        rects.append((x0, y0, x1, y1)); return
    if w >= h:
        c = x0 + int(w * rng.uniform(0.35, 0.65)); split(x0, y0, c, y1, depth + 1); split(c, y0, x1, y1, depth + 1)
    else:
        c = y0 + int(h * rng.uniform(0.35, 0.65)); split(x0, y0, x1, c, depth + 1); split(x0, c, x1, y1, depth + 1)


split(0, 0, W, H)
mx = np.zeros((H, W), np.float32); mn = np.zeros((H, W), np.float32)
tones = [(215, 48), (200, 55), (110, 45), (125, 50)]           # розовое, розовое, тёмное, тёмное
for (x0, y0, x1, y1) in rects:
    a, b = tones[rng.choice(4, p=[.35, .3, .2, .15])]
    mx[y0:y1, x0:x1] = a + rng.normal(0, 3); mn[y0:y1, x0:x1] = b + rng.normal(0, 2)
# границы между полями
for (x0, y0, x1, y1) in rects:
    for side in range(4):
        r = rng.random()
        if r < .15: continue
        dm, dn = (-8, 10) if r < .75 else (-10, 60)
        sl = [(slice(y0, y1), slice(x0, x0 + 3)), (slice(y0, y1), slice(x1 - 3, x1)), (slice(y0, y0 + 3), slice(x0, x1)), (slice(y1 - 3, y1), slice(x0, x1))][side]
        mx[sl] += dm; mn[sl] += dn
mx += rng.normal(0, 4, mx.shape); mn += rng.normal(0, 3, mn.shape)
mx, mn = np.clip(mx, 1, 255), np.clip(mn, 1, 255)
diff = np.clip(mx - mn, 1, 255)
arr = np.stack([mx, mn, diff, np.full((H, W), 255)]).astype(np.uint8)
with rasterio.open(out, 'w', driver='GTiff', height=H, width=W, count=4, dtype='uint8', crs='EPSG:4326', transform=T, compress='deflate', tiled=True) as dst:
    dst.write(arr)
feats = [{'type': 'Feature', 'properties': {'i': i}, 'geometry': {'type': 'Polygon', 'coordinates': [[(T * (x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]]}}
         for i, (x0, y0, x1, y1) in enumerate(rects)]
json.dump({'type': 'FeatureCollection', 'features': feats}, open(gt, 'w'))
print(len(rects), 'полей; медиана %.0f га' % (np.median([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects]) * 56.5 / 1e4), out)
