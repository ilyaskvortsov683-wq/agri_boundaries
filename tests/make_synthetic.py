"""Синтетический композит (EPSG:4326, пиксель ≈ 6×10 м): розовые/тёмные/«жёлтые» поля, лесополосы, пятно.
python3 tests/make_synthetic.py out.tif"""
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin

H, W = 1000, 1200
rng = np.random.default_rng(1)
mx = np.full((H, W), 200, np.float32); mn = np.full((H, W), 50, np.float32)
# сетка полей 3×3 с межами 6 px
ys = np.linspace(0, H, 4).astype(int); xs = np.linspace(0, W, 5).astype(int)
for i in range(3):
    for j in range(4):
        v = (i * 4 + j) % 4
        a, b, c, d = ys[i] + 6, ys[i + 1] - 6, xs[j] + 6, xs[j + 1] - 6
        if v == 0: mx[a:b, c:d], mn[a:b, c:d] = 215, 45          # розовое
        elif v == 1: mx[a:b, c:d], mn[a:b, c:d] = 95, 40         # тёмное
        elif v == 2: mx[a:b, c:d], mn[a:b, c:d] = 140, 50        # пар
        else: mx[a:b, c:d], mn[a:b, c:d] = 170, 110              # «жёлтое»
mx[120:170, 100:160], mn[120:170, 100:160] = 90, 55               # пятно в розовом поле
# лесополоса внутри одного поля (слипшиеся)
mn[ys[1] + 6:ys[2] - 6, 640:644] = 130
mx += rng.normal(0, 4, mx.shape); mn += rng.normal(0, 3, mn.shape)
mx, mn = np.clip(mx, 1, 255), np.clip(mn, 1, 255)
diff = np.clip(mx - mn, 1, 255)
arr = np.stack([mx, mn, diff]).astype(np.uint8)
t = from_origin(50.0, 53.5, 0.0001, 0.00009)
with rasterio.open(sys.argv[1], 'w', driver='GTiff', height=H, width=W, count=3, dtype='uint8', crs='EPSG:4326', transform=t) as dst:
    dst.write(arr)
print('ok', sys.argv[1])
