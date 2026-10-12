"""Картинка: композит + контуры предсказания (жёлтый) + эталон (белый). python3 tests/view_gt.py fields.gpkg gt_fields.gpkg out.png x0 y0 x1 y1 [tif]"""
import sys, glob, numpy as np, rasterio, geopandas as gpd, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
fl, gt, out = sys.argv[1:4]; x0, y0, x1, y1 = map(int, sys.argv[4:8])
tif = sys.argv[8] if len(sys.argv) > 8 else glob.glob('/root/.claude/uploads/*/*2.tif')[0]
s = rasterio.open(tif); T = s.transform
rgb = np.moveaxis(s.read(window=((y0, y1), (x0, x1))), 0, -1)
xa, ya = T * (x0, y0); xb, yb = T * (x1, y1)
fig, ax = plt.subplots(figsize=(12, 12 * (y1 - y0) / (x1 - x0)))
ax.imshow(rgb, extent=(xa, xb, yb, ya))
gpd.read_file(fl, layer='fields' if fl.endswith('.gpkg') else None).to_crs(4326).clip_by_rect(xa, yb, xb, ya).boundary.plot(ax=ax, color='yellow', linewidth=0.8)
if gt != '-': gpd.read_file(gt).to_crs(4326).clip_by_rect(xa, yb, xb, ya).boundary.plot(ax=ax, color='white', linewidth=0.6, linestyle='--')
ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_axis_off(); plt.tight_layout(); plt.savefig(out, dpi=75)
