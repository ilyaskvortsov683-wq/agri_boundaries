"""Стенд: карты многодатных границ из помесячных медиан NDVI (Sentinel-2 2026) → GeoTIFF uint8 на сетке композита.
python3 edges_test.py <s2don_dir> <don.tif> <out_dir>
Варианты: med — медиана по месяцам модуля градиента NDVI; max — максимум; std — градиент временного СКО NDVI;
doy — градиент даты максимума NDVI; comb — максимум из med и нормированных std/doy."""
import sys, numpy as np, rasterio
from scipy.ndimage import gaussian_filter, sobel
d, tif, out = sys.argv[1:4]
mon = np.load(f'{d}/ndvi_monthly.npy').astype(np.float32); months = np.load(f'{d}/months.npy')
std = np.load(f'{d}/ndvi_std.npy').astype(np.float32); doy = np.load(f'{d}/ndvi_doymax.npy').astype(np.float32)
print('месяцы', months.tolist(), 'доля NaN по месяцам', [round(float(np.isnan(m).mean()), 3) for m in mon])
def grad(a, s=1.0):
    a = np.where(np.isnan(a), np.nanmedian(a), a); a = gaussian_filter(a, s)
    return np.hypot(sobel(a, 0), sobel(a, 1))
G = np.stack([grad(m) for m in mon if np.isnan(m).mean() < 0.3])
maps = {'med': np.median(G, 0), 'max': G.max(0), 'std': grad(std), 'doy': grad(doy / 365.0)}
norm = lambda a: a / (np.percentile(a, 99.5) + 1e-9)
maps['comb'] = np.maximum.reduce([norm(maps['med']), norm(maps['std']), 0.5 * norm(maps['doy'])])
src = rasterio.open(tif); prof = dict(driver='GTiff', width=src.width, height=src.height, count=1, dtype='uint8',
                                     crs=src.crs, transform=src.transform, compress='deflate', nodata=0)
for k, a in maps.items():
    a = np.clip(norm(a) * 254 + 1, 1, 255).astype(np.uint8)
    with rasterio.open(f'{out}/edge_{k}.tif', 'w', **prof) as o:
        o.write(a, 1)
    print(k, 'перцентили 50/90/95/98:', np.percentile(a, [50, 90, 95, 98]).round(0).tolist())
