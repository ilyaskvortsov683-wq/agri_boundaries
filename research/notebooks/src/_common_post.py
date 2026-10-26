# %% [markdown]
# ### Постобработка (общая): доля пашни по ESA WorldCover и перенос растров на сетку композита
# - `cropland_frac(gdf)` — доля пикселей WorldCover-«пашни» (класс 40) внутри каждого полигона; читается прямо из
#   открытого хранилища ESA (AWS), GEE не нужен;
# - `filter_fields(gdf, MIN_CROP_FRAC, MIN_HA)` — как в конвейере: оставляет полигоны, где пашни ≥ 50 % и площадь ≥ MIN_HA;
# - `to_grid(src_tif, band, composite)` — переносит канал растра (например, вероятность границы) на сетку вашего композита.

# %%
import math
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from rasterio.features import rasterize


def _wc_tiles(bbox):
    x0, y0, x1, y1 = bbox
    for lat in range(int(math.floor(y0 / 3) * 3), int(math.floor(y1 / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(x0 / 3) * 3), int(math.floor(x1 / 3) * 3) + 1, 3):
            ns, ew = ('N' if lat >= 0 else 'S'), ('E' if lon >= 0 else 'W')
            yield (f'/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/'
                   f'ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map.tif')


def cropland_frac(gdf, agri=(40,)):
    """Доля пашни WorldCover 2021 в каждом полигоне (0–1). gdf — любая СК."""
    g = gdf.to_crs(4326); frac = np.zeros(len(g)); tot = np.zeros(len(g))
    if not len(g):
        return frac
    bbox = tuple(g.total_bounds)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        for url in _wc_tiles(bbox):
            try:
                s = rasterio.open(url)
            except Exception:
                continue
            with s:
                b = (max(bbox[0], s.bounds.left), max(bbox[1], s.bounds.bottom), min(bbox[2], s.bounds.right), min(bbox[3], s.bounds.top))
                if b[0] >= b[2] or b[1] >= b[3]:
                    continue
                win = from_bounds(*b, s.transform).round_offsets().round_lengths()
                a = s.read(1, window=win); t = s.window_transform(win)
            ids = rasterize([(geom, i + 1) for i, geom in enumerate(g.geometry.values)], out_shape=a.shape, transform=t,
                            fill=0, dtype='int32')
            n = len(g) + 1
            tot += np.bincount(ids.ravel(), minlength=n)[1:]
            frac += np.bincount(ids.ravel(), weights=np.isin(a, agri).ravel().astype(float), minlength=n)[1:]
    return np.where(tot > 0, frac / np.maximum(tot, 1), 0.0)


def filter_fields(gdf, min_crop_frac=0.5, min_ha=1.0):
    g = gdf.copy()
    g['area_ha'] = g.to_crs(6933).area / 1e4
    g['crop_frac'] = cropland_frac(g)
    keep = (g.crop_frac >= min_crop_frac) & (g.area_ha >= min_ha)
    print(f'фильтр: осталось {keep.sum()} из {len(g)} (пашня ≥ {min_crop_frac:.0%}, площадь ≥ {min_ha} га)')
    return g[keep].reset_index(drop=True)


def to_grid(src_tif, band, composite, resampling=Resampling.max):
    """Канал растра → массив на сетке композита (float32, NaN — нет данных)."""
    with rasterio.open(composite) as c:
        out = np.full((c.height, c.width), np.nan, np.float32); T, crs = c.transform, c.crs
    with rasterio.open(src_tif) as s:
        reproject(rasterio.band(s, band), out, dst_transform=T, dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    return out
