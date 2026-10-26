"""Локальная выборка Sentinel-2 L2A из открытого архива AWS (s3://sentinel-cogs, Element84) — для проверок вне GEE.
В Colab то же самое делает GEE-скрипт (см. gee_monthly_export.js / ноутбук), здесь — только стенд."""
import json, re, urllib.request, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://sentinel-cogs.s3.us-west-2.amazonaws.com'
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='4',
           GDAL_HTTP_RETRY_DELAY='2', VSI_CACHE='TRUE')


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def list_scenes(tile, year, months):
    """tile — MGRS, напр. '37TDN'. → [(дата, облачность %, item_json_url)]"""
    zone, band, sq = tile[:2], tile[2], tile[3:]
    out = []
    for m in months:
        pre = f'sentinel-s2-l2a-cogs/{int(zone)}/{band}/{sq}/{year}/{m}/'
        xml = _get(f'{BASE}/?list-type=2&prefix={pre}&delimiter=/')
        for p in re.findall(r'<Prefix>(' + re.escape(pre) + r'[^<]+/)</Prefix>', xml):
            name = p.rstrip('/').split('/')[-1]
            out.append((name, f'{BASE}/{p}{name}.json'))
    res = []
    def one(t):
        name, url = t
        try:
            it = json.loads(_get(url))
        except Exception:
            return None
        return dict(name=name, date=it['properties']['datetime'][:10], cloud=it['properties'].get('eo:cloud_cover', 100),
                    assets={k: v['href'] for k, v in it['assets'].items()})
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, out):
            if r: res.append(r)
    return sorted(res, key=lambda r: r['date'])


def read_to_grid(href, dst_transform, dst_shape, dst_crs, resampling=Resampling.bilinear):
    out = np.full(dst_shape, np.nan, np.float32)
    with rasterio.Env(**ENV), rasterio.open('/vsicurl/' + href) as s:
        reproject(rasterio.band(s, 1), out, src_nodata=0, dst_nodata=np.nan, dst_transform=dst_transform,
                  dst_crs=dst_crs, resampling=resampling)
    return out


def scene_ndvi(sc, dst_transform, dst_shape, dst_crs, bands=('red', 'nir')):
    """NDVI сцены на заданной сетке, облака/тени/снег по SCL → NaN. Возвращает (ndvi, dict каналов)."""
    a = sc['assets']
    scl = read_to_grid(a['scl'], dst_transform, dst_shape, dst_crs, Resampling.nearest)
    bad = np.isnan(scl) | np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    ch = {b: read_to_grid(a[b], dst_transform, dst_shape, dst_crs) for b in bands}
    for b in ch: ch[b][bad] = np.nan
    red, nir = ch['red'], ch['nir']
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi, ch
