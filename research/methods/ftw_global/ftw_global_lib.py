"""Чтение открытой глобальной карты полей Fields of The World (2-е издание, 2017–2025, CC-BY-4.0).

Данные: https://source.coop/ftw/global-data-2e
  vector/{год}/zone=NN/utmNN.parquet  — полигоны (EPSG:4326, колонки metrics:area м², score 0–100)
  raster/{год}/.../{tile}.tif          — вероятности 2.5 м: канал 1 «поле», канал 2 «граница» (uint8, /255)
Файлы читаются кусками по HTTP/S3 (range-запросы), целиком ничего не скачивается.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.fs as pfs, geopandas as gpd, shapely

BUCKET = 'us-west-2.opendata.source.coop/ftw/global-data-2e'
HTTP = 'https://data.source.coop/ftw/global-data-2e'          # в Colab открывается; запасной: S3_HTTP
S3_HTTP = 'https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/ftw/global-data-2e'
_fs = None


def fs():
    global _fs
    if _fs is None:
        _fs = pfs.S3FileSystem(anonymous=True, region='us-west-2')
    return _fs


def utm_zones(bbox):
    x0, _, x1, _ = bbox
    return sorted({int((x + 180) // 6) + 1 for x in np.linspace(x0, x1, 8)})


def fetch_fields(bbox, year, min_score=0):
    """Полигоны FTW в рамке bbox=(lon_min, lat_min, lon_max, lat_max) за год 2017–2025 → GeoDataFrame EPSG:4326.
    Файлы отсортированы по пространству, поэтому читаются только 1–3 группы строк (секунды)."""
    x0, y0, x1, y1 = bbox
    parts = []
    for z in utm_zones(bbox):
        f = pq.ParquetFile(f'{BUCKET}/vector/{year}/zone={z:02d}/utm{z:02d}.parquet', filesystem=fs())
        md = f.metadata
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        ix = {n: i for i, n in enumerate(names)}
        sel = []
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            st = {k: rg.column(ix[f'bbox.{k}']).statistics for k in ('xmin', 'ymin', 'xmax', 'ymax')}
            if st['xmin'].min <= x1 and st['xmax'].max >= x0 and st['ymin'].min <= y1 and st['ymax'].max >= y0:
                sel.append(g)
        if not sel:
            continue
        df = f.read_row_groups(sel).to_pandas()
        bb = df.pop('bbox')
        m = np.array([b['xmin'] <= x1 and b['xmax'] >= x0 and b['ymin'] <= y1 and b['ymax'] >= y0 for b in bb])
        df = df[m]
        parts.append(gpd.GeoDataFrame(df.drop(columns='geometry'), geometry=shapely.from_wkb(df.geometry.values), crs=4326))
    if not parts:
        return gpd.GeoDataFrame(columns=['id', 'score', 'metrics:area', 'geometry'], geometry='geometry', crs=4326)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    g = g[g.intersects(shapely.box(*bbox))]
    g = g[g['score'] >= min_score].reset_index(drop=True)
    g['area_ha'] = g['metrics:area'] / 1e4
    if 'determination:datetime' in g:
        g['determination:datetime'] = g['determination:datetime'].astype(str)
    return g


def raster_tiles(bbox, year):
    """Тайлы вероятностей FTW, пересекающие bbox (таблица индекса)."""
    t = pq.read_table(f'{BUCKET}/index/raster.parquet', filesystem=fs(),
                      columns=['year', 'tile_key', 'epsg', 'href', 'field_frac', 'boundary_frac', 'cropland_frac',
                               'xmin', 'ymin', 'xmax', 'ymax']).to_pandas()
    x0, y0, x1, y1 = bbox
    m = (t.year == year) & (t.xmin <= x1) & (t.xmax >= x0) & (t.ymin <= y1) & (t.ymax >= y0)
    return t[m].reset_index(drop=True)


def fetch_probability(bbox, year, dst_transform, dst_shape, dst_crs='EPSG:4326', base=None):
    """Вероятности «поле» и «граница» (0–1) FTW, перепроецированные на сетку вашего растра.
    dst_transform/dst_shape/dst_crs — от вашего композита (rasterio .transform/.shape/.crs).
    Возвращает (field, boundary) float32; NaN там, где тайлов нет."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    base = base or HTTP
    field = np.full(dst_shape, np.nan, np.float32); bnd = np.full(dst_shape, np.nan, np.float32)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
               GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES')
    for _, r in raster_tiles(bbox, year).iterrows():
        url = '/vsicurl/' + r.href.replace(HTTP, base)
        with rasterio.Env(**env), rasterio.open(url) as s:
            for band, out in ((1, field), (2, bnd)):
                tmp = np.full(dst_shape, np.nan, np.float32)
                # 2.5 м → ваша сетка: «поле» усредняем, «граница» берём максимумом, иначе тонкие линии растворяются
                rs = Resampling.average if band == 1 else Resampling.max
                reproject(rasterio.band(s, band), tmp, src_nodata=None, dst_nodata=np.nan,
                          dst_transform=dst_transform, dst_crs=dst_crs, resampling=rs)
                ok = ~np.isnan(tmp)
                out[ok] = np.where(np.isnan(out[ok]), tmp[ok], np.fmax(out[ok], tmp[ok]))
    return field / 255.0, bnd / 255.0
