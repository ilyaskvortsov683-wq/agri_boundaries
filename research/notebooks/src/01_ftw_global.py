# %% [markdown]
# # 01 · Открытая глобальная карта полей Fields of The World (FTW), 2-е издание
#
# **Что это.** Готовые полигоны полей, которые команда FTW посчитала нейросетью по Sentinel-2 для всего мира
# за **2017–2025 гг.** (≈1,24 млрд полигонов, лицензия **CC-BY-4.0**), и растры вероятностей «поле»/«граница»
# с шагом 2,5 м. Ничего обучать и запускать на GPU не нужно — данные просто вырезаются по вашему участку.
# Источник: <https://source.coop/ftw/global-data-2e>, интерактивная карта:
# <https://research.taylorgeospatial.org/global-ftw-2e/web/>.
#
# **Что делает ноутбук.**
# 1. Вырезает полигоны FTW по вашему участку за выбранные годы → GeoPackage на Диске (слой на каждый год).
# 2. (по желанию) Пересчитывает вероятности «поле»/«граница» на сетку вашего композита → GeoTIFF.
#    Этот GeoTIFF подаётся в конвейер v7 как дополнительные барьеры (гибрид: на Донбассе +0,04 IoU к v6).
# 3. (по желанию) Сравнивает FTW с вашей разметкой теми же метриками, что и конвейер, и рисует картинки.
#
# **Важно про смысл данных.** Поле FTW — «поле в смысле ДЗЗ» (связная область, похожая на пашню), а не кадастр.
# Модель обучена на 24 странах без России; пар/залежь в обучающей разметке отсутствовали, поэтому
# залежь FTW часто пропускает. Результат на Донбассе (тест, 301 эталонное поле):
#
# | слой | IoU | IoU≥0.7 | bF (границы) | охват эталона |
# |---|---|---|---|---|
# | FTW 2025 | 0,34 | 0,21 | 0,45 | 0,52 |
# | FTW 2022 (лучший год) | 0,50 | 0,34 | 0,70 | 0,81 |
# | ваш конвейер v6 | 0,64 | 0,55 | 0,71 | 0,93 |
# | **v6 + границы FTW 2022 как барьеры** | **0,68** | **0,60** | **0,73** | 0,92 |
#
# **Время:** вырезка полигонов — секунды на небольшой участок, минуты на область; GPU не нужен
# (Среда выполнения → Сменить среду → CPU).

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры — заполните под себя
# Участок задаётся **одним** из способов (используется первый заполненный):
# - `COMPOSITE_PATH` — ваш композит (охват берётся из растра; на его сетку пишутся вероятности);
# - `AOI_PATH` — граница участка (geojson/gpkg/shp);
# - `BBOX` — рамка `(долгота_мин, широта_мин, долгота_макс, широта_макс)`.
#
# Пути — относительно корня «Мой диск». Годы — от 2017 до 2025 (2026 появится, когда FTW его выпустит).

# %%
COMPOSITE_PATH = 'Colab_Notebooks/Agro_terra/TAT/TA_composite_2026_0504_0910_8b.tif'
AOI_PATH = None                     # например 'Colab_Notebooks/Agro_terra/TAT/TAT.geojson'
BBOX = None                         # например (38.12, 47.76, 38.34, 47.93)
YEARS = [2022, 2023, 2024, 2025]    # за какие годы вырезать полигоны
MIN_SCORE = 0                       # отсечь полигоны с уверенностью модели ниже (0–100); 0 — оставить все

SAVE_PROBABILITY = True             # вероятности «поле»/«граница» на сетку композита (нужен COMPOSITE_PATH)
PROB_YEARS = [2022, 2023, 2025]     # по каким годам собирать границы (берётся максимум по годам)
PROB_BLOCK = 4096                   # размер блока в пикселях при записи большого растра

GT_PATH = None                      # ваша разметка для сравнения, например 'Colab_Notebooks/Agro_terra/test/gt_pieces.gpkg'
GT_TYPE_COL = 'type'                # колонка типа использования в разметке (если есть)

OUT_DIR = 'Colab_Notebooks/Agro_terra/research/ftw_global/out'
NAME = 'TAT'                        # префикс имён выходных файлов

# %% [markdown]
# ## Шаг 2. Установка библиотек
# В Colab почти всё уже есть; ячейка доустанавливает недостающее (≈30 с). Перезапуск среды не нужен.

# %%
#!pip -q install pyarrow pyogrio folium mapclassify > /dev/null
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio
import pyarrow.parquet as pq, pyarrow.fs as pfs
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
print('Результаты будут в', OUT)

# %% [markdown]
# ## Шаг 3. Функции чтения FTW
# Выполните ячейку без изменений. Файлы FTW лежат в открытом хранилище Source Cooperative (AWS, us-west-2) и
# читаются кусками: из многогигабайтного файла зоны UTM подтягиваются только группы строк, попадающие в участок.

# %%
FTW_BUCKET = 'us-west-2.opendata.source.coop/ftw/global-data-2e'
FTW_HTTP = 'https://data.source.coop/ftw/global-data-2e'
_fs = pfs.S3FileSystem(anonymous=True, region='us-west-2')


def utm_zones(bbox):
    x0, _, x1, _ = bbox
    return sorted({int((x + 180) // 6) + 1 for x in np.linspace(x0, x1, 16)})


def ftw_fields(bbox, year, min_score=0):
    """Полигоны FTW в рамке bbox (lon/lat) за год → GeoDataFrame EPSG:4326."""
    x0, y0, x1, y1 = bbox; parts = []
    for z in utm_zones(bbox):
        f = pq.ParquetFile(f'{FTW_BUCKET}/vector/{year}/zone={z:02d}/utm{z:02d}.parquet', filesystem=_fs)
        md = f.metadata; ix = {md.schema.column(i).path: i for i in range(md.num_columns)}
        sel = []
        for g in range(md.num_row_groups):
            rg = md.row_group(g); st = {k: rg.column(ix[f'bbox.{k}']).statistics for k in ('xmin', 'ymin', 'xmax', 'ymax')}
            if st['xmin'].min <= x1 and st['xmax'].max >= x0 and st['ymin'].min <= y1 and st['ymax'].max >= y0:
                sel.append(g)
        if not sel:
            continue
        df = f.read_row_groups(sel).to_pandas(); bb = df.pop('bbox')
        m = np.array([b['xmin'] <= x1 and b['xmax'] >= x0 and b['ymin'] <= y1 and b['ymax'] >= y0 for b in bb])
        df = df[m]
        parts.append(gpd.GeoDataFrame(df.drop(columns='geometry'), geometry=shapely.from_wkb(df.geometry.values), crs=4326))
    if not parts:
        return gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=4326)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    g = g[g.intersects(shapely.box(*bbox)) & (g['score'] >= min_score)].reset_index(drop=True)
    g['area_ha'] = g['metrics:area'] / 1e4; g['year'] = year
    g['determination:datetime'] = g['determination:datetime'].astype(str)
    return g


_RIDX = None
def ftw_tiles(bbox, year):
    global _RIDX
    if _RIDX is None:
        _RIDX = pq.read_table(f'{FTW_BUCKET}/index/raster.parquet', filesystem=_fs,
                              columns=['year', 'tile_key', 'href', 'xmin', 'ymin', 'xmax', 'ymax']).to_pandas()
    t = _RIDX; x0, y0, x1, y1 = bbox
    return t[(t.year == year) & (t.xmin <= x1) & (t.xmax >= x0) & (t.ymin <= y1) & (t.ymax >= y0)]


GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES', GDAL_HTTP_MAX_RETRY='4')


def ftw_probability(bbox, year, dst_transform, dst_shape, dst_crs):
    """Вероятности «поле» (среднее) и «граница» (максимум) FTW на заданной сетке, 0–1; NaN — нет данных."""
    field = np.full(dst_shape, np.nan, np.float32); bnd = np.full(dst_shape, np.nan, np.float32)
    for _, r in ftw_tiles(bbox, year).iterrows():
        with rasterio.Env(**GDAL_ENV), rasterio.open('/vsicurl/' + r.href.replace(FTW_HTTP, FTW_BASE_URL)) as s:
            for band, out, rs in ((1, field, Resampling.average), (2, bnd, Resampling.max)):
                tmp = np.full(dst_shape, np.nan, np.float32)
                reproject(rasterio.band(s, band), tmp, src_nodata=None, dst_nodata=np.nan,
                          dst_transform=dst_transform, dst_crs=dst_crs, resampling=rs)
                ok = ~np.isnan(tmp); out[ok] = np.where(np.isnan(out[ok]), tmp[ok], np.fmax(out[ok], tmp[ok]))
    return field / 255.0, bnd / 255.0


FTW_BASE_URL = FTW_HTTP          # если data.source.coop недоступен: 'https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/ftw/global-data-2e'
print('ok')

# %% [markdown]
# ## Шаг 4. Определяем участок

# %%
if COMPOSITE_PATH:
    with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as s:
        GRID = dict(transform=s.transform, shape=(s.height, s.width), crs=s.crs)
        AOI_BBOX = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
    aoi_geom = None
elif AOI_PATH:
    aoi = gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326); aoi_geom = aoi.union_all()
    AOI_BBOX = tuple(aoi.total_bounds); GRID = None
else:
    AOI_BBOX = tuple(BBOX); aoi_geom = None; GRID = None
w_km = (AOI_BBOX[2] - AOI_BBOX[0]) * 111 * np.cos(np.radians((AOI_BBOX[1] + AOI_BBOX[3]) / 2)); h_km = (AOI_BBOX[3] - AOI_BBOX[1]) * 111
print(f'Рамка участка: {np.round(AOI_BBOX, 4)} ≈ {w_km:.0f} × {h_km:.0f} км; зоны UTM: {utm_zones(AOI_BBOX)}')

# %% [markdown]
# ## Шаг 5. Вырезаем полигоны FTW за выбранные годы
# Для каждой зоны UTM читается только нужная часть файла. Результат — `<NAME>_ftw_fields.gpkg`, по слою на год
# (`y2022`, `y2023`, …). Колонки: `score` — средняя уверенность модели внутри полигона (0–100; это рейтинг,
# а не вероятность), `area_ha` — площадь, `year`.

# %%
FTW = {}
gpkg = OUT / f'{NAME}_ftw_fields.gpkg'
for y in YEARS:
    t0 = time.time(); g = ftw_fields(AOI_BBOX, y, MIN_SCORE)
    if aoi_geom is not None:
        g = g[g.intersects(aoi_geom)].reset_index(drop=True)
    FTW[y] = g
    g.to_file(gpkg, layer=f'y{y}', driver='GPKG')
    print(f'{y}: {len(g):>7} полигонов, {g.area_ha.sum():>10.0f} га, медиана {g.area_ha.median() if len(g) else 0:.1f} га  ({time.time()-t0:.0f} с)')
print('Сохранено:', gpkg)

# %% [markdown]
# ## Шаг 6 (по желанию). Вероятности «поле»/«граница» на сетку вашего композита
# Пишется GeoTIFF `<NAME>_ftw_prob.tif` той же сетки, что композит (uint8, 0–255):
# - канал 1 — **граница**: максимум вероятности границы по годам `PROB_YEARS`;
# - канал 2 — **устойчивость границы**: в скольких годах вероятность границы ≥ 0,5;
# - канал 3 — **поле**: средняя вероятность «поле» по годам.
#
# Этот файл указывается в конвейере v7 в параметре `EXT_BOUNDARY['TIF']` — тогда границы FTW станут
# дополнительными барьерами при слиянии суперпикселей. Большой композит пишется блоками, это может занять
# 10–40 минут на область (работа идёт по сети, GPU не нужен).

# %%
if SAVE_PROBABILITY and GRID is not None:
    out_tif = OUT / f'{NAME}_ftw_prob.tif'
    prof = dict(driver='GTiff', width=GRID['shape'][1], height=GRID['shape'][0], count=3, dtype='uint8',
                crs=GRID['crs'], transform=GRID['transform'], compress='deflate', tiled=True, blockxsize=512,
                blockysize=512, BIGTIFF='IF_SAFER', nodata=0)
    H, W = GRID['shape']; nb = ((H + PROB_BLOCK - 1) // PROB_BLOCK) * ((W + PROB_BLOCK - 1) // PROB_BLOCK); k = 0
    with rasterio.open(out_tif, 'w', **prof) as dst:
        for r0 in range(0, H, PROB_BLOCK):
            for c0 in range(0, W, PROB_BLOCK):
                win = Window(c0, r0, min(PROB_BLOCK, W - c0), min(PROB_BLOCK, H - r0)); k += 1
                wt = rasterio.windows.transform(win, GRID['transform']); shp = (int(win.height), int(win.width))
                wb = transform_bounds(GRID['crs'], 'EPSG:4326', *rasterio.windows.bounds(win, GRID['transform']))
                bmax = np.zeros(shp, np.float32); bcnt = np.zeros(shp, np.uint8); fsum = np.zeros(shp, np.float32); fn = 0
                for y in PROB_YEARS:
                    f, b = ftw_probability(wb, y, wt, shp, GRID['crs'])
                    if np.isnan(f).all():
                        continue
                    bmax = np.fmax(bmax, np.nan_to_num(b)); bcnt += (np.nan_to_num(b) >= .5).astype(np.uint8)
                    fsum += np.nan_to_num(f); fn += 1
                out = np.stack([np.clip(bmax * 254 + 1, 1, 255), bcnt + 1, np.clip(fsum / max(fn, 1) * 254 + 1, 1, 255)]).astype(np.uint8)
                dst.write(out, window=win)
                print(f'\rблок {k}/{nb}', end='')
    print('\nСохранено:', out_tif, '(значения 1–255; 0 — нет данных)')
else:
    print('Пропущено (нет композита или SAVE_PROBABILITY=False)')

# %% include _common_eval.py

# %% [markdown]
# ## Шаг 7 (по желанию). Сравнение с вашей разметкой
# Нужен `GT_PATH`. Если указан композит — метрики считаются на его сетке, иначе на сетке 10 м по разметке.

# %%
if GT_PATH:
    gt = load_gt(DRIVE_ROOT / GT_PATH, AOI_BBOX)
    gt_bbox = tuple(gt.to_crs(4326).total_bounds)
    rows = []
    for y, g in FTW.items():
        gg = g[g.intersects(shapely.box(*gt_bbox))]
        kw = dict(transform=GRID['transform'], shape=GRID['shape'], crs=GRID['crs']) if GRID else {}
        rows.append(evaluate(gg, gt, type_col=GT_TYPE_COL, label=f'FTW {y}', **kw))
    pd.DataFrame(rows).to_csv(OUT / f'{NAME}_ftw_vs_gt.csv', index=False)
    display(pd.DataFrame(rows))
else:
    print('GT_PATH не задан — сравнение пропущено')

# %% [markdown]
# ## Шаг 8. Картинка: композит + полигоны FTW (+ эталон)
# Показывается фрагмент `VIEW_BBOX` (по умолчанию — охват разметки или центр участка 10×10 км).

# %%
VIEW_BBOX = None          # (lon_min, lat_min, lon_max, lat_max) или None
if VIEW_BBOX is None:
    gb = tuple(gt.to_crs(4326).total_bounds) if GT_PATH and len(gt) else None
    if gb and gb[2] - gb[0] < 0.4 and gb[3] - gb[1] < 0.3:        # разметка на небольшой участок — показываем её охват
        VIEW_BBOX = gb
    else:
        cx, cy = (AOI_BBOX[0] + AOI_BBOX[2]) / 2, (AOI_BBOX[1] + AOI_BBOX[3]) / 2
        VIEW_BBOX = (cx - .07, cy - .045, cx + .07, cy + .045)
if COMPOSITE_PATH:
    vb = transform_bounds('EPSG:4326', GRID['crs'], *VIEW_BBOX)
    colors = ['cyan', 'white', 'lime', 'orange', 'yellow']
    lays = [(g[g.intersects(shapely.box(*VIEW_BBOX))], colors[i % 5], f'FTW {y}') for i, (y, g) in enumerate(FTW.items())][-2:]
    if GT_PATH:
        lays.append((gt[gt.to_crs(4326).intersects(shapely.box(*VIEW_BBOX))], 'red', 'эталон'))
    show_overlay(DRIVE_ROOT / COMPOSITE_PATH, lays, title=f'{NAME}: FTW', bbox=vb, out_png=OUT / f'{NAME}_ftw_view.png')

# %% [markdown]
# ## Шаг 9 (по желанию). Интерактивная карта поверх спутниковой подложки
# Показывает последний год из `YEARS` в окне `VIEW_BBOX` (не более 20 000 полигонов, чтобы не тормозило).

# %%
import folium
y = YEARS[-1]; g = FTW[y]; g = g[g.intersects(shapely.box(*VIEW_BBOX))].head(20000)
m = folium.Map(location=[(VIEW_BBOX[1] + VIEW_BBOX[3]) / 2, (VIEW_BBOX[0] + VIEW_BBOX[2]) / 2], zoom_start=13,
               tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')
folium.GeoJson(g[['score', 'area_ha', 'geometry']], name=f'FTW {y}',
               style_function=lambda f: {'color': 'yellow', 'weight': 1, 'fillOpacity': 0},
               tooltip=folium.GeoJsonTooltip(['score', 'area_ha'])).add_to(m)
if GT_PATH:
    folium.GeoJson(gt.to_crs(4326)[lambda d: d.intersects(shapely.box(*VIEW_BBOX))], name='эталон',
                   style_function=lambda f: {'color': 'red', 'weight': 1, 'fillOpacity': 0}).add_to(m)
folium.LayerControl().add_to(m)
m

# %% [markdown]
# ## Что дальше
# - Посмотрите, в какие годы FTW лучше видит ваши поля (шаг 7): на Донбассе лучшим оказался 2022, а 2024–2025 —
#   заметно хуже (много пропусков на залежи и части посевов).
# - Файл `<NAME>_ftw_prob.tif` из шага 6 подключается к конвейеру v7 (`EXT_BOUNDARY['TIF']`), см. инструкцию в папке
#   `research/classical_v7`.
# - Полигоны FTW можно использовать как готовый слой «полей» в QGIS или как псевдо-разметку для дообучения
#   (ноутбук 04) — но только после проверки глазами.
