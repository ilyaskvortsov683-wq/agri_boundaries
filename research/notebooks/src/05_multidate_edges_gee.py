# %% [markdown]
# # 05 · Многодатные слои из Sentinel-2 (GEE): границы по месяцам, СКО NDVI, дата максимума
#
# **Зачем.** Ваш композит — одна «сводка» сезона (max/min/diff NDVI). Соседние поля с одинаковым пиком NDVI, но разной
# культурой или датой сева в ней почти неразличимы, а в отдельные месяцы граница между ними видна. Все удачные
# классические методы (CEWS, SACFEI, North и др.) сначала ищут границы **на каждой дате**, а потом собирают их по
# времени. Этот ноутбук делает то же в GEE и сохраняет результат **на сетке вашего композита**.
#
# **Что получится** — GeoTIFF `<NAME>_multidate.tif` (uint8, та же сетка, что композит):
# 1. **край-максимум** — максимум по месяцам модуля градиента месячной медианы NDVI (главный слой для конвейера v7);
# 2. **край-медиана** — медиана того же по месяцам (устойчивые границы);
# 3. **СКО NDVI** за сезон ×637 (0,4 → 255) — «насколько поле менялось»;
# 4. **дата максимума NDVI** (день года / 2).
#
# Канал 1 подключается к конвейеру v7: `EXT_TIF = '<путь>/<NAME>_multidate.tif'`, `EXT_BAND = 1`, `EXT_THR = 0.75`.
#
# **Чего ждать (Донбасс, 2026, тест на 301 поле).** Этот слой сам по себе даёт небольшой прирост:
# IoU 0,635 → 0,647 (доля IoU≥0,7: 0,55 → 0,57). Границы готовой нейросети FTW дают больше (0,670).
# Многодатный слой полезен там, где FTW нет (2026 год) или он плохо видит ваши поля, и как вход для дообучения.
#
# Дополнительно (по желанию) можно выгрузить **сам стек месячных NDVI** (uint8, канал на месяц) — для своих
# экспериментов и для небольших участков (для области целиком это десятки ГБ).
#
# **Время:** расчёт идёт на серверах GEE; экспорт области — 20–90 минут. GPU не нужен.

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры

# %%
EE_PROJECT = 'my-project-20260614-499410'
COMPOSITE_PATH = 'Colab_Notebooks/Agro_terra/TAT/TA_composite_2026_0504_0910_8b.tif'   # сетка результата = сетка композита
AOI_PATH = None                      # необязательно: граница области (иначе — весь охват композита)
YEAR = 2026
MONTHS = [4, 5, 6, 7, 8, 9]          # какие месяцы брать (для юга можно с 3-го по 10-й, для севера — 5–9)
SMOOTH_PX = 1.0                      # сглаживание месячного NDVI перед градиентом, пикс. (0 — без)
EXPORT_MONTHLY_STACK = False         # True — дополнительно выгрузить стек NDVI по месяцам (только для небольших участков!)
OUT_DIR = 'Colab_Notebooks/Agro_terra/research/multidate/out'
GEE_FOLDER = 'GEE_multidate'
NAME = 'TAT_2026'

# %% include _common_gee.py

# %% [markdown]
# ## Шаг 2. Сетка композита
# Экспорт идёт с той же проекцией и тем же шагом/началом пикселей (`crsTransform`), что у композита, поэтому
# пиксели результата совпадают с пикселями композита один в один.

# %%
import numpy as np, rasterio
from rasterio.warp import transform_bounds
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as s:
    C_CRS = s.crs.to_string(); C_T = list(s.transform)[:6]; C_W, C_H = s.width, s.height
    C_BOUNDS = s.bounds; AOI_BBOX = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
aoi = ee.Geometry.Rectangle(list(AOI_BBOX))
if AOI_PATH:
    import geopandas as gpd
    aoi = ee.Geometry(json.loads(gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326).dissolve().geometry.to_json())['features'][0]['geometry'])
UTM = f'EPSG:{utm_epsg((AOI_BBOX[0] + AOI_BBOX[2]) / 2, (AOI_BBOX[1] + AOI_BBOX[3]) / 2)}'
print('композит', C_W, '×', C_H, 'пикс.,', C_CRS, '| расчёт градиентов в', UTM)

# %% [markdown]
# ## Шаг 3. Расчёт слоёв в GEE
# Для каждого месяца: медиана NDVI безоблачных пикселей (Cloud Score+) → дыры заполняются сезонной медианой →
# лёгкое сглаживание → градиент в UTM 10 м → модуль. Затем максимум и медиана по месяцам. Нормировка в 0–255 — по
# 99,5-му перцентилю по участку (считается в GEE на грубой сетке).

# %%
season = s2_collection(aoi, f'{YEAR}-{MONTHS[0]:02d}-01', ee.Date.fromYMD(YEAR, MONTHS[-1], 1).advance(1, 'month'))
ndvi_col = season.map(lambda im: im.normalizedDifference(['B8', 'B4']).rename('ndvi')
                      .addBands(ee.Image.constant(ee.Date(im.get('system:time_start')).getRelative('day', 'year')).rename('doy').toInt16())
                      .copyProperties(im, ['system:time_start']))
season_med = ndvi_col.select('ndvi').median()
grads, months_nd = [], []
for m in MONTHS:
    st = ee.Date.fromYMD(YEAR, m, 1)
    nd = ndvi_col.filterDate(st, st.advance(1, 'month')).select('ndvi').median().unmask(season_med)
    months_nd.append(nd.rename(f'ndvi_{m:02d}'))
    nd10 = nd.reproject(crs=UTM, scale=10)
    if SMOOTH_PX:
        nd10 = nd10.convolve(ee.Kernel.gaussian(radius=SMOOTH_PX * 2, sigma=SMOOTH_PX, units='pixels'))
    g = nd10.gradient()
    grads.append(g.select('x').hypot(g.select('y')).rename('g'))
G = ee.ImageCollection(grads)
e_max, e_med = G.max().rename('edge_max'), G.median().rename('edge_med')
nd_std = ndvi_col.select('ndvi').reduce(ee.Reducer.stdDev()).rename('ndvi_std')
doy_max = ndvi_col.qualityMosaic('ndvi').select('doy').rename('doy_max')
p = ee.Image.cat([e_max, e_med]).reduceRegion(ee.Reducer.percentile([99.5]), aoi, scale=100, bestEffort=True,
                                              maxPixels=1e9).getInfo()
print('99,5-й перцентиль градиента:', p)
k_max, k_med = 254 / max(p['edge_max'], 1e-9), 254 / max(p['edge_med'], 1e-9)
out_img = ee.Image.cat([e_max.multiply(k_max).add(1), e_med.multiply(k_med).add(1),
                        nd_std.multiply(637).add(1), doy_max.divide(2).add(1)]).clamp(1, 255).toUint8()

# %% [markdown]
# ## Шаг 4. Экспорт на Диск и сборка в один файл
# Экспорт идёт одним файлом на сетке композита (GEE сам режет большие файлы на части; они склеиваются ниже).
# Ячейка ждёт окончания; её можно прервать и запустить снова — повторно файл не экспортируется, если уже есть.

# %%
final = OUT / f'{NAME}_multidate.tif'
if not final.exists():
    region = ee.Geometry.Rectangle([C_BOUNDS.left, C_BOUNDS.bottom, C_BOUNDS.right, C_BOUNDS.top], C_CRS, False)
    task = ee.batch.Export.image.toDrive(image=out_img, description=f'{NAME}_multidate', folder=GEE_FOLDER,
                                         fileNamePrefix=f'{NAME}_multidate', region=region, crs=C_CRS, crsTransform=C_T,
                                         maxPixels=1e13, fileDimensions=32768)
    task.start(); jobs = [('all', AOI_BBOX, f'{NAME}_multidate', task)]
    wait_tasks(jobs); collect_exports(jobs, GEE_FOLDER, OUT / 'parts')
    parts_ = sorted((OUT / 'parts').glob(f'{NAME}_multidate*.tif'))
    prof = dict(driver='GTiff', width=C_W, height=C_H, count=4, dtype='uint8', crs=C_CRS,
                transform=rasterio.Affine(*C_T), compress='deflate', tiled=True, BIGTIFF='IF_SAFER', nodata=0)
    with rasterio.open(final, 'w', **prof) as dst:
        for pth in parts_:
            with rasterio.open(pth) as srcp:
                win = rasterio.windows.from_bounds(*srcp.bounds, dst.transform).round_offsets().round_lengths()
                dst.write(srcp.read(window=rasterio.windows.Window(0, 0, win.width, win.height)), window=win)
    print('Сохранено:', final, '| частей:', len(parts_))
else:
    print('Уже есть:', final)

# %% [markdown]
# ## Шаг 5 (по желанию). Стек месячных NDVI
# Канал на месяц, uint8 (NDVI −0,2…1 → 1…255). Только для небольших участков: для области целиком файл огромный.

# %%
if EXPORT_MONTHLY_STACK:
    stack = ee.Image.cat(months_nd).add(0.2).multiply(254 / 1.2).add(1).clamp(1, 255).toUint8()
    region = ee.Geometry.Rectangle([C_BOUNDS.left, C_BOUNDS.bottom, C_BOUNDS.right, C_BOUNDS.top], C_CRS, False)
    t = ee.batch.Export.image.toDrive(image=stack, description=f'{NAME}_ndvi_monthly', folder=GEE_FOLDER,
                                      fileNamePrefix=f'{NAME}_ndvi_monthly', region=region, crs=C_CRS, crsTransform=C_T,
                                      maxPixels=1e13, fileDimensions=32768)
    t.start(); jobs = [('m', AOI_BBOX, f'{NAME}_ndvi_monthly', t)]
    wait_tasks(jobs); collect_exports(jobs, GEE_FOLDER, OUT)

# %% [markdown]
# ## Шаг 6. Быстрый просмотр
# Фрагмент 10×10 км в центре: композит и край-максимум рядом.

# %%
import matplotlib.pyplot as plt
with rasterio.open(final) as s, rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
    cy, cx = s.height // 2, s.width // 2; w = rasterio.windows.Window(cx - 700, cy - 700, 1400, 1400)
    e = s.read(1, window=w); comp = c.read([1, 2, 3], window=w).transpose(1, 2, 0)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(comp); ax[0].set_title('композит'); ax[1].imshow(e, cmap='magma', vmin=0, vmax=255); ax[1].set_title('край-максимум по месяцам')
for a in ax: a.axis('off')
plt.savefig(OUT / f'{NAME}_multidate_view.png', dpi=110, bbox_inches='tight'); plt.show()

# %% [markdown]
# ## Что дальше
# В ноутбуке конвейера v7 в пресете (или в `BASE`) поставьте:
# `EXT_TIF='…/research/multidate/out/<NAME>_multidate.tif', EXT_BAND=1, EXT_THR=0.75`.
# Порог удобно подобрать автоматически ноутбуком 06 (укажите этот файл в `EXT_BOUNDARY_TIF`).
# Несколько источников границ объединяются списком, например FTW + многодатные:
# `EXT_TIF=[dict(TIF='…/TAT_ftw_prob.tif', BAND=1, THR=0.3), dict(TIF='…/TAT_2026_multidate.tif', BAND=1, THR=0.75)]`
# (на Донбассе: только FTW — IoU 0,670; FTW + многодатные — 0,671; FTW + суперпиксели Higra — 0,676).
