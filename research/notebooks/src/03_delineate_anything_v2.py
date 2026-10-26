# %% [markdown]
# # 03 · Delineate Anything v2 — поля как отдельные объекты (YOLO11) по одному RGB-снимку
#
# **Что это.** Модель группы Лавренюка/Куссуль/Шелестова (ECCV 2026), обученная на 73 млн полей из 61 страны,
# в т.ч. по снимкам Sentinel-2. В отличие от FTW она сразу выдаёт **отдельные поля-полигоны** (сегментация
# экземпляров), работает с любым разрешением 0,25–10 м и уже умеет склеивать поля через швы тайлов и
# убирать «гигантов». Авторы прогнали её на всю Украину (ближайший к нашим степям ландшафт).
# Код и веса: <https://github.com/Lavreniuk/Delineate-Anything>, лицензия **AGPL-3.0** (для внутренних и
# исследовательских задач ограничений нет; при распространении изменённого кода или работе как веб-сервис
# нужно открыть исходники).
#
# **Что делает ноутбук.**
# 1. Собирает вход: либо **медианный RGB Sentinel-2** за окно середины сезона в GEE (по умолчанию; можно несколько
#    окон — модель запустится на каждом, результаты сравните), либо **ваш композит** (3 канала max/min/diff NDVI —
#    модель на таком не обучалась, но проверить стоит: это минуты).
# 2. Запускает `delineate.py` из официального репозитория на GPU.
# 3. Склеивает блоки, фильтрует по доле пашни WorldCover, сохраняет GeoPackage и растр границ для гибрида с v7.
# 4. (по желанию) Сравнивает с вашей разметкой.
#
# **Среда:** Среда выполнения → Сменить среду → **T4 GPU**. Скорость: вся Украина ≈5 ч на одной видеокарте 16 ГБ,
# т.е. область размера Татарстана — десятки минут.
#
# **Чего ждать.** Независимая проверка (США, Sentinel-2 10 м) дала объектную F1 ≈0,34; сопровождающий пишет, что
# вне похожих на Европу территорий без дообучения результат слабее. Для наших степей и лесостепи — проверяем.

# %% [markdown]
# ## Шаг 0. Установка (один раз за сеанс, ~2 мин)
# Клонируется официальный репозиторий и ставятся ultralytics/huggingface_hub. Библиотеки Colab (rasterio, shapely,
# GDAL) не трогаются, поэтому перезапуск среды обычно не нужен.

# %%
import os, subprocess, sys, importlib.util
DA_DIR = '/content/Delineate-Anything' if not os.environ.get('LOCAL_TEST') else os.environ.get('DA_DIR', '/tmp/Delineate-Anything')
if not os.path.exists(DA_DIR):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/Lavreniuk/Delineate-Anything.git', DA_DIR], check=True)
need = [p for p in ('ultralytics', 'huggingface_hub', 'numba', 'yaml') if importlib.util.find_spec(p) is None]
if need:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.3.148', 'huggingface_hub', 'numba', 'pyyaml'], check=True)
try:
    from osgeo import gdal
    print('GDAL', gdal.__version__)
except ImportError:
    subprocess.run('apt-get -qq install -y python3-gdal gdal-bin > /dev/null', shell=True)
    print('GDAL установлен из apt; если ниже будет ошибка импорта osgeo — перезапустите сеанс')
import torch
print('репозиторий:', DA_DIR, '| GPU:', torch.cuda.is_available())

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры — заполните под себя
# **Источник снимка** `IMAGE_SOURCE`:
# - `'gee'` — медиана Sentinel-2 RGB (B4, B3, B2) за окна `WINDOWS` (рекомендуется: модель обучалась на RGB);
#   для каждого окна будет отдельный результат. Окно середины сезона (июнь–июль) обычно лучше всего разделяет
#   соседние поля разных культур; позднее окно (август) — хорошо видны убранные/неубранные поля.
# - `'composite'` — ваш 3-канальный композит из `COMPOSITE_PATH` (будет перепроецирован в UTM, как требует модель).

# %%
EE_PROJECT = 'my-project-20260614-499410'
IMAGE_SOURCE = 'gee'                 # 'gee' или 'composite'
COMPOSITE_PATH = 'Colab_Notebooks/Agro_terra/TAT/TA_composite_2026_0504_0910_8b.tif'   # охват участка и сетка для растра границ
AOI_PATH = None
BBOX = None                          # (lon_min, lat_min, lon_max, lat_max) — для быстрого теста на маленьком участке
WINDOWS = {'jun_jul': ('2026-06-01', '2026-07-31')}     # можно добавить, напр. 'aug': ('2026-08-01', '2026-08-31')
BLOCK_DEG = 0.5

MODEL = 'large_v2'                   # 'large_v2' — Delineate Anything v2; 'large' / 'small' — версия 1
MIN_CONF = 0.15                      # минимальная уверенность объекта (ниже — больше полей и больше мусора)
GIANT_FILTER = True                  # фильтр «гигантов»; на очень крупных полях может удалить и настоящие — сравните с False
BATCH = 8                            # тайлов за раз на GPU (T4 16 ГБ: 8; при нехватке памяти — 4)

MIN_CROP_FRAC = 0.5
MIN_HA = 1.0
GT_PATH = None
GT_TYPE_COL = 'type'
OUT_DIR = 'Colab_Notebooks/Agro_terra/research/delineate_anything/out'
GEE_FOLDER = 'GEE_delineate_anything'
NAME = 'TAT_2026'

# %% [markdown]
# ## Шаг 2. Участок и блоки

# %%
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio, yaml
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/content/da_work') if not LOCAL_TEST else Path(os.environ.get('DA_WORK', '/tmp/da_work'))
IMG_ROOT = OUT / f'{NAME}_input'; IMG_ROOT.mkdir(parents=True, exist_ok=True)
if BBOX:
    AOI_BBOX = tuple(BBOX)
elif COMPOSITE_PATH:
    with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as s:
        AOI_BBOX = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
else:
    AOI_BBOX = tuple(gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326).total_bounds)
print('рамка', np.round(AOI_BBOX, 3))

# %% [markdown]
# ## Шаг 3. Входные снимки
# **Вариант `gee`.** Для каждого окна — медиана безоблачных пикселей (Cloud Score+), каналы B4,B3,B2 переводятся
# в 8 бит как «истинные цвета» Sentinel-2 (отражение 0–0,3 → 0–255), проекция UTM 10 м, блоки с перекрытием.
# Ячейка ждёт окончания экспорта; её можно прервать и перезапустить — готовые блоки повторно не экспортируются.
#
# **Вариант `composite`.** Композит перепроецируется в UTM 10 м (модель требует метровую проекцию).

# %%
ENTRIES = {}          # имя запуска → папка с тифами

# %% [markdown]
# Следующая ячейка (GEE) нужна только для варианта `IMAGE_SOURCE = 'gee'`; для `'composite'` её можно пропустить.

# %% include _common_gee.py

# %%
if IMAGE_SOURCE == 'gee':
    aoi_ee = ee.Geometry.Rectangle(list(AOI_BBOX))
    crs = f'EPSG:{utm_epsg((AOI_BBOX[0] + AOI_BBOX[2]) / 2, (AOI_BBOX[1] + AOI_BBOX[3]) / 2)}'
    jobs = []
    for wname, (a, b) in WINDOWS.items():
        rgb = window_median(aoi_ee, a, b, ['B4', 'B3', 'B2']).divide(3000).multiply(255).clamp(1, 255).toUint8()
        for bid, core, exp in block_grid(AOI_BBOX, BLOCK_DEG):
            entry = f'{NAME}_{wname}_{bid}'; d = IMG_ROOT / entry; d.mkdir(exist_ok=True); ENTRIES[entry] = (d, core, wname)
            if list(d.glob('*.tif')):
                continue
            reg = ee.Geometry.Rectangle(list(exp))
            t = ee.batch.Export.image.toDrive(image=rgb.clip(reg), description=entry, folder=GEE_FOLDER, fileNamePrefix=entry,
                                              region=reg, scale=10, crs=crs, maxPixels=1e13, fileDimensions=16384)
            t.start(); jobs.append((bid, core, entry, t))
    if jobs:
        wait_tasks(jobs)
        for bid, core, entry, t in jobs:
            collect_exports([(bid, core, entry, t)], GEE_FOLDER, IMG_ROOT / entry)
else:
    entry = f'{NAME}_composite'; d = IMG_ROOT / entry; d.mkdir(exist_ok=True); ENTRIES[entry] = (d, None, 'composite')
    dst_tif = d / f'{entry}.tif'
    if not dst_tif.exists():
        with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as s:
            cx, cy = (AOI_BBOX[0] + AOI_BBOX[2]) / 2, (AOI_BBOX[1] + AOI_BBOX[3]) / 2
            utm = f'EPSG:{(32600 if cy >= 0 else 32700) + int((cx + 180) // 6) + 1}'
            t, w, h = calculate_default_transform(s.crs, utm, s.width, s.height, *s.bounds, resolution=10)
            prof = dict(driver='GTiff', width=w, height=h, count=3, dtype='uint8', crs=utm, transform=t, nodata=0,
                        compress='deflate', tiled=True, BIGTIFF='IF_SAFER')
            with rasterio.open(dst_tif, 'w', **prof) as dd:
                for i in (1, 2, 3):
                    reproject(rasterio.band(s, i), rasterio.band(dd, i), resampling=Resampling.bilinear)
    print('композит в UTM:', dst_tif)
print('запусков:', len(ENTRIES))

# %% [markdown]
# ## Шаг 4. Запуск Delineate Anything
# Конфигурация берётся из `conf_sample.yaml` репозитория, меняются только: модель, каналы (1,2,3 = R,G,B),
# порог уверенности, размер пачки для GPU и фильтр гигантов. Готовые результаты пропускаются.

# %%
base = yaml.safe_load(open(f'{DA_DIR}/conf_sample.yaml'))
base['model'] = [MODEL]
base['data_loader'].update(bands=[1, 2, 3], nodata_value=[0, 0, 0], min=None, max=None)
base['passes'][0]['batch_size'] = BATCH
base['passes'][0]['model_args'][0].update(name=MODEL, minimal_confidence=MIN_CONF, use_half=bool(torch.cuda.is_available()))
base['giant_filter']['enabled'] = GIANT_FILTER
(WORK / 'temp').mkdir(parents=True, exist_ok=True)
conf_path = WORK / 'da_config.yaml'; yaml.safe_dump(base, open(conf_path, 'w'), allow_unicode=True)
todo = [e for e in ENTRIES if not (OUT / f'{NAME}_raw' / f'{e}.gpkg').exists() and list(ENTRIES[e][0].glob('*.tif'))]
if todo:
    batch = dict(base_config=str(conf_path), data_root=str(IMG_ROOT), output_root=str(OUT / f'{NAME}_raw'),
                 temp_root=str(WORK / 'temp'), keep_temp=False, mask_root=str(WORK / 'no_masks'), include=todo,
                 exclude=None, override=None)
    (OUT / f'{NAME}_raw').mkdir(exist_ok=True)
    bpath = WORK / 'batch.yaml'; yaml.safe_dump(batch, open(bpath, 'w'), allow_unicode=True)
    r = subprocess.run([sys.executable, 'delineate.py', '-b', str(bpath)], cwd=DA_DIR, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
print('готово:', sorted(p.name for p in (OUT / f'{NAME}_raw').glob('*.gpkg')))

# %% include _common_post.py

# %% [markdown]
# ## Шаг 5. Склейка блоков, фильтр по пашне, сохранение
# Для каждого окна дат получается свой слой в `<NAME>_DA_fields.gpkg`. Полигон относится к блоку, в «ядро»
# которого попадает его центр (перекрытие блоков убирает разрезы по швам).

# %%
RESULTS = {}
gpkg = OUT / f'{NAME}_DA_fields.gpkg'
for wname in sorted({v[2] for v in ENTRIES.values()}):
    parts = []
    for e, (d, core, wn) in ENTRIES.items():
        if wn != wname:
            continue
        f = OUT / f'{NAME}_raw' / f'{e}.simp.gpkg'
        f = f if f.exists() else OUT / f'{NAME}_raw' / f'{e}.gpkg'
        if not f.exists():
            continue
        g = gpd.read_file(f).to_crs(4326)
        if core is not None and len(g):
            g = g[g.representative_point().within(shapely.box(*core)).values]
        parts.append(g)
    if not parts:
        continue
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    if AOI_PATH:
        g = g[g.intersects(gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326).union_all())]
    g = filter_fields(g, MIN_CROP_FRAC, MIN_HA); RESULTS[wname] = g
    g.to_file(gpkg, layer=wname, driver='GPKG')
    print(wname, ':', len(g), 'полей, медиана', round(g.area_ha.median(), 1), 'га')
print('Сохранено:', gpkg)

# %% [markdown]
# ## Шаг 6. Растр границ на сетку композита (для гибрида с конвейером v7)
# Контуры полигонов растеризуются линиями (255) на сетку `COMPOSITE_PATH` → `<NAME>_DA_<окно>_boundary.tif`.
# В конвейере v7 укажите его в `EXT_BOUNDARY['TIF']`.

# %%
if COMPOSITE_PATH:
    with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
        prof = dict(driver='GTiff', width=c.width, height=c.height, count=1, dtype='uint8', crs=c.crs, transform=c.transform,
                    compress='deflate', tiled=True, BIGTIFF='IF_SAFER', nodata=0)
        for wname, g in RESULTS.items():
            lines = g.to_crs(c.crs).boundary
            arr = rasterize([(l, 255) for l in lines.values if l is not None and not l.is_empty],
                            out_shape=(c.height, c.width), transform=c.transform, fill=1, dtype='uint8')
            p = OUT / f'{NAME}_DA_{wname}_boundary.tif'
            with rasterio.open(p, 'w', **prof) as d:
                d.write(arr, 1)
            print('Сохранено:', p)

# %% include _common_eval.py

# %% [markdown]
# ## Шаг 7 (по желанию). Сравнение с разметкой и картинка

# %%
if GT_PATH:
    gt = gpd.read_file(DRIVE_ROOT / GT_PATH); kw = {}
    if COMPOSITE_PATH:
        with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
            kw = dict(transform=c.transform, shape=(c.height, c.width), crs=c.crs)
    rows = [evaluate(g[g.intersects(shapely.box(*gt.to_crs(4326).total_bounds))], gt, type_col=GT_TYPE_COL,
                     label=f'DA {w}', **kw) for w, g in RESULTS.items()]
    pd.DataFrame(rows).to_csv(OUT / f'{NAME}_DA_vs_gt.csv', index=False); display(pd.DataFrame(rows))
    if COMPOSITE_PATH:
        with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
            vb = transform_bounds('EPSG:4326', c.crs, *gt.to_crs(4326).total_bounds)
        w0 = list(RESULTS)[0]
        show_overlay(DRIVE_ROOT / COMPOSITE_PATH, [(RESULTS[w0], 'cyan', f'DA {w0}'), (gt, 'red', 'эталон')], title=NAME,
                     bbox=vb, out_png=OUT / f'{NAME}_DA_view.png')

# %% [markdown]
# ## Что дальше
# - Сравните окна дат между собой и с FTW (ноутбуки 01–02); лучшее окно запомните для всей области.
# - Если модель хорошо видит границы, но режет/склеивает поля иначе, чем вам нужно, — подайте её границы в конвейер v7
#   (`EXT_BOUNDARY['TIF']`): нарезку на поля по вашему определению сделает конвейер.
# - Дообучение Delineate Anything возможно только общим обучением YOLO (Ultralytics) на разметке в формате YOLO-сегментации;
#   для этого проекта проще и надёжнее дообучать FTW U-Net (ноутбук 04).
