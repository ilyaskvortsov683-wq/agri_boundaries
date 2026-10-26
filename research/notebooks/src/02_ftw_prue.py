# %% [markdown]
# # 02 · Нейросеть FTW (PRUE) на ваших снимках через `ftw-tools`
#
# **Что это.** Модели Fields of The World версии 3 (рецепт **PRUE**, U-Net с кодировщиком EfficientNet), которые
# по двум сезонным снимкам Sentinel-2 (4 канала: красный, зелёный, синий, ближний ИК) выдают для каждого пикселя
# вероятности «фон / поле / граница». Из них собираются полигоны полей. Это та же «семья» моделей, что построила
# глобальную карту (ноутбук 01), но здесь вы запускаете её **на свой год** (например, 2026, которого в карте ещё нет)
# и на **свои сезонные окна**.
#
# **Что делает ноутбук.**
# 1. В GEE собирает **многодатный вход**: медиана Sentinel-2 за окно A (начало сезона) и окно B (вторая половина
#    сезона), 8 каналов `B4,B3,B2,B8 (A) + B4,B3,B2,B8 (B)`, uint16, UTM 10 м. Большой участок режется на блоки.
# 2. Запускает `ftw inference run` на GPU (вероятности 0–255 по трём классам) и `ftw inference polygonize`.
# 3. Склеивает блоки, фильтрует полигоны по доле пашни WorldCover (как в конвейере), сохраняет GeoPackage.
# 4. Переносит вероятность **границы** на сетку вашего композита → GeoTIFF для гибрида с конвейером v7.
# 5. (по желанию) Сравнивает с вашей разметкой.
#
# **Среда:** Среда выполнения → Сменить среду → **T4 GPU**. Время: экспорт GEE 5–30 мин на блок (идёт на серверах
# Google), инференс ≈1–3 мин на блок 0,5°×0,5° на T4.
#
# **Чего ждать (проверено на Донбассе, 2026, окна 10.04–31.05 и 15.07–31.08, модель B5, 301 эталонное поле).**
# Сами полигоны слабые: IoU 0,29, точность 0,71, полнота 0,40; залежь почти не находится (охват 0,12), соседние
# поля одного тона склеиваются. Но **карта границ** (шаг 6) полезна как барьеры в конвейере v7: v6 0,635 →
# с границами PRUE 0,657; вместе с картой FTW Global (ноутбук 01) 0,678; ещё и с Higra — **0,684**.
# На CPU блок 16×18 км считался ~2 мин; на T4 — в разы быстрее.
#
# **Лицензии моделей.** `FTW_PRUE_EFNET_B5` (по умолчанию) обучена на смеси лицензий, включая некоммерческие (NC);
# для коммерческого использования берите `…_CCBY` (обучены только на CC-BY/CC0 данных, чуть слабее).

# %% [markdown]
# ## Шаг 0. Установка `ftw-tools` (один раз за сеанс)
# Ячейка ставит `ftw-tools 2.0.0b5` (в ней есть модели PRUE; обычный `pip install ftw-tools` ставит старую 1.4.3 без них).
# **После установки:** меню «Среда выполнения» → «Перезапустить сеанс», затем продолжайте **с шага 1** (шаг 0 повторно
# запускать не нужно — он сам увидит, что всё уже стоит).

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec('ftw_tools') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ftw-tools[all]==2.0.0b5'], check=True)
    print('Установлено. Теперь: Среда выполнения → Перезапустить сеанс, и дальше с шага 1.')
else:
    import ftw_tools, torch
    print('ftw-tools уже установлен; torch', torch.__version__, '| GPU:', torch.cuda.is_available())

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры — заполните под себя
#
# **Окна дат (A и B)** — главное, что нужно подобрать под регион. Модель ждёт снимок «в начале сезона» (озимые
# зелёные, яровые только посеяны/всходы) и «во второй половине» (озимые убраны или желтеют, яровые в максимуме).
# Ориентиры:
#
# | регион | окно A | окно B |
# |---|---|---|
# | Донбасс, Ростов, Краснодар, Ставрополь, Волгоград | 10.04–31.05 | 15.07–31.08 |
# | Тамбов, Липецк, Воронеж, Саратов, Самара | 20.04–10.06 | 20.07–05.09 |
# | Татарстан, Ульяновск, Москва | 01.05–15.06 | 25.07–10.09 |
# | Вологда, север Нечерноземья | 15.05–25.06 | 01.08–15.09 |
#
# Если сомневаетесь — запустите два варианта окон на маленьком тестовом участке и сравните (шаг 6).

# %%
EE_PROJECT = 'my-project-20260614-499410'

COMPOSITE_PATH = 'Colab_Notebooks/Agro_terra/TAT/TA_composite_2026_0504_0910_8b.tif'   # охват + сетка для вероятностей
AOI_PATH = None                     # или граница участка (geojson/gpkg)
BBOX = None                         # или (lon_min, lat_min, lon_max, lat_max) — удобно для быстрого теста
YEAR = 2026
WIN_A = ('2026-05-01', '2026-06-15')
WIN_B = ('2026-07-25', '2026-09-10')

MODEL = 'FTW_PRUE_EFNET_B5'         # или 'FTW_PRUE_EFNET_B7' (точнее, медленнее), '…_B3', '…_B5_CCBY' (лицензия CC-BY)
BLOCK_DEG = 0.5                     # размер блока экспорта/инференса в градусах
MIN_CROP_FRAC = 0.5                 # доля пашни WorldCover в полигоне (как в конвейере)
MIN_HA = 1.0                        # минимальная площадь поля, га
SIMPLIFY_M = 10                     # упрощение контуров при полигонизации, м

GT_PATH = None                      # ваша разметка для сравнения (необязательно)
GT_TYPE_COL = 'type'
OUT_DIR = 'Colab_Notebooks/Agro_terra/research/ftw_prue/out'
GEE_FOLDER = 'GEE_ftw_prue'         # временная папка в корне Диска для экспорта GEE
NAME = 'TAT_2026'

# %% include _common_gee.py

# %% [markdown]
# ## Шаг 2. Участок и блоки

# %%
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio
from rasterio.warp import transform_bounds
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT / f'{NAME}_input'; INF_DIR = OUT / f'{NAME}_inference'; INF_DIR.mkdir(parents=True, exist_ok=True)
if BBOX:
    AOI_BBOX = tuple(BBOX)
elif COMPOSITE_PATH:
    with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as s:
        AOI_BBOX = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
else:
    AOI_BBOX = tuple(gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326).total_bounds)
BLOCKS = block_grid(AOI_BBOX, BLOCK_DEG)
print('рамка', np.round(AOI_BBOX, 3), '| блоков:', len(BLOCKS))

# %% [markdown]
# ## Шаг 3. Многодатный вход в GEE: окно A + окно B → 8 каналов, экспорт на Диск
# Для каждого окна берётся медиана безоблачных пикселей (маска Cloud Score+); дыры заполняются медианой окна,
# расширенного на ±30 дней. Если файлы блоков уже лежат в `<NAME>_input` (например, после прерванного сеанса),
# экспорт этих блоков пропускается.
#
# Ячейка ждёт окончания экспортов (видно «готово N из M»). Её можно прервать — задачи продолжатся на серверах GEE;
# тогда потом просто запустите её снова: готовые блоки не будут экспортироваться повторно.

# %%
aoi_ee = ee.Geometry.Rectangle(list(AOI_BBOX))
BANDS = ['B4', 'B3', 'B2', 'B8']
img = (window_median(aoi_ee, WIN_A[0], WIN_A[1], BANDS).rename([f'{b}_A' for b in BANDS])
       .addBands(window_median(aoi_ee, WIN_B[0], WIN_B[1], BANDS).rename([f'{b}_B' for b in BANDS])))
todo = [b for b in BLOCKS if not list(IMG_DIR.glob(f'{NAME}_{b[0]}*.tif'))]
if todo:
    jobs = []
    crs = f'EPSG:{utm_epsg((AOI_BBOX[0] + AOI_BBOX[2]) / 2, (AOI_BBOX[1] + AOI_BBOX[3]) / 2)}'
    for bid, core, exp in todo:
        name = f'{NAME}_{bid}'; reg = ee.Geometry.Rectangle(list(exp))
        t = ee.batch.Export.image.toDrive(image=img.toUint16().clip(reg), description=name, folder=GEE_FOLDER,
                                          fileNamePrefix=name, region=reg, scale=10, crs=crs, maxPixels=1e13,
                                          fileDimensions=16384)
        t.start(); jobs.append((bid, core, name, t))
    wait_tasks(jobs)
    collect_exports(jobs, GEE_FOLDER, IMG_DIR)
else:
    print('Все блоки уже экспортированы:', len(BLOCKS))

# %% [markdown]
# ## Шаг 4. Инференс FTW и полигоны
# Для каждого блока: `ftw inference run --save_scores` (3 канала вероятностей ×255: фон, поле, граница) и
# `ftw inference polygonize --softmax_threshold 0.5`. Готовые блоки пропускаются (можно перезапускать).
# Первый запуск скачивает веса модели (~100–300 МБ).

# %%
import subprocess, torch
GPU = 0 if torch.cuda.is_available() else -1
print('GPU' if GPU == 0 else 'ВНИМАНИЕ: GPU нет, будет медленно (CPU)')
for f in sorted(IMG_DIR.glob(f'{NAME}_b*.tif')):
    scores = INF_DIR / f'{f.stem}_scores.tif'; poly = INF_DIR / f'{f.stem}_fields.gpkg'
    if not scores.exists():
        r = subprocess.run(['ftw', 'inference', 'run', str(f), '--model', MODEL, '--out', str(scores), '--gpu', str(GPU),
                            '--save_scores', '--num_workers', '2', '--overwrite'], capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit('ошибка инференса ' + f.name)
    if not poly.exists():
        r = subprocess.run(['ftw', 'inference', 'polygonize', str(scores), '--out', str(poly), '--softmax_threshold', '0.5',
                            '--simplify', str(SIMPLIFY_M), '--min_size', str(int(MIN_HA * 1e4 / 2)), '--overwrite'],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit('ошибка полигонизации ' + f.name)
    print('готов блок', f.stem)

# %% include _common_post.py

# %% [markdown]
# ## Шаг 5. Склейка блоков, фильтр по пашне, сохранение
# Полигон относится к тому блоку, в «ядро» которого попадает его центр (перекрытие блоков убирает разрезы по шву).

# %%
core_of = {f'{NAME}_{bid}': core for bid, core, _ in BLOCKS}
parts = []
for p in sorted(INF_DIR.glob(f'{NAME}_b*_fields.gpkg')):
    g = gpd.read_file(p); key = p.stem.replace('_fields', '').split('-')[0]
    core = core_of.get(key)
    if core is not None and len(g):
        c = g.to_crs(4326).representative_point()
        g = g[c.within(shapely.box(*core)).values]
    parts.append(g.to_crs(4326))
fields = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
if AOI_PATH:
    fields = fields[fields.intersects(gpd.read_file(DRIVE_ROOT / AOI_PATH).to_crs(4326).union_all())]
print('всего полигонов до фильтра:', len(fields))
fields = filter_fields(fields, MIN_CROP_FRAC, MIN_HA)
out_gpkg = OUT / f'{NAME}_{MODEL}_fields.gpkg'
fields.to_file(out_gpkg, layer='fields', driver='GPKG')
print('Сохранено:', out_gpkg, '| медиана площади', round(fields.area_ha.median(), 1), 'га')

# %% [markdown]
# ## Шаг 6. Вероятность границы на сетку композита (для гибрида с конвейером v7)
# Пишется `<NAME>_<MODEL>_boundary.tif` (uint8, 1–255; 0 — нет данных) на сетке `COMPOSITE_PATH`. В конвейере v7
# укажите его в `EXT_BOUNDARY['TIF']` — линии границ нейросети станут дополнительными барьерами при слиянии.

# %%
if COMPOSITE_PATH:
    with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
        prof = dict(driver='GTiff', width=c.width, height=c.height, count=1, dtype='uint8', crs=c.crs, transform=c.transform,
                    compress='deflate', tiled=True, BIGTIFF='IF_SAFER', nodata=0)
        acc = np.zeros((c.height, c.width), np.uint8)
    for sc in sorted(INF_DIR.glob(f'{NAME}_b*_scores.tif')):
        b = to_grid(sc, 3, DRIVE_ROOT / COMPOSITE_PATH)          # канал 3 = граница
        ok = ~np.isnan(b); acc[ok] = np.maximum(acc[ok], np.clip(b[ok] * 254 / 255 + 1, 1, 255).astype(np.uint8))
    bnd_tif = OUT / f'{NAME}_{MODEL}_boundary.tif'
    with rasterio.open(bnd_tif, 'w', **prof) as d:
        d.write(acc, 1)
    print('Сохранено:', bnd_tif)

# %% include _common_eval.py

# %% [markdown]
# ## Шаг 7 (по желанию). Сравнение с разметкой и картинка

# %%
if GT_PATH:
    gt = load_gt(DRIVE_ROOT / GT_PATH, AOI_BBOX)
    kw = {}
    if COMPOSITE_PATH:
        with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
            kw = dict(transform=c.transform, shape=(c.height, c.width), crs=c.crs)
    res = evaluate(fields[fields.intersects(shapely.box(*gt.to_crs(4326).total_bounds))], gt, type_col=GT_TYPE_COL,
                   label=f'{MODEL} {YEAR}', **kw)
    pd.DataFrame([res]).to_csv(OUT / f'{NAME}_{MODEL}_vs_gt.csv', index=False)
    if COMPOSITE_PATH:
        with rasterio.open(DRIVE_ROOT / COMPOSITE_PATH) as c:
            vb = transform_bounds('EPSG:4326', c.crs, *gt.to_crs(4326).total_bounds)
        show_overlay(DRIVE_ROOT / COMPOSITE_PATH, [(fields, 'cyan', MODEL), (gt, 'red', 'эталон')], title=NAME, bbox=vb,
                     out_png=OUT / f'{NAME}_{MODEL}_view.png')

# %% [markdown]
# ## Что дальше
# - Сравните с картой FTW (ноутбук 01) и с конвейером v6 на одном и том же участке.
# - Если поля дробятся по культурам, а залежь пропускается — это ожидаемо: модель обучена на декларациях ЕС.
#   Лечится либо гибридом (шаг 6 → конвейер v7), либо дообучением на ваших полях (ноутбук 04).
# - Свою дообученную модель (файл `.ckpt` из ноутбука 04) можно подставить в `MODEL` полным путём к файлу.
