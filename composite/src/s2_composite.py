# %% [markdown]
# # Сезонный композит NDVI Sentinel-2 без Google Earth Engine
#
# **Что делает.** Считает ваш композит **max / min / diff NDVI** (как скрипт в GEE) или **max / min / std NDVI** за
# период сезона по области интереса и сохраняет один GeoTIFF: 3 канала, 8 бит, значения **1–255** (0 — нет данных),
# сжатие **JPEG 75** (или без потерь).
#
# **Откуда данные.** Те же снимки Sentinel-2 L2A, что в GEE (`S2_SR_HARMONIZED`), но из открытого архива на AWS
# (Earth Search, Element84): файлы COG читаются по кусочкам, только каналы B4, B8 и маска сцены SCL и только в границах
# вашей области. GEE не нужен вообще, поэтому нет ни очередей экспорта, ни лимитов на размер.
#
# **Насколько совпадает с GEE.** Проверено на Донбассе (2026, 30.04–10.09, 16×18 км): новый композит против вашего
# композита из GEE (оба в JPEG 75). Пиксели совпадают один в один (та же сетка), корреляция по каналам max / min / diff —
# **0,96 / 0,97 / 0,96**, среднее расхождение 4 / 5 / 7 единиц из 255. Остаток — от разных масок облаков (здесь SCL
# с запасом 3 пикселя, в GEE — ваша) и от самого JPEG. Сильнее всего от маски облаков зависит канал min.
#
# **Как пользоваться**
# 1. Положите границы областей в папку `aoi` рядом с этим ноутбуком: `aoi/TAT.geojson`, `aoi/SAM.geojson`, … Подойдёт
#    любой GeoJSON с полигоном(ами) в любой СК. Можно выгрузить все субъекты из вашего слоя в GEE (шаг 2, по желанию).
# 2. В шаге 1 укажите `AOI` (имя файла без `.geojson`), год и даты начала/конца периода.
# 3. Сначала запустите `MODE = 'test'` (квадрат 20×20 км, ~5–10 мин). Будут посчитаны оба варианта (diff и std) и, если
#    указан `REFERENCE`, сравнение с вашим композитом из GEE.
# 4. Потом `MODE = 'full'` — вся область. Считается блоками, каждый готовый блок сразу сохраняется на Диск. Если Colab
#    отключился, просто запустите ноутбук снова: посчитанные блоки пропускаются.
#
# **Что получится:** `composite/out/<AOI>/composite_<год>_<ММДД начала>_<ММДД конца>.tif`
# (например `composite_2026_0504_0910.tif`), для варианта std — с суффиксом `_std`.
#
# **Время.** Около 1,5 мин на 1000 км² при сезоне 4–5 месяцев (проверено: квадрат 30×30 км — 80–115 с). Область
# размера Татарстана (68 тыс. км²) — примерно 2–4 часа, можно частями: ноутбук продолжает с места остановки. GPU не нужен.
#
# **JPEG.** Качество 75 (как у вашего композита из GEE) даёт среднюю ошибку 2–3 единицы из 255 (у 95 % пикселей — не
# больше 7–9) и файл в 3 раза меньше, чем без потерь. JPEG 90: ошибка ~1,5, файл на 70 % больше, чем при 75.
#
# ```
# Colab_Notebooks/Agro_terra/composite/
# ├── S2_composite.ipynb      ← этот ноутбук
# ├── aoi/                    ← границы областей интереса (*.geojson)
# └── out/<AOI>/              ← готовые композиты
# ```

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры

# %%
AOI = 'TAT'                          # имя файла в папке aoi (без .geojson)
YEAR = 2026
START = '05-04'                      # начало периода (месяц-день), включительно
END = '09-10'                        # конец периода, включительно
VARIANTS = ['diff']                  # 'diff' → max/min/diff (как в GEE); 'std' → max/min/std; оба сразу: ['diff', 'std']

MODE = 'test'                        # 'test' — квадрат TEST_KM × TEST_KM (оба варианта + сравнение); 'full' — вся область
TEST_KM = 20
TEST_CENTER = None                   # (долгота, широта) центра тестового квадрата; None — центр области
REFERENCE = None                     # для теста: ваш композит из GEE, путь от корня Диска, напр.
                                     # 'Colab_Notebooks/Agro_terra/TAT/TA_composite_2026_0504_0910_8b.tif'

# снимки и облака
MAX_CLOUD = 80                       # сцены с облачностью выше (%, по метаданным сцены) не берутся
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]    # выкидываемые классы маски SCL: нет данных, дефект, тень облака, облака, перистые, снег
CLOUD_BUFFER_PX = 3                  # расширить маску облаков и теней на столько пикселей (10 м); 0 — как есть
REDUCER = 'minmax'                   # 'minmax' — строгие max/min, как в GEE; 'pct' — процентили PCT_LOW/PCT_HIGH
PCT_LOW, PCT_HIGH = 2, 98            # (устойчивее к пропущенным облакам, но медленнее и требует BLOCK_PX ≤ 1024)
COLLECTION = 'sentinel-2-c1-l2a'     # архив Element84: 'sentinel-2-c1-l2a' (новый) или 'sentinel-2-l2a' (старый)
SOURCE = 'stac'                      # 'stac' — поиск сцен через Earth Search; 's3' — запасной путь (перечень архива по тайлам)

# сетка и файл
GRID = 'gee'                         # 'gee' — EPSG:4326, шаг 8,98·10⁻⁵° (как экспорт GEE со scale=10); 'utm' — UTM, 10 м
COMPRESS = 'JPEG'                    # 'JPEG' (с потерями, качество JPEG_QUALITY) или 'DEFLATE' (без потерь, в 3–4 раза больше)
JPEG_QUALITY = 75
NAME_PREFIX = ''                     # приставка к имени файла, напр. 'TA_' → TA_composite_2026_0504_0910.tif
RESAMPLING = 'nearest'               # перенос из сетки снимков (UTM 10 м) в выходную: 'nearest' или 'bilinear' (мягче)
BLOCK_PX = 2048                      # размер блока расчёта, пикс.
N_THREADS = 16                       # одновременных чтений снимков
SCRIPT_DIR = 'Colab_Notebooks/Agro_terra/composite'   # папка ноутбука на Диске (в ней aoi/ и out/)
KEEP_BLOCKS = False                  # True — не удалять промежуточные блоки после сборки

# %% [markdown]
# ## Шаг 2 (по желанию). Выгрузить границы субъектов из вашего слоя в GEE в папку `aoi`
# Нужно один раз, если субъекты лежат в ассете GEE. Каждый объект сохраняется как `aoi/<значение поля>.geojson`
# (границы упрощаются до 20 м). Уже существующие файлы не перезаписываются. Свои области (не совпадающие с
# субъектами) просто кладите в `aoi` вручную, например нарисовав в QGIS или geojson.io.

# %%
GEE_AOI_ASSET = None                 # напр. 'projects/my-project-20260614-499410/assets/subjects'
GEE_AOI_FIELD = 'name'               # поле с названием субъекта
EE_PROJECT = 'my-project-20260614-499410'

import re, shapely, geopandas as gpd
AOI_DIR = DRIVE_ROOT / SCRIPT_DIR / 'aoi'; AOI_DIR.mkdir(parents=True, exist_ok=True)
if GEE_AOI_ASSET:
    import ee
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=EE_PROJECT)
    fc = ee.FeatureCollection(GEE_AOI_ASSET)
    for nm in fc.aggregate_array(GEE_AOI_FIELD).distinct().getInfo():
        f = AOI_DIR / (re.sub(r'[\\/:*?"<>|]+', '_', str(nm)).strip() + '.geojson')
        if f.exists():
            continue
        geo = fc.filter(ee.Filter.eq(GEE_AOI_FIELD, nm)).geometry().simplify(20).getInfo()
        gpd.GeoDataFrame({'name': [nm]}, geometry=[shapely.geometry.shape(geo)], crs=4326).to_file(f, driver='GeoJSON')
        print('сохранено', f.name)
print('в папке aoi:', sorted(p.stem for p in AOI_DIR.glob('*.geojson')))

# %% [markdown]
# ## Шаг 3. Функции (выполните без изменений)
# - `find_scenes()` — список сцен за период по области (облачность сцены < `MAX_CLOUD`);
# - `block_composite()` — композит в одном блоке выходной сетки. Для каждого тайла MGRS сцены накапливаются в его родной
#   сетке UTM (max, min, сумма и сумма квадратов NDVI — без хранения всех снимков в памяти), и только итог переносится
#   в выходную сетку методом ближайшего соседа (так же делает GEE по умолчанию);
# - масштаб: канал = round(NDVI × 254) + 1 (NDVI < 0 → 1), diff = round((max − min) × 254) + 1,
#   std = round(СКО × 635) + 1 (СКО 0,4 → 255). Так же масштабирован ваш композит из GEE.

# %%
import subprocess, sys, importlib.util
if importlib.util.find_spec('pystac_client') is None or importlib.util.find_spec('mgrs') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pystac-client', 'mgrs'], check=True)
import math, threading, hashlib, datetime as dt, urllib.request, numpy as np, pandas as pd, rasterio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import Affine
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', AWS_NO_SIGN_REQUEST='YES',
                  GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES', GDAL_HTTP_MAX_RETRY='5',
                  GDAL_HTTP_RETRY_DELAY='2', VSI_CACHE='TRUE', VSI_CACHE_SIZE='100000000', GDAL_CACHEMAX='512',
                  GDAL_TIFF_INTERNAL_MASK='YES')
GEE_PX = 10 / 111319.49079327357          # шаг GEE при scale=10 в EPSG:4326, градусов
S3 = {'sentinel-2-l2a': ('https://sentinel-cogs.s3.us-west-2.amazonaws.com', 'sentinel-s2-l2a-cogs'),
      'sentinel-2-c1-l2a': ('https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com', 'sentinel-2-c1-l2a')}


def parse_item(d):
    """Описание сцены STAC → дата, тайл, сетка 10 м, ссылки на B4/B8/SCL и смещение отражения (в единицах файла)."""
    p, a = d['properties'], d['assets']
    if not all(k in a for k in ('red', 'nir', 'scl')):
        return None
    rb = (a['red'].get('raster:bands') or [{}])[0]
    if d.get('collection') == 'sentinel-2-l2a':
        off = 0.0          # старый архив: смещение +1000 уже вычтено во всех сценах (флаг boa_offset_applied бывает неверен)
    elif 'offset' in rb:
        off = rb['offset'] / rb.get('scale', 1e-4)    # новый архив c1: −0,1 → −1000 (проверено по пикселям)
    else:
        off = -1000.0 if str(p.get('s2:processing_baseline', '0')) >= '04.00' else 0.0
    code = p.get('proj:epsg') or a['red'].get('proj:epsg') or str(p.get('proj:code', a['red'].get('proj:code', ''))).split(':')[-1]
    href = lambda k: re.sub(r'^s3://([^/]+)/', r'https://\1.s3.us-west-2.amazonaws.com/', a[k]['href'])
    return dict(id=d['id'], date=p['datetime'][:10], tile=p.get('grid:code', 'MGRS-' + d['id']).replace('MGRS-', '')[:5],
                epsg=int(code), cloud=float(p.get('eo:cloud_cover', 0)), off=off,
                transform=Affine(*a['red']['proj:transform'][:6]), shape=tuple(a['red']['proj:shape']),
                red=href('red'), nir=href('nir'), scl=href('scl'))


def find_scenes(geom, d0, d1):
    """Сцены за [d0, d1] по геометрии (EPSG:4326)."""
    items = []
    if SOURCE == 'stac':
        from pystac_client import Client
        cat = Client.open('https://earth-search.aws.element84.com/v1')
        s = cat.search(collections=[COLLECTION], intersects=shapely.geometry.mapping(geom.simplify(0.01)),
                       datetime=f'{d0}T00:00:00Z/{d1}T23:59:59Z', query={'eo:cloud_cover': {'lt': MAX_CLOUD}})
        items = [parse_item(i.to_dict()) for i in s.items()]
    else:                                             # перечень архива по тайлам MGRS (без STAC)
        import mgrs, re as _re
        base, root = S3[COLLECTION]; m = mgrs.MGRS()
        x0, y0, x1, y1 = geom.bounds
        pts = [(x, y) for x in np.arange(x0, x1 + .05, .05) for y in np.arange(y0, y1 + .05, .05)]
        tiles = {m.toMGRS(y, x, MGRSPrecision=0) for x, y in pts if geom.buffer(.05).contains(shapely.Point(x, y))}
        BANDS = 'CDEFGHJKLMNPQRSTUVWX'                # у границы широтных поясов тайл Sentinel-2 носит букву соседнего пояса
        tiles = sorted({t[:2] + BANDS[j] + t[3:] for t in tiles for j in range(max(BANDS.index(t[2]) - 1, 0),
                                                                                    min(BANDS.index(t[2]) + 2, len(BANDS)))})
        urls = []
        for t in tiles:
            for mo in range(int(d0[5:7]), int(d1[5:7]) + 1):
                pre = f'{root}/{int(t[:2])}/{t[2]}/{t[3:]}/{d0[:4]}/{mo}/'
                xml = urllib.request.urlopen(f'{base}/?list-type=2&prefix={pre}&delimiter=/', timeout=60).read().decode()
                urls += [f'{base}/{p}{p.rstrip("/").split("/")[-1]}.json'
                         for p in _re.findall(r'<Prefix>(' + _re.escape(pre) + r'[^<]+/)</Prefix>', xml)]
        def one(u):
            try:
                return parse_item(json.loads(urllib.request.urlopen(u, timeout=60).read()))
            except Exception:
                return None
        with ThreadPoolExecutor(16) as ex:
            items = [i for i in ex.map(one, urls) if i and d0 <= i['date'] <= d1 and i['cloud'] < MAX_CLOUD]
    return sorted([i for i in items if i], key=lambda i: (i['date'], i['tile']))


def read_scene(it, win):
    """NDVI сцены в окне её родной сетки 10 м; облака/тени/снег по SCL → NaN. None — в окне нет чистых пикселей."""
    h, w = int(win.height), int(win.width)
    with rasterio.open(it['scl']) as s:              # SCL 20 м → 10 м ближайшим соседом
        scl = s.read(1, window=Window(win.col_off / 2, win.row_off / 2, w / 2, h / 2), out_shape=(h, w),
                     resampling=Resampling.nearest)
    ok = ~np.isin(scl, SCL_BAD)
    if CLOUD_BUFFER_PX:                              # края облаков и теней SCL часто пропускает — расширяем маску
        ok &= ~binary_dilation(np.isin(scl, [3, 8, 9, 10]), iterations=int(CLOUD_BUFFER_PX))
    if not ok.any():
        return None
    with rasterio.open(it['red']) as s:
        red = s.read(1, window=win).astype(np.float32)
    with rasterio.open(it['nir']) as s:
        nir = s.read(1, window=win).astype(np.float32)
    edge = (scl == 0) | (red == 0) | (nir == 0)      # край полосы съёмки: каналы обрезаны по-разному, там NDVI ≈ 1
    ok &= ~binary_dilation(edge, iterations=3)
    red += it['off']; nir += it['off']
    ok &= (red > 50) & (nir > 50)                    # отражение > 0,005: тёмные артефакты не считаются
    return np.where(ok, (nir - red) / np.where(ok, red + nir, 1), np.nan).astype(np.float32)


def tile_accumulate(its, win):
    """Накопление по сценам одного тайла MGRS: max, min, сумма, сумма квадратов, число наблюдений."""
    h, w = int(win.height), int(win.width)
    A = dict(mx=np.full((h, w), -np.inf, np.float32), mn=np.full((h, w), np.inf, np.float32),
             s1=np.zeros((h, w)), s2=np.zeros((h, w)), n=np.zeros((h, w), np.float32))
    stack, lock = [], threading.Lock()

    def one(it):
        for k in range(4):
            try:
                nd = read_scene(it, win); break
            except Exception as e:
                err = e; time.sleep(2 + 4 * k)
        else:
            print('  пропущена сцена', it['id'], '—', str(err)[:120]); return
        if nd is None:
            return
        ok = np.isfinite(nd); z = np.where(ok, nd, 0)
        with lock:
            np.maximum(A['mx'], np.where(ok, nd, -np.inf), out=A['mx']); np.minimum(A['mn'], np.where(ok, nd, np.inf), out=A['mn'])
            A['s1'] += z; A['s2'] += z.astype(np.float64) ** 2; A['n'] += ok
            if REDUCER == 'pct':
                stack.append(nd.astype(np.float16))
    with ThreadPoolExecutor(N_THREADS) as ex:
        list(ex.map(one, its))
    if REDUCER == 'pct' and stack:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = np.nanpercentile(np.stack(stack).astype(np.float32), [PCT_LOW, PCT_HIGH], axis=0)
        A['mn'], A['mx'] = np.where(np.isnan(q[0]), np.inf, q[0]), np.where(np.isnan(q[1]), -np.inf, q[1])
    return A


def block_composite(bt, bshape, bcrs, items):
    """Композит в блоке выходной сетки (transform bt, размер bshape, СК bcrs) → dict max/min/s1/s2/n на этой сетке."""
    H, W = bshape
    out = dict(mx=np.full(bshape, np.nan, np.float32), mn=np.full(bshape, np.nan, np.float32),
               s1=np.zeros(bshape), s2=np.zeros(bshape), n=np.zeros(bshape, np.float32))
    bb = rasterio.transform.array_bounds(H, W, bt)
    by_tile = defaultdict(list)
    for it in items:
        by_tile[(it['tile'], it['epsg'])].append(it)
    for (tile, epsg), its in by_tile.items():
        t0, (th, tw) = its[0]['transform'], its[0]['shape']
        l, b, r, t = transform_bounds(bcrs, f'EPSG:{epsg}', *bb, densify_pts=21)
        win = from_bounds(l - 30, b - 30, r + 30, t + 30, t0)
        c0, r0 = max(int(math.floor(win.col_off)) // 2 * 2, 0), max(int(math.floor(win.row_off)) // 2 * 2, 0)
        c1, r1 = min(int(math.ceil(win.col_off + win.width)), tw), min(int(math.ceil(win.row_off + win.height)), th)
        c1, r1 = c1 + (c1 - c0) % 2, r1 + (r1 - r0) % 2
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        w = Window(c0, r0, c1 - c0, r1 - r0)
        A = tile_accumulate(its, w)
        if not A['n'].any():
            continue
        wt = rasterio.windows.transform(w, t0)
        A['mx'][A['n'] == 0] = np.nan; A['mn'][A['n'] == 0] = np.nan
        for k in ('mx', 'mn', 's1', 's2', 'n'):
            src = A[k].astype(np.float32) if k != 's1' and k != 's2' else A[k]
            dst = np.full(bshape, np.nan, src.dtype)
            reproject(src, dst, src_transform=wt, src_crs=f'EPSG:{epsg}', dst_transform=bt, dst_crs=bcrs,
                      src_nodata=np.nan if k in ('mx', 'mn') else None, dst_nodata=np.nan,
                      resampling=Resampling.bilinear if RESAMPLING == 'bilinear' else Resampling.nearest)
            if k == 'mx':
                out[k] = np.fmax(out[k], dst)
            elif k == 'mn':
                out[k] = np.fmin(out[k], dst)
            else:
                out[k] += np.nan_to_num(dst)
    return out


def to_bands(c, inside):
    """Итоги блока → uint8 (1–255; 0 — нет данных): max, min, diff, std, число наблюдений."""
    ok = (c['n'] > 0) & inside & np.isfinite(c['mx'])
    sc = lambda v, k: np.where(ok, np.round(np.clip(np.nan_to_num(v), 0, 1) * k) + 1, 0).clip(0, 255).astype(np.uint8)
    n = np.maximum(c['n'], 1)
    std = np.sqrt(np.maximum(c['s2'] / n - (c['s1'] / n) ** 2, 0))
    return np.stack([sc(c['mx'], 254), sc(c['mn'], 254), sc(c['mx'] - c['mn'], 254), sc(std * 635 / 254, 254),
                     np.where(ok, np.clip(c['n'], 1, 255), 0).astype(np.uint8)])

print('функции готовы')

# %% [markdown]
# ## Шаг 4. Область, сетка, блоки
# Сетка `gee` совпадает с сеткой экспорта GEE (`crs='EPSG:4326', scale=10`): тот же шаг и то же начало отсчёта
# пикселей, поэтому новый композит ложится пиксель в пиксель на старый.

# %%
import geopandas as gpd, shapely
aoi_path = AOI_DIR / f'{AOI}.geojson'
assert aoi_path.exists(), f'нет файла {aoi_path} — положите границу области в папку aoi'
AOI_GEOM = shapely.make_valid(gpd.read_file(aoi_path).to_crs(4326).union_all())
D0, D1 = f'{YEAR}-{START}', f'{YEAR}-{END}'
NAME = f"{NAME_PREFIX}composite_{YEAR}_{START.replace('-', '')}_{END.replace('-', '')}"
if MODE == 'test':
    cx, cy = TEST_CENTER or (AOI_GEOM.representative_point().x, AOI_GEOM.representative_point().y)
    dx, dy = TEST_KM / 2 / (111.32 * math.cos(math.radians(cy))), TEST_KM / 2 / 110.57
    WORK_GEOM = shapely.box(cx - dx, cy - dy, cx + dx, cy + dy)
else:
    WORK_GEOM = AOI_GEOM
if GRID == 'gee':
    CRS = 'EPSG:4326'; px = GEE_PX
    x0, y0, x1, y1 = WORK_GEOM.bounds
    gx0, gy1 = math.floor(x0 / px) * px, math.ceil(y1 / px) * px
    W, H = math.ceil((x1 - gx0) / px), math.ceil((gy1 - y0) / px)
    T = Affine(px, 0, gx0, 0, -px, gy1)
else:
    lon, lat = WORK_GEOM.centroid.x, WORK_GEOM.centroid.y
    CRS = f'EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}'
    x0, y0, x1, y1 = gpd.GeoSeries([WORK_GEOM], crs=4326).to_crs(CRS).total_bounds
    gx0, gy1 = math.floor(x0 / 10) * 10, math.ceil(y1 / 10) * 10
    W, H = math.ceil((x1 - gx0) / 10), math.ceil((gy1 - y0) / 10)
    T = Affine(10, 0, gx0, 0, -10, gy1)
WORK_LOCAL = gpd.GeoSeries([WORK_GEOM.intersection(AOI_GEOM) if MODE == 'test' else AOI_GEOM], crs=4326).to_crs(CRS).iloc[0]
BLOCKS = []
for r0 in range(0, H, BLOCK_PX):
    for c0 in range(0, W, BLOCK_PX):
        w = Window(c0, r0, min(BLOCK_PX, W - c0), min(BLOCK_PX, H - r0))
        if WORK_LOCAL.intersects(shapely.box(*rasterio.windows.bounds(w, T))):
            BLOCKS.append(w)
OUT = DRIVE_ROOT / SCRIPT_DIR / 'out' / AOI; OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = [AOI, D0, D1, MODE, TEST_KM, TEST_CENTER, MAX_CLOUD, SCL_BAD, CLOUD_BUFFER_PX, REDUCER, PCT_LOW, PCT_HIGH,
            COLLECTION, GRID, BLOCK_PX, RESAMPLING]                # блоки другого расчёта (с другими параметрами) не подхватываются
BLK_DIR = OUT / f"_blocks_{NAME}_{MODE}_{hashlib.md5(json.dumps(SETTINGS, default=str).encode()).hexdigest()[:6]}"
BLK_DIR.mkdir(exist_ok=True)
print(f'{AOI}: {NAME}, режим {MODE}; сетка {CRS} {W}×{H} пикс., блоков {len(BLOCKS)}')

# %% [markdown]
# ## Шаг 5. Список сцен
# Печатается число сцен по месяцам и тайлам. Обычно 15–40 сцен на тайл за 4–5 месяцев (после фильтра облачности).

# %%
ITEMS = find_scenes(WORK_GEOM, D0, D1)
if not ITEMS and SOURCE == 'stac' and COLLECTION == 'sentinel-2-c1-l2a':
    print('в sentinel-2-c1-l2a сцен нет — беру sentinel-2-l2a'); COLLECTION = 'sentinel-2-l2a'; ITEMS = find_scenes(WORK_GEOM, D0, D1)
assert ITEMS, 'сцен не найдено: проверьте даты, область и MAX_CLOUD'
df = pd.DataFrame(ITEMS)
print('сцен:', len(df), '| тайлов MGRS:', df.tile.nunique(), '| дат:', df.date.nunique())
display(df.assign(month=df.date.str[5:7]).pivot_table(index='tile', columns='month', values='id', aggfunc='count', fill_value=0))

# %% [markdown]
# ## Шаг 6. Расчёт по блокам
# Каждый блок сохраняется на Диск в `out/<AOI>/_blocks_…` сразу после расчёта. Ячейку можно прервать или потерять
# сеанс — при повторном запуске готовые блоки пропускаются.

# %%
t_start = time.time(); done = 0
for i, w in enumerate(BLOCKS):
    f = BLK_DIR / f'b{int(w.row_off) // BLOCK_PX:03d}_{int(w.col_off) // BLOCK_PX:03d}.tif'
    if f.exists():
        continue
    t0 = time.time(); bt = rasterio.windows.transform(w, T); shp = (int(w.height), int(w.width))
    c = block_composite(bt, shp, CRS, ITEMS)         # тайлы, не задевающие блок, отсекаются внутри по окну
    inside = rasterize([(WORK_LOCAL, 1)], out_shape=shp, transform=bt, fill=0, dtype='uint8').astype(bool)
    bands = to_bands(c, inside)
    prof = dict(driver='GTiff', width=shp[1], height=shp[0], count=5, dtype='uint8', crs=CRS, transform=bt,
                compress='deflate', tiled=True)
    with rasterio.open(f, 'w', **prof) as d:
        d.write(bands)
    done += 1; el = time.time() - t_start; left = (len(BLOCKS) - i - 1) * el / done
    print(f'блок {i + 1}/{len(BLOCKS)}: {time.time() - t0:.0f} с, наблюдений (медиана) '
          f'{int(np.median(bands[4][bands[4] > 0])) if (bands[4] > 0).any() else 0}; осталось ≈ {left / 60:.0f} мин')
print('все блоки готовы')

# %% [markdown]
# ## Шаг 7. Сборка итоговых файлов
# Каналы: `diff` → 1 = max NDVI, 2 = min NDVI, 3 = max − min; `std` → 1 = max, 2 = min, 3 = СКО NDVI за период.
# Значения 1–255, вне области — 0 и внутренняя маска (в QGIS прозрачно). JPEG без цветовой субдискретизации
# (каналы не «цвета», поэтому каждый хранится в полном разрешении).

# %%
LOCAL = Path('/content/composite_tmp') if not LOCAL_TEST else Path(os.environ.get('WORK_DIR', '/tmp/composite_tmp'))
LOCAL.mkdir(parents=True, exist_ok=True)
FILES = {}
for v in (['diff', 'std'] if MODE == 'test' else VARIANTS):
    fname = NAME + ('_std' if v == 'std' else '') + ('_test' if MODE == 'test' else '') + '.tif'
    prof = dict(driver='GTiff', width=W, height=H, count=3, dtype='uint8', crs=CRS, transform=T, tiled=True,
                blockxsize=512, blockysize=512, interleave='pixel', BIGTIFF='IF_SAFER')
    prof.update(dict(compress='JPEG', jpeg_quality=JPEG_QUALITY, photometric='RGB') if COMPRESS == 'JPEG'
                else dict(compress='DEFLATE', predictor=2))
    with rasterio.open(LOCAL / fname, 'w', **prof) as dst:
        for w in BLOCKS:
            with rasterio.open(BLK_DIR / f'b{int(w.row_off) // BLOCK_PX:03d}_{int(w.col_off) // BLOCK_PX:03d}.tif') as s:
                a = s.read([1, 2, 3] if v == 'diff' else [1, 2, 4])
            dst.write(a, window=w); dst.write_mask(np.where(a[0] > 0, 255, 0).astype(np.uint8), window=w)
    shutil.copy(LOCAL / fname, OUT / fname); FILES[v] = OUT / fname
    print('Сохранено:', OUT / fname, f'({(OUT / fname).stat().st_size / 1e6:.1f} МБ)')
if not KEEP_BLOCKS and MODE == 'full':
    shutil.rmtree(BLK_DIR, ignore_errors=True)

# %% [markdown]
# ## Шаг 8 (тест). Картинки и сравнение с композитом из GEE
# Показываются оба варианта (diff и std). Если задан `REFERENCE`, ваш композит переносится на ту же сетку и для
# каналов max / min / diff печатаются корреляция и среднее расхождение (в единицах 0–255).

# %%
import matplotlib.pyplot as plt
if MODE == 'test':
    imgs = {k: rasterio.open(p).read() for k, p in FILES.items()}
    panels = [('max/min/diff (новый)', imgs['diff']), ('max/min/std (новый)', imgs['std'])]
    if REFERENCE:
        ref = np.zeros((3, H, W), np.uint8)
        with rasterio.open(DRIVE_ROOT / REFERENCE) as s:
            for b in range(3):
                reproject(rasterio.band(s, b + 1), ref[b], dst_transform=T, dst_crs=CRS, resampling=Resampling.nearest)
        new = imgs['diff'].astype(float); m = (ref[0] > 0) & (new[0] > 0)
        rows = [dict(канал=n, корреляция=round(float(np.corrcoef(new[b][m], ref[b][m])[0, 1]), 3),
                     расхождение=round(float(np.abs(new[b][m] - ref[b][m]).mean()), 1),
                     сдвиг=round(float((new[b][m] - ref[b][m]).mean()), 1)) for b, n in enumerate(['max', 'min', 'diff'])]
        display(pd.DataFrame(rows))
        panels.append(('ваш композит из GEE', ref))
    fig, ax = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    asp = 1 / math.cos(math.radians(WORK_GEOM.centroid.y)) if GRID == 'gee' else 1     # пиксель 4326 не квадратный
    for a, (t, im) in zip(np.atleast_1d(ax), panels):
        a.imshow(np.moveaxis(im, 0, -1), aspect=asp); a.set_title(t); a.axis('off')
    plt.savefig(OUT / f'{NAME}_test_view.png', dpi=110, bbox_inches='tight'); plt.show()
