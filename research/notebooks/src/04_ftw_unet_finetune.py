# %% [markdown]
# # 04 · Дообучение FTW U-Net (PRUE) на российских полях
#
# **Зачем.** Готовые модели FTW обучены на 24 странах без России и на «европейском» понятии поля (участок одной
# культуры, без пара/залежи). Дообучение на ваших полях учит модель **вашему определению поля** и **вашим ландшафтам**
# (степь, лесостепь, лесополосы, залежь). Начинаем не с нуля, а с весов PRUE — поэтому хватает сотен–тысяч полей.
#
# **Что делает ноутбук.**
# 1. Собирает разметку из нескольких источников (каждый — со своим годом снимков):
#    - **ваши поля** (GeoPackage/GeoJSON; можно с колонкой типа использования — все типы считаются «полем»);
#    - **GloCAB** — открытый набор ручных контуров (Zenodo 10479122), автоматически берутся регионы России
#      (и Донбасса), список регионов печатается — лишние можно исключить;
#    - позже — ваши большие наборы по России (Липецк–Вологда, Подмосковье, Татарстан…): просто добавьте их в `SOURCES`.
# 2. Для каждого региона разметки в GEE экспортирует **многодатный вход FTW** (окно A + окно B, 8 каналов) за год разметки.
# 3. Режет снимки на фрагменты 256×256 (2,56 км) и строит маски «фон / поле / граница / неизвестно».
#    «Неизвестно» (не учитывается в обучении) — всё за пределами размеченной территории, чтобы неразмеченные поля
#    не учили модель «это фон».
# 4. Дообучает модель (сначала только декодер, потом целиком) и **честно проверяет** на отложенных регионах:
#    метрики «до» (исходная модель) и «после».
# 5. Сохраняет `.ckpt`, который принимает `ftw inference run` — его можно подставить в ноутбук 02 (`MODEL = путь`).
#
# **Среда:** **T4 GPU**. Время: подготовка данных 20–60 мин (экспорт GEE), обучение 20–60 мин.
#
# **Чего ждать (короткая проверка на Донбассе, 2026).** Обучение на западной половине разметки (27 фрагментов 1,28 км,
# 5 эпох на CPU), проверка на восточной (153 поля; между половинами зазор 2 км). Модель `FTW_PRUE_EFNET_B3_CCBY`:
#
# | | исходная | после дообучения |
# |---|---|---|
# | пиксели: IoU «поле» / «граница» | 0,67 / 0,19 | **0,81 / 0,34** |
# | полигоны (ноутбук 02): IoU, IoU≥0,7 | 0,43, 0,31 | **0,54, 0,41** |
# | охват залежи | 0,16 | **0,56** |
# | карта границ как барьеры в v7 (`L_base`): IoU | 0,64 (B5) | **0,68** (карта FTW Global даёт 0,65) |
#
# Это один снимок и один год, поэтому оценка оптимистичная. Но направление ясно: даже маленькая своя разметка учит
# модель залежи и нашим полям. Полноценное дообучение — на GPU, с GloCAB и вашими наборами.
#
# **Важно.** Год снимков должен совпадать с годом, когда поля были такими, как в разметке. Для ваших полей 2026 — 2026.
# Для GloCAB (оцифрован по снимкам 2016–2020) — год задаётся параметром; если в данных есть колонка года, она
# используется автоматически.

# %% [markdown]
# ## Шаг 0. Установка `ftw-tools` (один раз за сеанс)
# После установки: «Среда выполнения» → «Перезапустить сеанс», затем продолжайте с шага 1.

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec('ftw_tools') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ftw-tools[all]==2.0.0b5'], check=True)
    print('Установлено. Теперь: Среда выполнения → Перезапустить сеанс, и дальше с шага 1.')
else:
    import torch
    print('ftw-tools есть; torch', torch.__version__, '| GPU:', torch.cuda.is_available())

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры
# `SOURCES` — список источников разметки. Для каждого: путь (или `'glocab'`), год снимков, окна A/B и роль:
# `'train'` — учить, `'test'` — только проверять (отложенный регион; не участвует в обучении).
# Хорошая практика: ваш размеченный тестовый участок — `'test'`, всё остальное — `'train'`.
#
# Окна A/B — как в ноутбуке 02 (начало сезона и вторая половина). Ниже — ориентиры для юга; для Татарстана/центра
# сдвигайте на 2–3 недели позже, для Вологды — на 4–5 недель.

# %%
EE_PROJECT = 'my-project-20260614-499410'
BASE_MODEL = 'FTW_PRUE_EFNET_B3_CCBY'   # с чего дообучать: B3 быстрее; '…_B5' точнее, но дольше учится

SOURCES = [
    dict(name='don_gt', path='Colab_Notebooks/Agro_terra/test/gt_pieces.gpkg', year=2026,
         win_a=('2026-04-10', '2026-05-31'), win_b=('2026-07-15', '2026-08-31'), role='test'),
    dict(name='glocab', path='glocab', year=2019,
         win_a=('2019-04-10', '2019-05-31'), win_b=('2019-07-15', '2019-08-31'), role='train'),
    # Разметка на всю область, например Донецкую (если это проверенные поля, а не результат конвейера). Тестовый
    # участок и поля рядом с ним из неё вырезаются автоматически (см. конец шага 2):
    # dict(name='dnr_2026', path='Colab_Notebooks/Agro_terra/DON/DNR_type_2026.geojson', year=2026,
    #      win_a=('2026-04-10', '2026-05-31'), win_b=('2026-07-15', '2026-08-31'), role='train'),
    # dict(name='tat_2025', path='Colab_Notebooks/Agro_terra/fields/tat_2025.gpkg', year=2025,
    #      win_a=('2025-05-01', '2025-06-15'), win_b=('2025-07-25', '2025-09-10'), role='train'),
]
GLOCAB_REGIONS = 'russia'          # 'russia' — регионы России (+ Донбасс) автоматически; или список номеров из таблицы шага 2
GLOCAB_YEAR_COL = None             # имя колонки года в GloCAB, если есть (найдётся автоматически)

CHIP = 256                         # размер фрагмента, пикселей (10 м)
MIN_LABELED = 0.5                  # фрагмент берётся, если ≥ 50 % его пикселей размечены (не «неизвестно»)
BOUNDARY_PX = 1                    # толщина класса «граница» в пикселях (с каждой стороны линии)
VAL_FRAC = 0.15                    # доля train-фрагментов на валидацию (выбирается блоками, не случайными пикселями)
MAX_CHIPS_PER_SOURCE = 3000        # не больше стольких фрагментов с одного источника (~3 ГБ памяти; для разметки на всю область)

EPOCHS_FROZEN = 5                  # эпох только с декодером (кодировщик заморожен)
EPOCHS_FULL = 25                   # эпох целиком
BATCH = 8
LR = 3e-4
OUT_DIR = 'Colab_Notebooks/Agro_terra/research/ftw_finetune/out'
GEE_FOLDER = 'GEE_ftw_finetune'
RUN_NAME = 'ru_v1'

# %% include _common_gee.py

# %% [markdown]
# ## Шаг 2. Разметка: ваши поля + GloCAB
# GloCAB скачивается с Zenodo (≈17 МБ) один раз. Печатается таблица регионов (номер, число полей, центр, страна).
# По умолчанию берутся регионы, центр которых в России или в Донецкой/Луганской областях.
# **Каждый регион разметки GloCAB оцифрован целиком** (все поля внутри прямоугольного блока), поэтому всё внутри
# блока, что не поле, — это «фон»; всё вне блока — «неизвестно».

# %%
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio, rasterio.windows, zipfile, urllib.request
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'; DATA.mkdir(exist_ok=True)


def load_glocab():
    z = DATA / 'glocab.zip'
    if not z.exists():
        meta = json.loads(urllib.request.urlopen('https://zenodo.org/api/records/10479122', timeout=60).read())
        f = [x for x in meta['files'] if x['key'].lower().endswith('.zip')][0]
        print('скачиваю', f['key'], round(f['size'] / 1e6, 1), 'МБ'); urllib.request.urlretrieve(f['links']['self'], z)
    d = DATA / 'glocab'
    if not d.exists():
        zipfile.ZipFile(z).extractall(d)
    files = [p for p in d.rglob('*') if p.suffix.lower() in ('.shp', '.gpkg', '.geojson', '.json') and 'MACOSX' not in str(p)]
    parts = []
    for p in files:
        try:
            g = gpd.read_file(p)
        except Exception as e:
            print('пропуск', p.name, e); continue
        if g.empty or g.geometry.iloc[0].geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        g = g.to_crs(4326); g['src_file'] = p.stem; parts.append(g)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    print('GloCAB: файлов', len(parts), '| полигонов', len(g), '| колонки:', [c for c in g.columns if c != 'geometry'][:15])
    return g


def glocab_regions(g):
    """Регион = файл GloCAB (если файлов много) или ячейка сетки 1° по центру поля (если всё в одном файле)."""
    if g['src_file'].nunique() > 3:
        g['region'] = g['src_file'].astype('category').cat.codes
    else:
        c = g.representative_point()
        key = np.floor(c.x).astype(int).astype(str) + '_' + np.floor(c.y).astype(int).astype(str)
        g['region'] = key.astype('category').cat.codes
    return g


LABELS = []      # (source dict, GeoDataFrame полей EPSG:4326, геометрия размеченной территории)
for src in SOURCES:
    if src['path'] == 'glocab':
        g = glocab_regions(load_glocab())
        try:
            ne = gpd.read_file('https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip').to_crs(4326)
            rus = ne[ne['ADM0_A3'] == 'RUS'].union_all()
        except Exception:
            rus = shapely.box(27, 41, 180, 82)
        donbas = shapely.box(36.5, 46.8, 40.3, 50.2)
        tab = []
        for r, gg in g.groupby('region'):
            c = gg.union_all().centroid
            tab.append(dict(region=int(r), fields=len(gg), lon=round(c.x, 2), lat=round(c.y, 2),
                            russia=bool(rus.contains(c) or donbas.contains(c)), file=gg['src_file'].iloc[0]))
        tab = pd.DataFrame(tab); display(tab)
        keep = tab[tab.russia].region.tolist() if GLOCAB_REGIONS == 'russia' else list(GLOCAB_REGIONS)
        ycol = GLOCAB_YEAR_COL or next((c for c in g.columns if c.lower() in ('year', 'yr', 'date', 'img_year')), None)
        for r in keep:
            gg = g[g.region == r].copy(); s = dict(src); s['name'] = f"glocab_r{r}"
            if ycol is not None and pd.notna(gg[ycol].iloc[0]):
                y = int(str(gg[ycol].iloc[0])[:4])
                s['year'] = y; s['win_a'] = tuple(d.replace(str(src['year']), str(y)) for d in s['win_a'])
                s['win_b'] = tuple(d.replace(str(src['year']), str(y)) for d in s['win_b'])
            terr = shapely.box(*gg.total_bounds).intersection(gg.union_all().convex_hull.buffer(0.002))
            LABELS.append((s, gg[['geometry']].reset_index(drop=True), terr))
        print('GloCAB: взято регионов', len(keep))
    else:
        g = gpd.read_file(DRIVE_ROOT / src['path']).to_crs(4326)
        g = g[g.geometry.notna()]
        terr = g.buffer(0.001).union_all().buffer(-0.0005)        # размеченная территория: поля + узкие промежутки между ними
        LABELS.append((src, g[['geometry']].reset_index(drop=True), terr))
# Тестовая территория (+ ~2 км) и поля, задевающие её, убираются из обучающих источников: иначе проверка нечестная
# (например, если в SOURCES и вся разметка области в роли 'train', и тестовый участок из неё же в роли 'test').
tests = [t for s, g, t in LABELS if s['role'] == 'test']
if tests:
    hole = shapely.union_all(tests).buffer(0.02)
    for k, (s, g, t) in enumerate(LABELS):
        if s['role'] == 'train' and t.intersects(hole):
            cut = g.intersects(hole).values
            t2 = t.difference(shapely.union_all([hole] + list(g.geometry.values[cut])))
            LABELS[k] = (s, g[~cut].reset_index(drop=True), t2)
            print(f"{s['name']}: убрано {cut.sum()} полей у тестовой территории")
for s, g, t in LABELS:
    print(f"{s['name']:>14}: {len(g):6d} полей, год {s['year']}, роль {s['role']}")

# %% [markdown]
# ## Шаг 3. Снимки для каждого региона разметки (GEE → Диск)
# Тот же вход, что у модели FTW: медианы окна A и окна B, каналы B4,B3,B2,B8, uint16, UTM 10 м.
# Готовые файлы пропускаются; ячейку можно прервать и запустить снова.

# %%
IMG = DATA / 'images'; IMG.mkdir(exist_ok=True)
BANDS = ['B4', 'B3', 'B2', 'B8']
jobs = []
for s, g, terr in LABELS:
    name = f"{RUN_NAME}_{s['name']}_{s['year']}"
    if list(IMG.glob(f'{name}*.tif')):
        continue
    b = shapely.box(*g.total_bounds).buffer(0.01).bounds
    reg = ee.Geometry.Rectangle(list(b))
    img = (window_median(reg, *s['win_a'], BANDS).rename([f'{x}_A' for x in BANDS])
           .addBands(window_median(reg, *s['win_b'], BANDS).rename([f'{x}_B' for x in BANDS])))
    crs = f'EPSG:{utm_epsg((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)}'
    t = ee.batch.Export.image.toDrive(image=img.toUint16().clip(reg), description=name[:100], folder=GEE_FOLDER,
                                      fileNamePrefix=name, region=reg, scale=10, crs=crs, maxPixels=1e13, fileDimensions=16384)
    t.start(); jobs.append((s['name'], b, name, t))
if jobs:
    wait_tasks(jobs); collect_exports(jobs, GEE_FOLDER, IMG)
print('снимков:', len(list(IMG.glob('*.tif'))))

# %% [markdown]
# ## Шаг 4. Фрагменты 256×256 и маски
# Классы маски: 0 — фон, 1 — поле, 2 — граница (стык полей и край поля, `BOUNDARY_PX` пикселей), 3 — неизвестно
# (вне размеченной территории; в обучении не учитывается). Фрагменты сохраняются в `data/chips_<RUN_NAME>.npz`.


# %%
def make_mask(g, terr, transform, shape, boundary_px=1):
    """g — поля и terr — размеченная территория, уже в СК снимка. Маска: 0 фон, 1 поле, 2 граница, 3 неизвестно."""
    ids = rasterize([(geom, i + 1) for i, geom in enumerate(g.geometry.values)], out_shape=shape, transform=transform,
                    fill=0, dtype='int32') if len(g) else np.zeros(shape, np.int32)
    b = np.zeros(shape, bool)
    b[:, :-1] |= ids[:, :-1] != ids[:, 1:]; b[:, 1:] |= ids[:, :-1] != ids[:, 1:]
    b[:-1] |= ids[:-1] != ids[1:]; b[1:] |= ids[:-1] != ids[1:]
    if boundary_px > 1:
        from scipy.ndimage import binary_dilation
        b = binary_dilation(b, iterations=boundary_px - 1)
    m = np.where(ids > 0, 1, 0).astype(np.uint8); m[b] = 2
    known = rasterize([(terr, 1)], out_shape=shape, transform=transform, fill=0, dtype='uint8') > 0 if not terr.is_empty \
        else np.zeros(shape, bool)
    m[~known] = 3
    return m


# Снимок читается полосами высотой CHIP (с запасом в пару пикселей для линий границ на стыке полос), поэтому
# хватает памяти и на экспорт целой области.
X, Y, META = [], [], []
for s, g, terr in LABELS:
    n0 = len(X)
    for f in sorted(IMG.glob(f"{RUN_NAME}_{s['name']}_{s['year']}*.tif")):
        k = 0
        with rasterio.open(f) as src:
            gg = g.to_crs(src.crs); tt = gpd.GeoSeries([terr], crs=4326).to_crs(src.crs).iloc[0]; sidx = gg.sindex
            halo = BOUNDARY_PX + 1
            for r in range(0, src.height - CHIP + 1, CHIP):
                win = rasterio.windows.Window(0, r - halo, src.width, CHIP + 2 * halo)
                box = shapely.box(*rasterio.windows.bounds(win, src.transform))
                if not tt.intersects(box):
                    continue
                mask = make_mask(gg.iloc[sidx.query(box)], tt.intersection(box), src.window_transform(win),
                                 (CHIP + 2 * halo, src.width), BOUNDARY_PX)[halo:-halo]
                img = src.read(window=rasterio.windows.Window(0, r, src.width, CHIP)).astype(np.uint16)
                for c in range(0, src.width - CHIP + 1, CHIP):
                    mm = mask[:, c:c + CHIP]; im = img[:, :, c:c + CHIP]
                    if (mm != 3).mean() < MIN_LABELED or (im.sum(0) == 0).mean() > 0.05:
                        continue
                    X.append(im); Y.append(mm); META.append((s['name'], s['role'], r // CHIP, c // CHIP)); k += 1
        print(f"{f.name}: фрагментов {k}")
    if len(X) - n0 > MAX_CHIPS_PER_SOURCE:                     # большой источник (вся область) — случайная выборка
        keep = n0 + np.sort(np.random.default_rng(1).choice(len(X) - n0, MAX_CHIPS_PER_SOURCE, replace=False))
        X[n0:], Y[n0:], META[n0:] = [X[i] for i in keep], [Y[i] for i in keep], [META[i] for i in keep]
        print(f"{s['name']}: оставлено {MAX_CHIPS_PER_SOURCE} фрагментов")
X = np.stack(X); Y = np.stack(Y); META = pd.DataFrame(META, columns=['source', 'role', 'row', 'col'])
np.savez_compressed(DATA / f'chips_{RUN_NAME}.npz', X=X, Y=Y)
META.to_csv(DATA / f'chips_{RUN_NAME}.csv', index=False)
print('всего фрагментов', len(X), '| по ролям:', META.role.value_counts().to_dict(),
      '| доли классов:', {k: round(float((Y == k).mean()), 3) for k in range(4)})

# %% [markdown]
# ## Шаг 5. Разбиение: обучение / валидация / тест
# Тест — источники с ролью `'test'` (целиком). Валидация — `VAL_FRAC` train-фрагментов, выбранных **блоками** 4×4
# фрагмента (≈10×10 км), чтобы соседние фрагменты не попадали в обучение и проверку одновременно.

# %%
rng = np.random.default_rng(0)
test_idx = np.where(META.role == 'test')[0]
tr = META[META.role == 'train'].copy(); tr['blk'] = tr.source + '_' + (tr.row // 4).astype(str) + '_' + (tr.col // 4).astype(str)
blks = tr.blk.unique(); rng.shuffle(blks); nval = max(1, int(len(blks) * VAL_FRAC)) if len(blks) > 1 else 0
val_idx = tr.index[tr.blk.isin(blks[:nval])].values; train_idx = tr.index[~tr.blk.isin(blks[:nval])].values
print('train', len(train_idx), '| val', len(val_idx), '| test', len(test_idx))

# %% [markdown]
# ## Шаг 6. Модель, потери, метрики
# Модель берётся из реестра ftw-tools (веса качаются автоматически). Потеря — как у PRUE: log-cosh Dice + CE с весами
# классов (фон 0,05 / поле 0,2 / граница 0,75), «неизвестно» игнорируется. Вход — как в `ftw inference run`: деление на
# 3000 и увеличение снимка в 2 раза (`RESIZE`) перед сетью; ответ сети уменьшается обратно до 10 м.

# %%
import torch, torch.nn as nn
from ftw_tools.inference.model_registry import MODEL_REGISTRY
from ftw_tools.inference.models import load_model_from_checkpoint
from ftw_tools.training.losses import logCoshDiceCE
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ck_dir = Path(torch.hub.get_dir()) / 'checkpoints'; ck_dir.mkdir(parents=True, exist_ok=True)
BASE_CKPT = ck_dir / f'{BASE_MODEL}.ckpt'
if not BASE_CKPT.exists():
    torch.hub.download_url_to_file(MODEL_REGISTRY[BASE_MODEL].url, str(BASE_CKPT))
NORM, RESIZE = 3000.0, 2            # как в ftw-tools: image / 3000 и --resize_factor 2


def fwd(m, x):
    """Прямой проход как при инференсе FTW: снимок ×2 (билинейно) → сеть → логиты обратно на сетку 10 м."""
    h, w = x.shape[-2:]
    y = m(nn.functional.interpolate(x, scale_factor=RESIZE, mode='bilinear', align_corners=False))
    return nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


def new_model():
    m, _ = load_model_from_checkpoint(str(BASE_CKPT)); return m.to(DEV)


crit = logCoshDiceCE(class_weights=torch.tensor([0.05, 0.2, 0.75]), ignore_index=3, classes=3).to(DEV)


def batches(idx, shuffle=False, aug=False):
    idx = np.array(idx); (rng.shuffle(idx) if shuffle else None)
    for i in range(0, len(idx), BATCH):
        j = idx[i:i + BATCH]
        x = torch.from_numpy(X[j].astype(np.float32) / NORM); y = torch.from_numpy(Y[j].astype(np.int64))
        if aug:                                               # повороты/отражения (безопасно для надирной съёмки)
            k = int(rng.integers(4)); x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
            if rng.random() < .5: x, y = x.flip(3), y.flip(2)
            x = x * torch.empty(x.shape[0], x.shape[1], 1, 1).uniform_(0.95, 1.05)     # лёгкий разброс яркости по каналам
        yield x.to(DEV), y.to(DEV)


@torch.no_grad()
def evaluate_chips(model, idx):
    """Пиксельные IoU классов «поле» и «граница» (как в статьях FTW) на фрагментах idx."""
    model.eval(); inter = np.zeros(3); union = np.zeros(3)
    for x, y in batches(idx):
        p = fwd(model, x).argmax(1); ok = y != 3
        for c in (1, 2):
            inter[c] += ((p == c) & (y == c) & ok).sum().item(); union[c] += (((p == c) | (y == c)) & ok).sum().item()
    return dict(IoU_field=round(float(inter[1] / max(union[1], 1)), 3), IoU_boundary=round(float(inter[2] / max(union[2], 1)), 3))


base = new_model()
BEFORE = {'val': evaluate_chips(base, val_idx) if len(val_idx) else None, 'test': evaluate_chips(base, test_idx) if len(test_idx) else None}
print('исходная модель:', BEFORE); del base

# %% [markdown]
# ## Шаг 7. Обучение
# Сначала `EPOCHS_FROZEN` эпох учится только декодер (кодировщик заморожен — так не «ломаются» хорошие признаки),
# затем `EPOCHS_FULL` эпох вся сеть с меньшим шагом. Лучшая по валидации эпоха сохраняется. Прогресс печатается.

# %%
model = new_model()
scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda')
best, best_state, hist = -1, None, []
for phase, epochs, lr in (('decoder', EPOCHS_FROZEN, LR), ('full', EPOCHS_FULL, LR / 3)):
    for p in model.encoder.parameters():
        p.requires_grad = phase == 'full'
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    for ep in range(epochs):
        model.train(); tl, n = 0.0, 0
        for x, y in batches(train_idx, shuffle=True, aug=True):
            opt.zero_grad()
            with torch.autocast(DEV, enabled=DEV == 'cuda'):
                loss = crit(fwd(model, x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item(); n += 1
        sch.step()
        v = evaluate_chips(model, val_idx) if len(val_idx) else evaluate_chips(model, train_idx[:200])
        score = v['IoU_field'] + v['IoU_boundary']; hist.append(dict(phase=phase, epoch=ep, loss=tl / max(n, 1), **v))
        print(f"{phase} {ep + 1}/{epochs}: loss {tl / max(n, 1):.4f} | val {v}")
        if score > best:
            best, best_state = score, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
model.load_state_dict(best_state)
AFTER = {'val': evaluate_chips(model, val_idx) if len(val_idx) else None, 'test': evaluate_chips(model, test_idx) if len(test_idx) else None}
print('до:   ', BEFORE); print('после:', AFTER)
pd.DataFrame(hist).to_csv(OUT / f'{RUN_NAME}_history.csv', index=False)

# %% [markdown]
# ## Шаг 8. Сохранение модели в формате ftw-tools
# Файл `<RUN_NAME>_<BASE_MODEL>.ckpt` можно указать в ноутбуке 02: `MODEL = '/content/drive/MyDrive/…/….ckpt'`
# (в `ftw inference run` подставляется путь к файлу вместо имени из реестра).

# %%
ck = torch.load(str(BASE_CKPT), map_location='cpu', weights_only=False)
ck['state_dict'] = {f'model.{k}': v for k, v in best_state.items()}
ck.setdefault('hyper_parameters', {})['finetuned'] = dict(run=RUN_NAME, sources=[s['name'] for s, _, _ in LABELS],
                                                          before=BEFORE, after=AFTER)
out_ckpt = OUT / f'{RUN_NAME}_{BASE_MODEL}.ckpt'
torch.save(ck, out_ckpt)
m2, _ = load_model_from_checkpoint(str(out_ckpt))            # проверка: файл читается так же, как в ftw inference
print('Сохранено и проверено:', out_ckpt)
json.dump(dict(before=BEFORE, after=AFTER), open(OUT / f'{RUN_NAME}_metrics.json', 'w'), ensure_ascii=False, indent=1)

# %% [markdown]
# ## Шаг 9. Картинка: исходная модель vs дообученная на тестовом фрагменте

# %%
import matplotlib.pyplot as plt
if len(test_idx):
    base = new_model().eval(); model.eval()
    i = test_idx[np.argmax([(Y[j] == 1).mean() for j in test_idx])]
    x = torch.from_numpy(X[i:i + 1].astype(np.float32) / NORM).to(DEV)
    with torch.no_grad():
        p0 = fwd(base, x).argmax(1)[0].cpu().numpy(); p1 = fwd(model, x).argmax(1)[0].cpu().numpy()
    rgb = np.clip(X[i, [4, 5, 6]].transpose(1, 2, 0) / 3000, 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for a, (im, t) in zip(ax, [(rgb, 'снимок (окно B)'), (Y[i], 'разметка'), (p0, 'исходная модель'), (p1, 'после дообучения')]):
        a.imshow(im, vmin=0, vmax=3 if im.ndim == 2 else None, cmap='viridis' if im.ndim == 2 else None); a.set_title(t); a.axis('off')
    plt.savefig(OUT / f'{RUN_NAME}_example.png', dpi=110, bbox_inches='tight'); plt.show()

# %% [markdown]
# ## Что дальше
# - Если на тесте «после» лучше «до» — запускайте дообученную модель на всю область через ноутбук 02
#   (`MODEL = путь к .ckpt`) и сравнивайте полигоны с разметкой по вашим метрикам.
# - Добавляйте свои наборы полей в `SOURCES` (Липецк, Вологда, Татарстан, Подмосковье, Алтай…): чем разнообразнее
#   регионы и годы, тем лучше модель переносится. Держите 1–2 региона в роли `'test'` для честной проверки.
# - Вологда и вообще северные поля мельче и с другим сезоном: для них задавайте свои окна A/B.
