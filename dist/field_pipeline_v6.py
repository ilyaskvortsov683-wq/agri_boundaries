# %% 0_intro [markdown]
# # Поля Самарской области: выделение → дополнение → классификация (v6)
#
# Один ноутбук, **шесть этапов (+ 1b — разбивка)**. Каждый включается флагом `1`/`0` в `STAGES`, и каждый может работать
# **на ваших полях** (`IN_LAYERS`) — например, только классификация по уже готовому вектору.
#
# | этап | что делает | вход |
# |---|---|---|
# | 1 | **Основные поля** (розовые и тёмные): суперпиксели → слияние → барьеры-лесополосы → пиксельное уточнение | растр (+ WorldCover) |
# | 1b | **Разбивка слипшихся полигонов**: режет полигон, в который слиплось несколько полей, по длинным тонким линиям (лесополосы, дороги, межи) и ступенькам тона | поля этапа 1 (или свои) |
# | 2 | **Мелкие поля**: берутся из тех же сегментов, что и этап 1 (без нового прохода по растру), но с мягкими порогами по площади; пересекающиеся с уже найденными отбрасываются | поля этапа 1 (или свои) |
# | 3 | **«Жёлтые» поля** (высокий min NDVI): тоже из тех же сегментов, строже по форме и по WorldCover | поля после этапа 2 (или свои) |
# | 4 | **Пятна**: вырезает из больших полей участки, резко отличающиеся по тону (жёлтые/тёмные в розовых, розовые в тёмных) | любые поля |
# | 5 | **Уточнение по подложке** ~1 м (XYZ-тайлы), watershed по градиенту подложки, 3 прохода | любые поля |
# | 6 | **Классификация** (вместе с этапом 4 — один проход по полям; при включённом этапе 5 по умолчанию после него): засеянное / пар / неиспользуемое (пиксельная карта → сглаживание → класс поля, второй класс при доле ≥ 25 %, разбивка) | любые поля |
#
# **Что нового в v6.** Линии (лесополосы, дороги) и ступеньки тона (граница другого класса) стали барьерами ДО слияния (`LINE_SENS`, `LINE_EDGE_SENS`); границы выпрямляются
# (`REGULARIZE_TOL_M`); готовые полигоны всех типов должны быть на ≥ 50 % пашней WorldCover (`CROP_POLY_MIN`); пороги классов подобраны по размеченным полям;
# у шва тайла свой потолок площади (`SEAM_MAX_REGION_HA`).
#
# **Что было в v5.** Гигантские полигоны (десятки тысяч га) возникали из-за *цепного слияния*: склейка через швы тайлов сливала соседние поля одного тона
# без проверки границы, а слияние внутри тайла не ограничивало размер. Теперь слияние (внутри тайла и через швы) ограничено по размеру (`MAX_REGION_HA`)
# и компактности (`MIN_COMPACT`), а на шве проверяется резкость границы. Режим сравнения (`COMPARE_PRESETS=True`, `RUN_MODE='test'`) прогоняет **каждый пресет
# через все включённые этапы**, сохраняя только слой после этапа 1 и финал (`fields_final_test_<пресет>.gpkg`), и печатает сводную таблицу.
#
# **Скорость.** Этапы 1+2+3 делают *одну* сегментацию растра (самая дорогая часть, ~75 % времени): сегменты считаются один раз и фильтруются
# под основные/мелкие/жёлтые поля. Этапы 4+6 читают окно каждого поля один раз. Тайлы считаются параллельно (`N_WORKERS`).
#
# **Как подключить свои поля.** В `IN_LAYERS` укажите путь к GeoJSON/GPKG/Parquet для нужного этапа — он возьмёт ваш слой
# вместо результата предыдущего этапа (любая CRS; атрибут `kind` необязателен). Если этап пропущен и своего слоя нет,
# а `RESUME=True`, берётся последний сохранённый `fields_after_stageN` из `OUT_DIR`.
#
# **Порядок работы:** `RUN_MODE='test'` (окно в центре) → проверить → `RUN_MODE='full'`.
# ⚠️ Подложка (этап 5): массовое скачивание ограничено условиями провайдера; для области целиком это ~10⁵–10⁶ тайлов — запускайте по участкам.

# %% 1_install
!pip -q install rasterio scikit-image pyarrow pillow requests scikit-learn
import shapely, geopandas, skimage
print('shapely', shapely.__version__, '| geopandas', geopandas.__version__, '| skimage', skimage.__version__)

# %% 3_imports
import os, math, json, time, hashlib, shutil, io, zipfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import STRtree
from shapely.geometry import shape as shp_shape, box as shp_box, Polygon, MultiPolygon
from shapely.affinity import affine_transform
import rasterio
import rasterio.warp
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.features import shapes as rio_shapes, rasterize
from rasterio.transform import Affine
from scipy.ndimage import (gaussian_filter, grey_opening, grey_closing, binary_dilation,
                           label as ndi_label, distance_transform_edt, maximum as ndi_maximum)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold
from scipy.signal import fftconvolve
from tqdm.auto import tqdm

from scipy.ndimage import uniform_filter

# %% 2_params
# ───────────── Пути ─────────────
DRIVE_DIR      = '/content/drive/MyDrive/Colab_Notebooks/Agro_terra/TAT'
COMPOSITE_NAME = 'TA_composite_2026_0504_0910_8b.tif'   # 1=max NDVI, 2=min NDVI, 3=diff (0–255); 4-й канал (маска) не обязателен
BOUNDARY_NAME  = 'TAT.geojson'                        # None — без границы области
OUT_DIR        = f'{DRIVE_DIR}/fields_out'
LOCAL_DIR      = '/content/work'
COPY_LOCAL     = True

# ───────────── Скорость ─────────────
N_WORKERS = min(os.cpu_count() or 1, 3)   # процессов для тайлов (в Colab обычно 2); 1 — без параллелизма. Память: ~1.5 ГБ на процесс при TILE=2048

# ───────────── Режим ─────────────
RUN_MODE    = 'test'        # 'test' — окно в центре растра; 'full' — вся область
TEST_SIZE   = 5048
TEST_CENTER = (0.5, 0.5)
TILE, HALO  = 2048, 64
MIN_TILE_CROP = 0.002

# ───────────── Этапы: 1 — выполнять, 0 — пропустить ─────────────
STAGES = {1: 1,      # основные поля (розовые и тёмные)
          '1b': 1,   # разбивка слипшихся полигонов (много полей в одном) по линиям и ступенькам тона
          2: 1,      # мелкие поля
          3: 1,      # «жёлтые» поля
          4: 1,      # пятна-вырезы в больших полях
          5: 0,      # уточнение границ по подложке
          6: 1}      # классификация

# Свои поля на входе этапа: путь к GeoJSON/GPKG/Parquet. None — брать результат предыдущего этапа.
# Для этапа 1 входа нет (он работает по растру). Пример: IN_LAYERS = {6: f'{DRIVE_DIR}/my_fields.geojson'} и STAGES = {1:0,'1b':0,2:0,3:0,4:0,5:0,6:1}
IN_LAYERS = {'1b': None, 2: None, 3: None, 4: None, 5: None, 6: None}
RESUME    = True            # если этап пропущен и своего слоя нет — читать последний fields_after_stageN.gpkg из OUT_DIR

# ───────────── GEE / WorldCover ─────────────
USE_WORLDCOVER = True       # False — без WorldCover (вся площадь считается пашней; для проверок без GEE)
EE_PROJECT    = 'my-project-20260614-499410'
WC_ASSET      = 'ESA/WorldCover/v200'   # 2021
AGRI_CLASSES  = [40]                    # Cropland

# ───────────── Тон «жёлтого» (общий для этапов 1, 3, 4, 6) ─────────────
YELLOW_MIN_THR = 85         # средний min-канал региона выше порога → «жёлтое» (многолетняя растительность); розовые/тёмные ≤ ~60
REGULARIZE_TOL_M = 18       # финальное выпрямление границ: допуск упрощения (м) с общей топологией соседних полей (0 — выкл.); эталонные поля ≈ 10 вершин/км
CROP_POLY_MIN  = 0.5        # ГОТОВЫЙ полигон любого типа (основные, мелкие, жёлтые) должен быть не менее чем на эту долю пашней WorldCover (0 — выкл.)

# ───────────── Этап 1: основные поля ─────────────
BASE = dict(
    # сегментация (каналы 0–255)
    SMOOTH_SIGMA=1.0,      # сглаживание перед SLIC
    SP_SIZE_PX=110,        # средний размер суперпикселя, px
    COMPACTNESS=0.2,       # SLIC: больше → квадратнее, меньше → точнее по цвету
    MERGE_THR=48,          # макс. расстояние между средними регионов для слияния
    JUMP_THR=14,           # макс. средний скачок на общей границе (0 — выкл.): не даёт слиться полям с резкой границей
    MIN_CONTACT=3,         # мин. длина общей границы для слияния, px
    MIN_REGION_PX=100,     # регионы меньше поглощаются соседом
    USE_WC_IN_SEGMENTATION=False, WC_WEIGHT=0.5,
    # лесополосы-барьеры
    BARRIER_THR=25,        # порог top-hat (0 — выкл.); 18–28 ловит слабые полосы
    BARRIER_WIDTH_PX=5,    # макс. ширина полосы, px
    BARRIER_MIN_PX=50,     # мин. размер связной компоненты барьера, px (короткие полоски внутри полей игнорируются)
    BARRIER_DILATE=0,      # утолщение барьера; 0 — границы доходят до полосы
    BARRIER_WC_CLASSES=[], # доп. барьеры из WorldCover, напр. [10] = Tree cover
    # пиксельное уточнение
    RELAX_ITERS=6,         # 0 — выкл.
    RELAX_SIGMA=1.0,
    RELAX_BETA=250,        # штраф Потса: больше → глаже границы
    # векторизация и отбор
    MIN_FIELD_HA=15.5, FILL_HOLES_HA=10.0, SIMPLIFY_PX=2.0, MIN_WIDTH_PX=4,
    CROP_FRAC_MIN=0.4,     # доля cropland WorldCover в регионе для признания полем
    # ограничения против гигантских «матричных» регионов (v5): слияние не должно создавать регион, который …
    MAX_REGION_HA=800,     # … крупнее этой площади (0 — выкл.); типичное поле Самарской обл. 50–300 га
    SEAM_MAX_REGION_HA=2500, # потолок площади для склейки через шов тайла (там уже проверена резкость границы; настоящие поля бывают > 1000 га)
    MIN_COMPACT=6.5,       # … «рыхлее» этого порога компактности C = n/√detΣ (прямоугольник/круг ≈ 12, цепочка полей ≪ 8; 0 — выкл.)
    COMPACT_MIN_HA=30,     # компактность проверяется только у регионов крупнее этой площади
    SEAM_CONSTRAIN=True,   # склейка через швы тайлов: с проверкой резкости границы (JUMP_THR), размера и компактности; False — как в v3
    WC_BARRIER_MIN_PX=30,  # BARRIER_WC_CLASSES: компоненты WorldCover мельче этого размера (px) игнорируются (шум)
    # длинные прямые линии/ступеньки тона как барьеры ДО слияния (v6; ловят слабые границы между соседними полями)
    LINE_SENS=0.45,        # чувствительность к ТОНКИМ линиям — лесополосы, дороги, межи (0 — выкл.); ≥1 — пересегментация текстуры
    LINE_EDGE_SENS=0.25,   # чувствительность к СТУПЕНЬКАМ тона (граница другого класса); меньше → слабые границы между участками одного массива не режутся; 0 — только тонкие линии
    LINE_MIN_PX=40,        # линии короче (px) игнорируются (1 px ≈ 6×10 м)
    LINE_SIGMA_ALONG=4,    # длина усреднения вдоль линии; больше → только прямые длинные линии
    LINE_DIL=1,            # утолщение линии-барьера, px (закрывает разрывы)
    LINE_FILL=True,        # пиксели линий отдаются ближайшему региону (поля соприкасаются по линии)
)

PRESETS = {
    # ── семейство L (v6): линии/ступеньки как барьеры ДО слияния + ограничения размера/компактности/швов ──
    # LINE_SENS — чувствительность к тонким линиям (лесополосы, дороги), LINE_EDGE_SENS — к ступенькам тона (граница класса).
    # Культуры внутри одного массива не разделяются: ступеньки слабее LINE_EDGE_SENS игнорируются.
    'L_base':     dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0),                                   # основной кандидат; леса/застройка/вода — барьеры, сёла внутри поля не заливать
    'L_ridge':    dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, LINE_EDGE_SENS=0, SPLIT=dict(EDGE_SENS=0)),            # режет только по тонким линиям
    'L_edges':    dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, LINE_EDGE_SENS=0.45, SPLIT=dict(SENS=0.8, EDGE_SENS=0.8)),   # режет и по более слабым ступенькам (больше мелких кусков)
    'L_soft':     dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, LINE_SENS=0.3, SPLIT=dict(SENS=0.6)),                   # только самые явные линии
    'L_strong':   dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, LINE_SENS=0.7, SPLIT=dict(SENS=1.0)),                   # больше тонких линий
    'L_long':     dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, LINE_SIGMA_ALONG=8, LINE_MIN_PX=60),                    # только длинные прямые линии
    'L_nowc':     dict(),                                                              # без WorldCover-барьеров
    'L_cap400':   dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, MAX_REGION_HA=400),
    'L_sharp':    dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, MERGE_THR=40, JUMP_THR=10, BARRIER_THR=22),
    'L_smooth':   dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, SMOOTH_SIGMA=1.2, RELAX_BETA=600),                      # глаже границы
    'L_split_off': dict(BARRIER_WC_CLASSES=[10, 50, 80, 90], FILL_HOLES_HA=2.0, NO_STAGES=['1b']),                                      # без этапа 1b
    # ── v5 и v3 для сравнения (в сравнении по умолчанию не участвуют) ──
    'OLD_C_base': dict(LINE_SENS=0, SIMPLIFY_PX=0.8),
    'OLD_A_best': dict(LINE_SENS=0, SIMPLIFY_PX=0.8, MAX_REGION_HA=0, MIN_COMPACT=0, SEAM_CONSTRAIN=False),
}

STAGE1_PRESET   = 'L_base'         # что использовать
COMPARE_PRESETS = True               # True (+ RUN_MODE='test'): каждый пресет проходит ВСЕ включённые этапы; сохраняются слой после этапа 1 и финал
COMPARE_NAMES   = None               # какие пресеты сравнивать: список имён; None — все, кроме начинающихся с 'OLD'
COMPARE_CENTERS = [(0.5, 0.5)]       # центры окон для картинок сравнения (доли области по x, y); на каждый — отдельный PNG
COMPARE_WIN_PX  = 1400               # размер окна картинок, px
GIANT_HA        = 600                # полигон крупнее — «гигант»: считается в сводной таблице и в диагностике

# ───────────── Этап 1b: разбивка слипшихся полигонов ─────────────
# Режет полигон на поля по длинным тонким линиям (лесополосы, дороги, межи) и резким ступенькам тона внутри него.
SPLIT = dict(
    KINDS=('main', 'input'),   # к каким полям применять (kind); None — ко всем
    MIN_HA=40,                 # полигоны меньше этой площади не трогаем
    SENS=0.8,                  # чувствительность к тонким линиям: больше → режет по более слабым (1.4 — много разбивок, 0.6 — только явные)
    EDGE_SENS=0.25,            # чувствительность к ступенькам тона (граница другого класса); 0 — резать только по линиям
    PIECE_MIN_HA=8,            # кусок меньше этого не становится отдельным полем (отдаётся соседу)
    LINE_MIN_PX=30,            # минимальная длина линии, px (1 px ≈ 8 м)
    LINE_DIL=2,                # «толщина» разреза, px: закрывает разрывы линии
    SIGMA_ALONG=4,             # длина усреднения вдоль линии; больше → ловит только прямые длинные линии
    EDGE_PX=2,                 # линии ближе этого к границе поля игнорируются
    MAX_WINDOW_PX=40_000_000,  # окно полигона больше этого не обрабатывается (защита памяти)
)

# ───────────── Этап 2: мелкие поля ─────────────
# Берутся из регионов единственного прохода сегментации (этап 1) — без повторного чтения растра. Ниже — только фильтры.
S2 = dict(MIN_FIELD_HA=1.0, MIN_WIDTH_PX=5, FILL_HOLES_HA=0.3, CROP_FRAC_MIN=0.5, MAX_HOLE_FRAC=0.25)
S2_MIN_SOLIDITY = 0.65               # площадь / площадь выпуклой оболочки: отсекает «обломки» и полосы между полями
S2_FINE_PASS = False                 # True: дополнительный проход более мелкой сегментацией остатка (S2_FINE; в разы дольше)
S2_FINE = dict(SP_SIZE_PX=70, SMOOTH_SIGMA=0.8, COMPACTNESS=0.2, MERGE_THR=50, JUMP_THR=15, BARRIER_THR=25, RELAX_ITERS=4,
          MIN_REGION_PX=50, MIN_FIELD_HA=1.0, MIN_WIDTH_PX=5, FILL_HOLES_HA=0.3, CROP_FRAC_MIN=0.5)

# ───────────── Этап 3: «жёлтые» поля (из тех же регионов) ─────────────
S3 = dict(MIN_FIELD_HA=15.0, MIN_WIDTH_PX=8, FILL_HOLES_HA=5.0, CROP_FRAC_MIN=0.7,   # CROP_FRAC_MIN не ниже, чем в BASE
          MAX_FIELD_HA=300, MAX_HOLE_FRAC=0.25)   # крупнее / с большей долей дыр — не поле (фон между полями)
S3_MIN_SOLIDITY = 0.6

# Вложенные поля: поле не может лежать внутри другого. Относится к мелким/жёлтым, добавленным на этапах 2 и 3.
NESTED_POLICY    = 'absorb'   # 'absorb' — вложенное поле поглощается внешним (дыра заливается; крупное внешнее поле пятна потом вырежет этап 4);
                              # 'drop' — вложенное поле удаляется, дыра остаётся; 'keep' — не трогать
NESTED_FRAC      = 0.9        # доля площади поля внутри внешнего контура другого (с заливкой дыр), с которой оно считается вложенным
NESTED_HOLE_FRAC = 0.5        # поле лежит в дыре: оно должно занимать не менее такой доли этой дыры (иначе дыра — это другое, поле остаётся)

# ───────────── Этап 4: пятна ─────────────
SPOT_KINDS         = ('main', 'input')   # к каким полям применять (kind); None — ко всем
SPOT_MIN_FIELD_HA  = 30                  # только «большие» поля
SPOT_MIN_HA        = 1.0                 # минимальная площадь пятна
SPOT_MAX_FRAC      = 0.35                # пятно больше этой доли поля — не вырезается (это уже другое поле/неоднородность)
SPOT_EDGE_PX       = 3                   # пятно должно отстоять от границы поля минимум на столько px (иначе это край, а не пятно)
SPOT_SIGMA         = 1.0
SPOT_DIST_PINK     = 90                  # в розовом поле: расстояние в плоскости (max,min) от тона поля, при котором пятно «явное»
SPOT_DIST_DARK     = 90                  # в тёмном поле: то же (учитываются только более светлые пятна: розовые/жёлтые)
DARK_MAX_THR       = 115                 # поле с медианным max ниже порога считается тёмным
SPOT_CUT           = True                # False — только слой spots, поля не менять
CLASSIFY_AFTER_REFINE = True             # если включён этап 5: классифицировать ПОСЛЕ уточнения границ (отдельный дешёвый проход);
                                         # False — классификация в одном проходе с этапом 4 (по контурам до уточнения)

# ───────────── Этап 5: уточнение по подложке ─────────────
REFINE_KINDS      = None      # None — все поля
BASEMAP_URL       = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'   # любой XYZ-шаблон; соблюдайте условия провайдера
BASEMAP_ZOOM      = 16        # 17 ≈ 0.72 м/px на широте 53°; 16 → 1.4 м (в 4 раза меньше тайлов)
BASEMAP_CACHE     = '/content/basemap_cache'
BASEMAP_WORKERS   = 6
REFINE_RES_M      = 1.0
REFINE_BLOCK_PX   = 2000
REFINE_MARGIN_PX  = 60
REFINE_PASSES = [             # от грубого к тонкому; e_* — полуширина неопределённой зоны, м
    dict(sigma=2.0, e_field=12, e_bg=6),
    dict(sigma=1.2, e_field=6,  e_bg=3.5),
    dict(sigma=0.8, e_field=3,  e_bg=1.5),
]
REFINE_PRIOR      = 0.3
REFINE_COMPACT    = 0.0
REFINE_MAX_AREA_CHANGE = 0.4
REFINE_AOI_M      = 6000      # test: только поля внутри квадрата такого размера в центре; None — все
REFINE_MAX_TILES  = 20000

# ───────────── Этап 6: классификация ─────────────
# Правила по сглаженным каналам пикселя (0–255). Порядок: неиспользуемое → пар → засеянное.
CLASS_MODE        = 'rules'   # 'rules' — пороги ниже; 'tree' — дерево решений, обученное на ваших размеченных полях (CLASS_TRAIN_*)
CLASS_NAMES       = {1: 'засеянное', 2: 'пар', 3: 'неиспользуемое'}
UNUSED_MIN_THR    = 75        # min ≥ порога → растительность не спадает весь сезон (залежь/травостой/кустарник)
FALLOW_MAX_THR    = 170       # max < порога → за сезон растительность не развилась (чистый пар / голая почва)
UNUSED_DIFF_MAX   = 130       # diff < порога (при max ≥ FALLOW_MAX_THR) → тоже «неиспользуемое»: слабый размах NDVI за сезон (0 — выкл.)
FALLOW_MIN_FIELD_HA = 15.5    # поля меньше этой площади паром не бывают: пиксели «пар» в них считаются засеянными
CLASS_SIGMA       = 1.0       # сглаживание каналов перед правилами
CLASS_SMOOTH_WIN  = 15        # мажоритарное окно, px (0 — выкл.): крупнее → ровнее границы между классами внутри поля
CLASS_MINOR_FRAC  = 0.3       # второй класс записывается, если его доля ≥ 30 %
SPLIT_BY_CLASS    = True      # слой с разбивкой поля по границе между двумя основными классами
SPLIT_MIN_HA      = 8.0       # куски разбивки мельче поглощаются соседним классом
CLASS_TRAIN_LAYER = None      # для 'tree': GeoJSON/GPKG с размеченными полями
CLASS_TRAIN_COL   = None      # колонка с классом (значения — любые метки)
# Пороги выше подобраны по размеченным полям Донбасса (hold-out: залежь 41 % → 75–82 %, пар 81 % → 91 % при той же общей точности).
# Для Самарской области раньше были 150 / 85 / 0 — вернуть: FALLOW_MAX_THR, UNUSED_MIN_THR, UNUSED_DIFF_MAX = 150, 85, 0

# %% 4_segmentation_funcs
def smooth_feats(feats, sigma):
    if sigma <= 0:
        return feats.astype(np.float32)
    return np.stack([gaussian_filter(feats[..., k], sigma) for k in range(feats.shape[-1])], -1).astype(np.float32)


def _compact(lab):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return remap[lab]


def detect_barriers(feats, valid, wcmap=None):
    """Лесополосы/дороги: узкие яркие линии в канале min (high min NDVI) и узкие тёмные в diff (low diff).
    Белый top-hat: x - opening(x, s×s) оставляет детали уже s px."""
    m = np.zeros(valid.shape, bool)
    if BARRIER_THR:
        s = int(BARRIER_WIDTH_PX) + 2
        f = smooth_feats(feats[..., :3], 0.6)
        mn, df = f[..., 1], f[..., 2]
        score = np.maximum(mn - grey_opening(mn, size=(s, s)), grey_closing(df, size=(s, s)) - df)
        m = (score > BARRIER_THR) & valid
        if BARRIER_MIN_PX > 1 and m.any():
            cc, _ = ndi_label(m, structure=np.ones((3, 3)))
            keep = np.bincount(cc.ravel()) >= BARRIER_MIN_PX
            keep[0] = False
            m = keep[cc]
    if len(BARRIER_WC_CLASSES) and wcmap is not None:
        wb = np.isin(wcmap, BARRIER_WC_CLASSES)
        if WC_BARRIER_MIN_PX > 1 and wb.any():                    # одиночные «лесные»/«застроенные» пиксели WorldCover внутри поля — шум
            cc, _ = ndi_label(wb, structure=np.ones((3, 3)))
            keep = np.bincount(cc.ravel()) >= WC_BARRIER_MIN_PX
            keep[0] = False
            wb = keep[cc]
        m = m | wb
    if BARRIER_DILATE > 0 and m.any():
        m = binary_dilation(m, structure=np.ones((3, 3), bool), iterations=int(BARRIER_DILATE))
    return m & valid


def _moments(flat, shape, K, x0=0.0, y0=0.0):
    """Суммы (x, y, x², y², xy) по регионам: из них считается компактность региона без геометрических операций."""
    h, w = shape
    yy, xx = np.divmod(np.arange(h * w, dtype=np.int64), w)
    xx = xx.astype(np.float64) + x0
    yy = yy.astype(np.float64) + y0
    cols = [xx, yy, xx * xx, yy * yy, xx * yy]
    return np.stack([np.bincount(flat, weights=c_, minlength=K + 1) for c_ in cols], 1)


def _compact_ratio(n, m):
    """C = n / sqrt(det Σ): прямоугольник ≈ 12, круг ≈ 12.6, цепочка полей/L-образный «хвост» заметно меньше.
    Инвариантна к сдвигу, повороту и растяжению осей (пиксели 6×10 м не мешают)."""
    mx, my = m[0] / n, m[1] / n
    vxx, vyy, vxy = m[2] / n - mx * mx + 1 / 12, m[3] / n - my * my + 1 / 12, m[4] / n - mx * my
    det = vxx * vyy - vxy * vxy
    return n / np.sqrt(det) if det > 1e-9 else np.inf


_SPLIT_K = {}
_SPLIT_BASE = dict(E_MIN=6.3, R_MIN=3.15, E_MAX=9.1)   # пороги отклика при SENS=1 (в единицах NDVI 0–255); SENS делит их


def _line_kernel(theta, s_across, s_along, kind):
    key = (round(float(theta), 4), s_across, s_along, kind)
    if key not in _SPLIT_K:
        r = int(3 * max(s_across, s_along)) + 1
        y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        u = x * np.cos(theta) + y * np.sin(theta)                 # поперёк линии
        v = -x * np.sin(theta) + y * np.cos(theta)                # вдоль линии
        gk = np.exp(-u ** 2 / (2 * s_across ** 2) - v ** 2 / (2 * s_along ** 2))
        k = (-u / s_across ** 2 * gk) if kind == 'edge' else ((1 - u ** 2 / s_across ** 2) * gk)
        if kind == 'ridge':
            k = k - k.mean()
        _SPLIT_K[key] = (k / np.abs(k).sum()).astype(np.float32)
    return _SPLIT_K[key]


def _oriented(ch, kind, s_across, s_along, n=12):
    out = np.zeros(ch.shape, np.float32)
    for i in range(n):
        r = fftconvolve(ch, _line_kernel(np.pi * i / n, s_across, s_along, kind), mode='same')
        np.maximum(out, np.abs(r) if kind == 'edge' else r, out=out)
    return out


def _line_resp(feats, sens, s_along, edge_sens=None):
    """sens — чувствительность к ТОНКИМ линиям (лесополосы, дороги, межи: ridge в канале min);
    edge_sens — к СТУПЕНЬКАМ тона (граница другого класса/культуры: edge в каналах min и max); по умолчанию = sens.
    Ступеньки между участками разных культур внутри одного массива резать не нужно — для этого edge_sens делают меньше."""
    edge_sens = sens if edge_sens is None else edge_sens
    s = np.stack([gaussian_filter(feats[..., k], 0.7) for k in range(2)], -1).astype(np.float32)
    mx, mn = s[..., 0], s[..., 1]
    b = _SPLIT_BASE
    comps = [_oriented(mn, 'ridge', 1.3, s_along) * sens / b['R_MIN']]
    if edge_sens:
        comps += [_oriented(mn, 'edge', 1.3, s_along) * edge_sens / b['E_MIN'], _oriented(mx, 'edge', 1.3, s_along) * edge_sens / b['E_MAX']]
    return np.maximum.reduce(comps)


def split_line_response(feats, sens, s_along, block=2500, halo=24, edge_sens=None):
    """Нормированный отклик «линия/ступенька» (≥ 1 — уверенная линия). Каналы: min (ступенька и тонкая линия), max (ступенька).
    Окна больше block×block считаются блоками с перекрытием (экономия памяти для полигонов-гигантов)."""
    h, w = feats.shape[:2]
    if h * w <= 1.5 * block * block:
        return _line_resp(feats, sens, s_along, edge_sens)
    R = np.zeros((h, w), np.float32)
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            ra, rb, ca, cb = max(r0 - halo, 0), min(r0 + block + halo, h), max(c0 - halo, 0), min(c0 + block + halo, w)
            Rb = _line_resp(feats[ra:rb, ca:cb], sens, s_along, edge_sens)
            R[r0:r0 + block, c0:c0 + block] = Rb[r0 - ra:r0 - ra + min(block, h - r0), c0 - ca:c0 - ca + min(block, w - c0)]
    return R


def detect_line_barriers(feats, valid, sens, s_along, min_px, edge_sens=None):
    """Длинные прямые тонкие линии и ступеньки тона (межи, дороги, слабые лесополосы, границы соседних культур).
    Ориентированные фильтры усредняют ВДОЛЬ линии, поэтому ловят линии, невидимые для слияния по средним (скачок < JUMP_THR).
    Результат: скелет линий (1 px) длиной ≥ min_px. Операция локальна: не зависит от разбиения на тайлы (при достаточном HALO)."""
    R = split_line_response(feats, sens, s_along, edge_sens=edge_sens)
    R = np.where(valid, R, 0)
    cand = apply_hysteresis_threshold(R, 0.6, 1.0)
    lab, n = ndi_label(skeletonize(cand), structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(valid.shape, bool)
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_px
    keep[0] = False
    return keep[lab]


def _build_graph(lab, feats):
    """Граф соседства: рёбра (u<v), число контактов c, сумма скачков значений js, размеры cnt, суммы признаков."""
    K = int(lab.max())
    a = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    b = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    dh = feats[:, :-1] - feats[:, 1:]
    dv = feats[:-1, :] - feats[1:, :]
    j = np.concatenate([np.sqrt((dh * dh).sum(-1)).ravel(), np.sqrt((dv * dv).sum(-1)).ravel()])
    m = (a != b) & (a > 0) & (b > 0)
    a, b, j = a[m].astype(np.int64), b[m].astype(np.int64), j[m].astype(np.float64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ukey, inv, c = np.unique(lo * (K + 1) + hi, return_inverse=True, return_counts=True)
    js = np.bincount(inv, weights=j, minlength=ukey.size)
    flat = lab.ravel()
    cnt = np.bincount(flat, minlength=K + 1).astype(np.float64)
    sums = np.stack([np.bincount(flat, weights=feats[..., k].ravel(), minlength=K + 1)
                     for k in range(feats.shape[-1])], 1)
    mom = _moments(flat, lab.shape, K)
    return K, ukey // (K + 1), ukey % (K + 1), c.astype(np.float64), js, cnt, sums, mom


def _contract(K, u, v, c, js, cnt, sums, mom, mp):
    ids = np.unique(mp[1:])
    remap = np.zeros(K + 1, np.int64)
    remap[ids] = np.arange(1, ids.size + 1)
    mp = remap[mp]
    K2 = ids.size
    cnt2 = np.bincount(mp, weights=cnt, minlength=K2 + 1)
    sums2 = np.stack([np.bincount(mp, weights=sums[:, k], minlength=K2 + 1) for k in range(sums.shape[1])], 1)
    mom2 = np.stack([np.bincount(mp, weights=mom[:, k], minlength=K2 + 1) for k in range(mom.shape[1])], 1)
    u2, v2 = mp[u], mp[v]
    keep = u2 != v2
    u2, v2, c2, j2 = u2[keep], v2[keep], c[keep], js[keep]
    lo, hi = np.minimum(u2, v2), np.maximum(u2, v2)
    ukey, inv = np.unique(lo * (K2 + 1) + hi, return_inverse=True)
    return (K2, ukey // (K2 + 1), ukey % (K2 + 1), np.bincount(inv, weights=c2, minlength=ukey.size),
            np.bincount(inv, weights=j2, minlength=ukey.size), cnt2, sums2, mom2, mp)


def _hac(K, u, v, c, js, cnt, sums, mom, thr, min_contact, jump_thr, max_px=0, min_compact=0, compact_min_px=0):
    """Иерархическое слияние (центроидная связь) на графе: всегда сливается самая близкая пара с d < thr,
    общей границей >= min_contact px и средним скачком на границе < jump_thr. Средние пересчитываются.
    Ограничения против «цепного» слияния: объединённый регион не больше max_px и (если он ≥ compact_min_px)
    не менее компактен, чем min_compact (см. _compact_ratio)."""
    import heapq
    jt = jump_thr if jump_thr else np.inf
    mean = sums / np.maximum(cnt[:, None], 1)
    d0 = np.linalg.norm(mean[u] - mean[v], axis=1)
    ok = (d0 < thr) & (c >= min_contact) & (js / np.maximum(c, 1) < jt)
    adj = [dict() for _ in range(K + 1)]
    for a, b, w, jj in zip(u.tolist(), v.tolist(), c.tolist(), js.tolist()):
        e = [w, jj]
        adj[a][b] = e
        adj[b][a] = e                        # один и тот же список для обоих направлений
    ver = [0] * (K + 1)
    alive = [True] * (K + 1)
    parent = list(range(K + 1))
    cnt = cnt.copy(); sums = sums.copy(); mom = mom.copy()
    heap = [(float(d), int(a), int(b), 0, 0) for d, a, b in zip(d0[ok], u[ok], v[ok])]
    heapq.heapify(heap)
    while heap:
        d, a, b, va, vb = heapq.heappop(heap)
        if d >= thr:
            break
        if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb:
            continue
        n_ab = cnt[a] + cnt[b]
        if max_px and n_ab > max_px:
            continue
        if min_compact and n_ab >= compact_min_px and _compact_ratio(n_ab, mom[a] + mom[b]) < min_compact:
            continue
        if cnt[a] < cnt[b]:
            a, b = b, a
        alive[b] = False; parent[b] = a
        cnt[a] += cnt[b]; sums[a] += sums[b]; mom[a] += mom[b]; ver[a] += 1
        for n, e in adj[b].items():
            del adj[n][b]
            if n == a:
                continue
            if n in adj[a]:
                ex = adj[a][n]
                ex[0] += e[0]; ex[1] += e[1]
            else:
                adj[a][n] = e
                adj[n][a] = e
        adj[a].pop(b, None)
        adj[b] = {}
        ma = sums[a] / cnt[a]
        for n, e in adj[a].items():
            if e[0] >= min_contact and e[1] / e[0] < jt:
                dd = float(np.linalg.norm(ma - sums[n] / cnt[n]))
                if dd < thr:
                    x, y = (a, n) if a < n else (n, a)
                    heapq.heappush(heap, (dd, x, y, ver[x], ver[y]))
    mp = np.arange(K + 1, dtype=np.int64)
    for i in range(1, K + 1):
        r = i
        while parent[r] != r:
            r = parent[r]
        mp[i] = r
    return mp


def merge_regions(lab, feats, thr, min_contact, min_px, jump_thr, max_px=0, min_compact=0, compact_min_px=0):
    """Фаза 1: иерархическое слияние (см. _hac). Фаза 2: регионы < min_px → к самому похожему соседу."""
    lab = lab.astype(np.int32)
    K, u, v, c, js, cnt, sums, mom = _build_graph(lab, feats)
    total = np.arange(K + 1, dtype=np.int64)
    if u.size:
        mp = _hac(K, u, v, c, js, cnt, sums, mom, thr, min_contact, jump_thr, max_px, min_compact, compact_min_px)
        K, u, v, c, js, cnt, sums, mom, mp = _contract(K, u, v, c, js, cnt, sums, mom, mp)
        total = mp[total]
    for _ in range(30):
        if u.size == 0:
            break
        small = cnt < min_px
        small[0] = False
        m = small[u] | small[v]
        if not m.any():
            break
        uo, vo = u[m], v[m]
        means = sums / np.maximum(cnt[:, None], 1)
        d = np.linalg.norm(means[uo] - means[vo], axis=1)
        su, sv = small[uo], small[vo]
        src = np.concatenate([uo[su], vo[sv]]); dst = np.concatenate([vo[su], uo[sv]])
        w = np.concatenate([d[su], d[sv]])
        order = np.lexsort((w, src))
        src_s, dst_s = src[order], dst[order]
        first = np.r_[True, src_s[1:] != src_s[:-1]]
        nodes = src_s[first]
        best = np.full(K + 1, -1, np.int64)
        best[nodes] = dst_s[first]
        mp = np.arange(K + 1, dtype=np.int64)
        mp[nodes] = best[nodes]
        mutual = (best[best[nodes]] == nodes) & (nodes < best[nodes])
        mp[nodes[mutual]] = nodes[mutual]
        for _ in range(8):
            mp = mp[mp]
        K, u, v, c, js, cnt, sums, mom, mp = _contract(K, u, v, c, js, cnt, sums, mom, mp)
        total = mp[total]
    total[0] = 0
    return total.astype(np.int32)[lab]


def relax_labels(lab, f, iters, beta):
    """Пиксельное уточнение (ICM, шахматный порядок): граничный пиксель переходит к соседнему региону, если
    (расстояние до среднего региона)² + beta·(число несовпадающих соседей) меньше. Метка 0 (барьер/nodata) не трогается."""
    lab = lab.astype(np.int32).copy()
    h, w = lab.shape
    par = (np.arange(h)[:, None] + np.arange(w)[None, :]) & 1
    for _ in range(iters):
        K = int(lab.max())
        flat = lab.ravel()
        cnt = np.bincount(flat, minlength=K + 1).astype(np.float64)
        means = np.stack([np.bincount(flat, weights=f[..., k].ravel(), minlength=K + 1)
                          for k in range(f.shape[-1])], 1) / np.maximum(cnt[:, None], 1)
        changed = 0
        for p in (0, 1):
            P = np.pad(lab, 1)
            bmask = np.zeros((h, w), bool)
            for sh in (P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]):
                bmask |= (sh != lab) & (sh > 0)
            bmask &= (lab > 0) & (par == p)
            ys, xs = np.nonzero(bmask)
            if ys.size == 0:
                continue
            nbl = np.stack([P[ys, xs + 1], P[ys + 2, xs + 1], P[ys + 1, xs], P[ys + 1, xs + 2]], 1)
            own = lab[ys, xs]
            cand = np.concatenate([own[:, None], nbl], 1)
            data = ((f[ys, xs][:, None, :] - means[cand]) ** 2).sum(-1)
            smooth = (nbl[:, None, :] != cand[:, :, None]).sum(-1)
            cost = data + beta * smooth
            cost[cand == 0] = np.inf
            new = cand[np.arange(ys.size), cost.argmin(1)]
            ch = new != own
            lab[ys[ch], xs[ch]] = new[ch]
            changed += int(ch.sum())
        if changed == 0:
            break
    return lab


def segment_regions(feats, valid, crop=None, wcmap=None):
    """feats (h,w,3) float32 0–255; valid (h,w) bool. Возвращает (метки регионов, маска барьеров).
    SLIC без параметра mask (с ним в ~30 раз медленнее): невалидное заполняется нулём и обнуляется после."""
    barrier = detect_barriers(feats, valid, wcmap) if (BARRIER_THR or len(BARRIER_WC_CLASSES)) \
        else np.zeros(valid.shape, bool)
    line_b = np.zeros(valid.shape, bool)
    if LINE_SENS:
        line_b = detect_line_barriers(feats, valid, LINE_SENS, LINE_SIGMA_ALONG, LINE_MIN_PX, LINE_EDGE_SENS)
        if LINE_DIL > 0 and line_b.any():
            line_b = binary_dilation(line_b, structure=np.ones((3, 3), bool), iterations=int(LINE_DIL))
        line_b &= valid & ~barrier                  # там, где уже барьер-лесополоса, линия не нужна
    seg_valid = valid & ~barrier & ~line_b
    extra = None
    if USE_WC_IN_SEGMENTATION and crop is not None:
        extra = (crop.astype(np.float32) * 255 * WC_WEIGHT)[..., None]

    def prep(sig):
        x = smooth_feats(feats, sig)
        if extra is not None:
            x = np.concatenate([x, extra], -1)
        return np.where(valid[..., None], x, 0).astype(np.float32)

    f = prep(SMOOTH_SIGMA)
    lab = slic(f / 255.0, n_segments=max(int(valid.size / SP_SIZE_PX), 1), compactness=COMPACTNESS,
               convert2lab=False, channel_axis=-1, start_label=1, enforce_connectivity=True,
               min_size_factor=0.25, max_size_factor=4)
    lab = lab.astype(np.int32)
    lab[~seg_valid] = 0
    lab = cc_label(lab, connectivity=1, background=0).astype(np.int32)   # половинки суперпикселя за барьером — разные регионы
    max_px = int(MAX_REGION_HA / px_area_ha) if MAX_REGION_HA else 0
    cmin_px = int(COMPACT_MIN_HA / px_area_ha) if COMPACT_MIN_HA else 0
    lab = merge_regions(lab, f, MERGE_THR, MIN_CONTACT, MIN_REGION_PX, JUMP_THR, max_px, MIN_COMPACT, cmin_px)
    if RELAX_ITERS > 0:
        lab = relax_labels(lab, prep(RELAX_SIGMA), RELAX_ITERS, RELAX_BETA)
    if line_b.any() and LINE_FILL:                  # пиксели линий-границ достаются ближайшему региону: соседние поля соприкасаются по линии без зазора
        idx = distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(line_b & (lab == 0), lab[idx[0], idx[1]], lab)
    return _compact(lab), barrier

# %% 5_vector_funcs
STAT_COLS = ['n_px', 'crop_px', 's0', 's1', 's2', 'sx', 'sy', 'sxx', 'syy', 'sxy']


def extract_polygons(lab, feats, crop, halo_off, core, region, tile_id):
    """Берёт ядро тайла, считает статистику по связным регионам, оставляет «поля» (доля cropland >= порога)
    и векторизует в ПИКСЕЛЬНЫХ координатах растра (координаты целые -> швы совпадают точно).
    halo_off=(hc0, hr0); core=(c0, r0, c1, r1); region=(X0, Y0, X1, Y1) — границы обрабатываемой области."""
    hc0, hr0 = halo_off
    c0, r0, c1, r1 = core
    X0, Y0, X1, Y1 = region
    sl = (slice(r0 - hr0, r1 - hr0), slice(c0 - hc0, c1 - hc0))
    labc, fc, crc = lab[sl], feats[sl], crop[sl]
    cc = cc_label(labc, connectivity=1, background=0)
    flat = cc.ravel()
    n = np.bincount(flat).astype(np.float64)
    cr = np.bincount(flat, weights=crc.ravel().astype(np.float64), minlength=n.size)
    ok = (n > 0) & (cr >= CROP_FRAC_MIN * n)
    ok[0] = False
    if not ok.any():
        return None
    s = [np.bincount(flat, weights=fc[..., k].ravel().astype(np.float64), minlength=n.size) for k in range(3)]
    mom = _moments(flat, cc.shape, n.size - 1, x0=c0 - X0, y0=r0 - Y0)      # координаты относительно начала области (точность float64)
    geoms, vals = [], []
    for g, v in rio_shapes(cc.astype(np.int32), mask=ok[cc], transform=Affine(1, 0, c0, 0, 1, r0), connectivity=4):
        geoms.append(shp_shape(g)); vals.append(int(v))
    vals = np.array(vals)
    gdf = gpd.GeoDataFrame({'tile_id': tile_id, 'n_px': n[vals], 'crop_px': cr[vals],
                            's0': s[0][vals], 's1': s[1][vals], 's2': s[2][vals],
                            'sx': mom[vals, 0], 'sy': mom[vals, 1], 'sxx': mom[vals, 2], 'syy': mom[vals, 3], 'sxy': mom[vals, 4]},
                           geometry=geoms, crs=None)
    b = shapely.bounds(np.array(geoms, dtype=object))
    gdf['seam'] = ((b[:, 0] <= c0) & (c0 > X0)) | ((b[:, 2] >= c1) & (c1 < X1)) | \
                  ((b[:, 1] <= r0) & (r0 > Y0)) | ((b[:, 3] >= r1) & (r1 < Y1))
    return gdf


def merge_across_seams_legacy(gdf, thr, min_shared):
    """[v3] Склеивает фрагменты, разрезанные границей тайла: соседи из разных тайлов с общей границей >= min_shared px
    и близкими средними (< thr). Без проверки резкости границы и размера: транзитивное замыкание по швам сливает соседние
    поля одного тона в гигантский регион (на сцене TAT — 38 тыс. га). Оставлено для сравнения (SEAM_CONSTRAIN=False)."""
    gdf = gdf.reset_index(drop=True)
    idx = np.flatnonzero(gdf['seam'].values)
    if idx.size < 2:
        return gdf
    geoms = gdf.geometry.values
    sg = geoms[idx]
    qi, ti = STRtree(sg).query(sg, predicate='intersects')
    m = qi < ti
    A, B = idx[qi[m]], idx[ti[m]]
    tid = gdf['tile_id'].values
    d_tile = tid[A] != tid[B]
    A, B = A[d_tile], B[d_tile]
    if A.size == 0:
        return gdf
    shared = shapely.length(shapely.intersection(geoms[A], geoms[B]))
    mean = gdf[['s0', 's1', 's2']].values / gdf['n_px'].values[:, None]
    dist = np.linalg.norm(mean[A] - mean[B], axis=1)
    ok = (shared >= min_shared) & (dist < thr)
    A, B = A[ok], B[ok]
    if A.size == 0:
        return gdf
    n = len(gdf)
    _, comp = connected_components(coo_matrix((np.ones(A.size), (A, B)), shape=(n, n)), directed=False)
    sizes = np.bincount(comp)
    multi = sizes[comp] > 1
    keep = gdf[~multi]
    sub = gdf[multi].copy()
    sub['comp'] = comp[multi]
    agg = sub.groupby('comp')[STAT_COLS].sum()
    ug = sub.groupby('comp')['geometry'].apply(lambda s: shapely.union_all(s.values))
    merged = gpd.GeoDataFrame(agg, geometry=ug.reindex(agg.index).values, crs=None)
    merged['tile_id'] = -1
    merged['seam'] = False
    out = pd.concat([keep, merged.reset_index(drop=True)], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry='geometry', crs=None)


def fill_small_holes(geoms, min_hole_px):
    """Заливает дыры меньше min_hole_px (пиксельные координаты)."""
    def fix(p):
        return Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= min_hole_px])
    out = []
    for g in geoms:
        if g.geom_type == 'Polygon' and len(g.interiors):
            g = fix(g)
        elif g.geom_type == 'MultiPolygon':
            g = MultiPolygon([fix(p) for p in g.geoms])
        out.append(g)
    return out


def finalize(gdf, T, crs, px_area_ha, min_field_ha, simplify_px, min_width_px=0, fill_holes_ha=0):
    """Заливка дыр, фильтры по площади и толщине, упрощение (в пикселях), перевод в координаты растра, атрибуты."""
    g = gdf[gdf['n_px'] * px_area_ha >= min_field_ha].copy().reset_index(drop=True)
    if fill_holes_ha > 0 and len(g):
        hole_px = min(fill_holes_ha, min_field_ha) / px_area_ha     # не больше MIN_FIELD_HA: внутри дыры не окажется другого поля
        g['geometry'] = gpd.GeoSeries(fill_small_holes(g.geometry.values, hole_px), index=g.index)
    if min_width_px > 0 and len(g):
        gv = g.geometry.values
        thick = 2 * shapely.area(gv) / np.maximum(shapely.length(gv), 1e-9)
        g = g[thick >= min_width_px].reset_index(drop=True)
    geoms = g.geometry.values
    if simplify_px > 0 and len(g):
        try:
            geoms = shapely.coverage_simplify(geoms, simplify_px)
        except Exception:
            geoms = shapely.simplify(geoms, simplify_px, preserve_topology=True)
        geoms = shapely.make_valid(geoms)
    M = [T.a, T.b, T.d, T.e, T.c, T.f]
    geoms = [affine_transform(x, M) for x in geoms]
    out = gpd.GeoDataFrame({
        'id': np.arange(1, len(g) + 1),
        'n_px': g['n_px'].astype(int).values,
        'mean_max': (g['s0'] / g['n_px']).round(1).values,
        'mean_min': (g['s1'] / g['n_px']).round(1).values,
        'mean_diff': (g['s2'] / g['n_px']).round(1).values,
        'crop_frac': (g['crop_px'] / g['n_px']).round(3).values,
    }, geometry=geoms, crs=crs)
    if crs is not None and crs.is_projected:
        out['area_ha'] = (out.geometry.area / 1e4).round(3)
    else:
        out['area_ha'] = (out.to_crs('EPSG:6933').geometry.area / 1e4).round(3)
    return out


def merge_across_seams(gdf, thr, min_shared, jump_fn=None, jump_thr=0, max_px=0, min_compact=0, compact_min_px=0):
    """Склейка фрагментов через швы тайлов с теми же правилами, что и слияние внутри тайла:
    близость средних (по АГРЕГИРОВАННЫМ средним — без дрейфа цепочки), резкость границы на шве (jump_fn(geomsA, geomsB) →
    средний скачок значений между пикселями по обе стороны шва; ≥ jump_thr — не сливать), размер ≤ max_px и компактность
    ≥ min_compact. Пары обрабатываются жадно по возрастанию расстояния (union-find)."""
    gdf = gdf.reset_index(drop=True)
    idx = np.flatnonzero(gdf['seam'].values)
    if idx.size < 2:
        return gdf
    geoms = gdf.geometry.values
    sg = geoms[idx]
    qi, ti = STRtree(sg).query(sg, predicate='intersects')
    m = qi < ti
    A, B = idx[qi[m]], idx[ti[m]]
    tid = gdf['tile_id'].values
    d_tile = tid[A] != tid[B]
    A, B = A[d_tile], B[d_tile]
    if A.size == 0:
        return gdf
    shared = shapely.length(shapely.intersection(geoms[A], geoms[B]))
    n_px = gdf['n_px'].values.astype(np.float64)
    S = gdf[['s0', 's1', 's2']].values.astype(np.float64)
    mean = S / n_px[:, None]
    dist = np.linalg.norm(mean[A] - mean[B], axis=1)
    ok = (shared >= min_shared) & (dist < thr)
    A, B, dist = A[ok], B[ok], dist[ok]
    if A.size == 0:
        return gdf
    if jump_fn is not None and jump_thr:
        jmp = jump_fn(geoms[A], geoms[B])
        keep = ~(jmp >= jump_thr)                          # nan (не удалось посчитать) — не блокирует
        A, B, dist = A[keep], B[keep], dist[keep]
        if A.size == 0:
            return gdf
    n = len(gdf)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    agg_n, agg_s = n_px.copy(), S.copy()
    agg_m = gdf[['sx', 'sy', 'sxx', 'syy', 'sxy']].values.astype(np.float64).copy()
    for k in np.argsort(dist, kind='stable'):
        ra, rb = find(A[k]), find(B[k])
        if ra == rb:
            continue
        nn = agg_n[ra] + agg_n[rb]
        if max_px and nn > max_px:
            continue
        if np.linalg.norm(agg_s[ra] / agg_n[ra] - agg_s[rb] / agg_n[rb]) >= thr:
            continue
        if min_compact and nn >= compact_min_px and _compact_ratio(nn, agg_m[ra] + agg_m[rb]) < min_compact:
            continue
        parent[rb] = ra
        agg_n[ra] = nn; agg_s[ra] += agg_s[rb]; agg_m[ra] += agg_m[rb]
    comp = np.array([find(i) for i in range(n)])
    sizes = np.bincount(comp, minlength=n)
    multi = sizes[comp] > 1
    if not multi.any():
        return gdf
    keep_df = gdf[~multi]
    sub = gdf[multi].copy()
    sub['comp'] = comp[multi]
    agg = sub.groupby('comp')[STAT_COLS].sum()
    ug = sub.groupby('comp')['geometry'].apply(lambda s_: shapely.union_all(s_.values))
    merged = gpd.GeoDataFrame(agg, geometry=ug.reindex(agg.index).values, crs=None)
    merged['tile_id'] = -1
    merged['seam'] = False
    out = pd.concat([keep_df, merged.reset_index(drop=True)], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry='geometry', crs=None)


# %% 6_drive_ee
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('не Colab: Диск не монтируется, пути берутся как есть')

if USE_WORLDCOVER:
    import ee
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    print('GEE OK')
else:
    print('USE_WORLDCOVER=False: без GEE, вся площадь считается пашней')

# %% 7_open_raster
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOCAL_DIR, exist_ok=True)
tif_drive = f'{DRIVE_DIR}/{COMPOSITE_NAME}'
tif = f'{LOCAL_DIR}/{COMPOSITE_NAME}'
if COPY_LOCAL:
    if not os.path.exists(tif) or os.path.getsize(tif) != os.path.getsize(tif_drive):
        t0 = time.time(); shutil.copy(tif_drive, tif)
        print(f'Скопировано локально за {time.time()-t0:.0f} с')
else:
    tif = tif_drive

src = rasterio.open(tif)
H, W, T, CRS = src.height, src.width, src.transform, src.crs
assert src.count >= 3, 'Нужно 3 канала: max, min, diff'
HAS_ALPHA = src.count >= 4
print(f'{W}x{H} px, {src.count} каналов, {src.dtypes[0]}, CRS={CRS}, pixel={T.a:.6g} x {T.e:.6g}, nodata={src.nodata}')

if CRS.is_projected:
    px_area_ha = abs(T.a * T.e) / 1e4
else:
    lat = (src.bounds.top + src.bounds.bottom) / 2
    px_area_ha = abs(T.a) * 111320 * math.cos(math.radians(lat)) * abs(T.e) * 110574 / 1e4
print(f'площадь пикселя ≈ {px_area_ha*1e4:.1f} м²')

epsg = CRS.to_epsg()
CRS_CODE = f'EPSG:{epsg}' if epsg else CRS.to_wkt()

boundary_geom = None
if BOUNDARY_NAME:
    b = gpd.read_file(f'{DRIVE_DIR}/{BOUNDARY_NAME}').to_crs(CRS)
    boundary_geom = shapely.union_all(b.geometry.values)

if RUN_MODE == 'test':
    cx, cy = int(W * TEST_CENTER[0]), int(H * TEST_CENTER[1])
    X0, Y0 = max(cx - TEST_SIZE // 2, 0), max(cy - TEST_SIZE // 2, 0)
    X1, Y1 = min(X0 + TEST_SIZE, W), min(Y0 + TEST_SIZE, H)
else:
    X0, Y0, X1, Y1 = 0, 0, W, H
REGION = (X0, Y0, X1, Y1)

WC_DIR = Path(OUT_DIR) / 'wc_cache'
WC_DIR.mkdir(parents=True, exist_ok=True)

tiles = []
for r0 in range(Y0, Y1, TILE):
    for c0 in range(X0, X1, TILE):
        c1, r1 = min(c0 + TILE, X1), min(r0 + TILE, Y1)
        halo = (max(c0 - HALO, 0), max(r0 - HALO, 0), min(c1 + HALO, W), min(r1 + HALO, H))
        if boundary_geom is not None:
            xa, ya = T * (c0, r0); xb, yb = T * (c1, r1)
            if not boundary_geom.intersects(shp_box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))):
                continue
        tiles.append(dict(core=(c0, r0, c1, r1), halo=halo))
print(f'режим: {RUN_MODE}, область {X1-X0}x{Y1-Y0} px, тайлов: {len(tiles)}')

# %% 8_stage_framework
# ───────────── Общий каркас: параметры, слои, кэш, сохранение ─────────────
def set_params(*overrides):
    """BASE → затем перекрытия (dict-ы) по порядку. Все ключи становятся глобальными переменными сегментации."""
    g = globals()
    g.update(BASE)
    for o in overrides:
        g.update({k: v for k, v in o.items() if k in BASE})     # служебные ключи пресета (SPLIT, NO_STAGES) в параметры сегментации не попадают
    g['SEAM_THR'] = g['MERGE_THR']


def read_any(path):
    p = str(path)
    gdf = gpd.read_parquet(p) if p.endswith('.parquet') else gpd.read_file(p)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS)
    gdf = gdf.to_crs(CRS)
    gdf['geometry'] = shapely.make_valid(gdf.geometry.values)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == 'Polygon'].reset_index(drop=True)
    if 'kind' not in gdf.columns:
        gdf['kind'] = 'input'
    return gdf


TAG = ''            # суффикс файлов (в режиме сравнения — '_<пресет>')
SAVE_STAGES = None  # какие этапы сохранять в GPKG: None — все; множество — только перечисленные (режим сравнения: {1})


def stage_file(k):
    return Path(OUT_DIR) / f'fields_after_stage{k}_{RUN_MODE}{TAG}.gpkg'


def save_stage(k, gdf):
    if SAVE_STAGES is not None and k not in SAVE_STAGES:
        print(f'  · этап {k}: {len(gdf)} полей, {gdf.area_ha.sum():,.0f} га')
        return
    gdf.to_file(stage_file(k), driver='GPKG', layer='fields')
    print(f'  → {stage_file(k).name}: {len(gdf)} полей, {gdf.area_ha.sum():,.0f} га')


FIELDS = None      # текущий слой полей, передаётся от этапа к этапу
CLASS_SPLIT = None # слой разбивки поля по классам (результат этапа 6)
SPOTS_LAYER = None # слой пятен (результат этапа 4)
SPLIT_OVER = {}    # перекрытия параметров этапа 1b из пресета
NO_STAGES_CUR = set()   # этапы, отключённые в текущем пресете
STAGE_FUNCS = {}   # k → функция этапа (заполняется ячейками этапов)
COMPARE = bool(COMPARE_PRESETS and RUN_MODE == 'test')


def stage_on(k):
    return bool(STAGES[k]) and k not in NO_STAGES_CUR


def describe(gdf):
    """Краткая статистика слоя: число, медиана/p90/максимум площади, гиганты (> GIANT_HA) и их доля площади."""
    a = gdf.area_ha.values.astype(float)
    big = a > GIANT_HA
    return dict(n=len(gdf), median_ha=round(float(np.median(a)), 1), p90_ha=round(float(np.quantile(a, .9)), 1), max_ha=int(a.max()),
                n_giant=int(big.sum()), giant_share=round(float(a[big].sum() / a.sum()), 3))


ORDER = [1, '1b', 2, 3, 4, 5, 6]     # порядок этапов (для RESUME)


def get_input(k):
    """Вход этапа k: свой слой (IN_LAYERS[k]) → результат предыдущего этапа (FIELDS) → файл предыдущего этапа с диска."""
    if IN_LAYERS.get(k):
        gdf = read_any(IN_LAYERS[k])
        print(f'этап {k}: вход — ваш слой {IN_LAYERS[k]} ({len(gdf)} объектов)')
        return add_area(gdf)
    if FIELDS is not None:
        return FIELDS
    if RESUME:
        for j in reversed(ORDER[:ORDER.index(k)]):
            if stage_file(j).exists():
                gdf = read_any(stage_file(j))
                print(f'этап {k}: вход — {stage_file(j).name} ({len(gdf)} полей)')
                return add_area(gdf)
    raise RuntimeError(f'Этап {k}: нет входных полей. Включите этап 1 (и промежуточные) или задайте IN_LAYERS[{k!r}].')


def regularize_boundaries(gdf, tol_m=None):
    """Выпрямление границ с сохранением общей топологии: границы всех полигонов разбиваются на дуги между узлами
    (там, где сходятся ≥3 границ), каждая дуга упрощается Дугласом–Пекером с допуском tol_m (метры, проекция UTM),
    полигоны собираются заново (polygonize) и получают атрибуты исходных. Общая граница двух полей остаётся общей.
    Площадь и IoU почти не меняются; эталонная цифровка ≈ 10 вершин/км и ~2/3 периметра в прямых отрезках ≥ 150 м."""
    tol_m = REGULARIZE_TOL_M if tol_m is None else tol_m
    if gdf is None or len(gdf) == 0 or not tol_m:
        return gdf
    from shapely.ops import polygonize
    c = gdf.to_crs(4326).total_bounds
    utm = f'EPSG:{32600 + int(((c[0] + c[2]) / 2 + 180) // 6) + 1}' if not CRS.is_projected else CRS
    g6 = gdf.reset_index(drop=True).to_crs(utm)
    geoms = shapely.make_valid(g6.geometry.values)
    geoms = np.array([x if x.geom_type in ('Polygon', 'MultiPolygon') else
                      shapely.union_all([q for q in shapely.get_parts(x) if q.geom_type in ('Polygon', 'MultiPolygon')]) for x in geoms], dtype=object)
    try:
        lines = shapely.line_merge(shapely.union_all(shapely.boundary(geoms)))
        simp = list(shapely.get_parts(lines.simplify(tol_m, preserve_topology=True)))     # все дуги вместе: пересечений между ними не возникает
        faces = list(polygonize(simp))
        tree = shapely.STRtree(geoms)
        rep = shapely.point_on_surface(np.array(faces, dtype=object))
        owner = {}
        for f, pt in zip(faces, rep):
            hit = [k for k in tree.query(pt, predicate='within')]
            if hit:
                owner.setdefault(int(hit[0]), []).append(f)
        out_geoms = [shapely.union_all(owner[k]) if k in owner else geoms[k] for k in range(len(geoms))]
    except Exception as e:
        print('  регуляризация: topo-упрощение не удалось, упрощение по полигонам:', str(e)[:80])
        out_geoms = [x.simplify(tol_m, preserve_topology=True) for x in geoms]
    g6 = g6.copy(); g6['geometry'] = shapely.make_valid(np.array(out_geoms, dtype=object))
    g6 = g6[~g6.geometry.is_empty].to_crs(gdf.crs)
    g6['geometry'] = shapely.make_valid(g6.geometry.values)
    g6 = g6.explode(index_parts=False).reset_index(drop=True)
    g6 = g6[g6.geometry.geom_type == 'Polygon'].reset_index(drop=True)
    return add_area(g6)


def add_area(gdf):
    gdf = gdf.copy()
    gdf['area_ha'] = (gdf.to_crs('EPSG:6933').geometry.area / 1e4).round(3) if not CRS.is_projected else (gdf.geometry.area / 1e4).round(3)
    return gdf


def layer_hash(gdf):
    if gdf is None or len(gdf) == 0:
        return 'none'
    s = f"{len(gdf)}|{float(shapely.area(gdf.geometry.values).sum()):.6g}|{np.round(gdf.total_bounds, 6).tolist()}"
    return hashlib.md5(s.encode()).hexdigest()[:8]


def subtract_existing(new, existing, min_ha):
    """Вычитает из новых полигонов уже существующие поля (нет наложений), убирает обломки < min_ha."""
    if len(new) == 0 or existing is None or len(existing) == 0:
        return new
    tree = STRtree(existing.geometry.values)
    geoms = []
    for g_ in new.geometry.values:
        idx = tree.query(g_, predicate='intersects')
        if idx.size:
            g_ = g_.difference(shapely.union_all(existing.geometry.values[idx]))
        geoms.append(g_)
    out = new.copy(); out['geometry'] = geoms
    out = out[~out.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.geom_type == 'Polygon']
    out = add_area(out)
    return out[out.area_ha >= min_ha].reset_index(drop=True)


def resolve_nested(gdf, policy=None):
    """Поле не может входить в другое. Вложенным считается поле, ≥ NESTED_FRAC площади которого лежит внутри внешнего
    контура (с заливкой дыр) более крупного поля; если оно лежит в дыре, то должно занимать ≥ NESTED_HOLE_FRAC этой дыры.
    absorb — поглощается внешним (дыра заливается), drop — удаляется, keep — без изменений."""
    policy = policy or NESTED_POLICY
    if policy == 'keep' or len(gdf) < 2:
        return gdf
    gdf = gdf.reset_index(drop=True).copy()
    geoms = list(gdf.geometry.values)
    order = np.argsort(shapely.area(np.array(geoms, dtype=object)))          # от мелких к крупным
    alive = np.ones(len(gdf), bool)
    filled = [shapely.Polygon(g_.exterior) if g_.geom_type == 'Polygon' else g_ for g_ in geoms]
    tree = STRtree(np.array(filled, dtype=object))
    n_nested = 0
    for j in order:
        gj = geoms[j]
        aj = gj.area
        if aj <= 0:
            continue
        for i in tree.query(gj, predicate='intersects'):
            if i == j or not alive[i] or filled[i].area <= aj:
                continue
            if filled[i].intersection(gj).area < NESTED_FRAC * aj:
                continue
            gi = geoms[i]
            hole_hit = [shapely.Polygon(r) for r in gi.interiors if shapely.Polygon(r).intersection(gj).area >= NESTED_HOLE_FRAC * shapely.Polygon(r).area]
            if gi.intersection(gj).area >= 0.5 * aj or hole_hit:
                if policy == 'absorb':
                    geoms[i] = shapely.union_all([gi, gj] + hole_hit)
                    filled[i] = shapely.Polygon(geoms[i].exterior) if geoms[i].geom_type == 'Polygon' else geoms[i]
                alive[j] = False
                n_nested += 1
                break
    gdf['geometry'] = geoms
    out = add_area(gdf[alive].reset_index(drop=True))
    out = out[out.geometry.geom_type == 'Polygon'] if (out.geometry.geom_type != 'Polygon').any() else out
    print(f'вложенные поля ({policy}): {n_nested}')
    return out.reset_index(drop=True)


def shape_filter(gdf, max_ha=None, max_hole_frac=None, min_solidity=0.0):
    """Отсев «не полей»: слишком крупных, с большой долей дыр (фон между полями с вырезанными полями) и «рваных» по форме
    (выпуклость считается по контуру с заливкой дыр)."""
    if len(gdf) == 0:
        return gdf
    gv = gdf.geometry.values
    filled = np.array([shapely.Polygon(g_.exterior) if g_.geom_type == 'Polygon' else g_ for g_ in gv], dtype=object)
    fa = shapely.area(filled)
    ok = np.ones(len(gdf), bool)
    if max_ha:
        ok &= gdf['area_ha'].values <= max_ha
    if max_hole_frac is not None:
        ok &= 1 - shapely.area(gv) / np.maximum(fa, 1e-12) <= max_hole_frac
    if min_solidity > 0:
        ok &= fa / np.maximum(shapely.area(shapely.convex_hull(filled)), 1e-12) >= min_solidity
    if (~ok).any():
        print(f'  отсеяно по форме/размеру: {int((~ok).sum())} из {len(gdf)}')
    return gdf[ok].reset_index(drop=True)


def solidity_filter(gdf, min_sol):
    if min_sol <= 0 or len(gdf) == 0:
        return gdf
    gv = gdf.geometry.values
    sol = shapely.area(gv) / np.maximum(shapely.area(shapely.convex_hull(gv)), 1e-12)
    return gdf[sol >= min_sol].reset_index(drop=True)



def field_window(geom, margin=3):
    """Окно растра вокруг полигона (границы в координатах растра) + его transform; None, если вне растра."""
    inv = ~T
    xs, ys = [], []
    minx, miny, maxx, maxy = geom.bounds
    for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
        c, r = inv * (x, y); xs.append(c); ys.append(r)
    c0, c1 = max(int(math.floor(min(xs))) - margin, 0), min(int(math.ceil(max(xs))) + margin, W)
    r0, r1 = max(int(math.floor(min(ys))) - margin, 0), min(int(math.ceil(max(ys))) + margin, H)
    if c1 <= c0 or r1 <= r0:
        return None, None
    win = Window(c0, r0, c1 - c0, r1 - r0)
    return win, rasterio.windows.transform(win, T)


print('каркас этапов готов')

# %% 9_segmentation_stage
import matplotlib.pyplot as plt


def get_wc_map(t, h, w, tag):
    """Классы WorldCover (uint8) ровно на сетке окна: ee.data.computePixels с явной affine-сеткой. Кэш на Диске."""
    cache = WC_DIR / f'wcmap_{tag}_{h}x{w}.npz'
    if cache.exists():
        return np.load(cache)['m']
    req = {'expression': _wc_img, 'fileFormat': 'GEO_TIFF',
           'grid': {'dimensions': {'width': w, 'height': h},
                    'affineTransform': {'scaleX': t.a, 'shearX': t.b, 'translateX': t.c,
                                        'shearY': t.d, 'scaleY': t.e, 'translateY': t.f},
                    'crsCode': CRS_CODE}}
    for attempt in range(5):
        try:
            raw = ee.data.computePixels(req)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f'  GEE retry {attempt+1}: {str(e)[:120]}'); time.sleep(5 * (attempt + 1))
    if raw[:2] == b'PK':
        zf = zipfile.ZipFile(io.BytesIO(raw)); raw = zf.read(zf.namelist()[0])
    with rasterio.MemoryFile(raw) as mf, mf.open() as ds:
        m = ds.read(1)
    assert m.shape == (h, w), f'WorldCover {m.shape} != {(h, w)}'
    np.savez_compressed(cache, m=m.astype(np.uint8))
    return m.astype(np.uint8)


if USE_WORLDCOVER:
    _wc_img = (ee.ImageCollection(WC_ASSET).mosaic().select('Map').unmask(0).rename('wc').toUint8())


_SRC = {}


def _src():
    """Растр открывается отдельно в каждом процессе (GDAL-дескрипторы нельзя делить между процессами)."""
    pid = os.getpid()
    if pid not in _SRC:
        _SRC[pid] = rasterio.open(tif)
    return _SRC[pid]


def read_window(win):
    s = _src()
    arr = s.read([1, 2, 3], window=win).astype(np.float32)
    valid = np.isfinite(arr).all(0) & (arr > 0).any(0)
    if HAS_ALPHA:
        valid &= s.read(4, window=win) > 0
    return np.nan_to_num(np.moveaxis(arr, 0, -1)), valid


def tile_info(tile):
    c0, r0, c1, r1 = tile['core']
    hc0, hr0, hc1, hr1 = tile['halo']
    h, w = hr1 - hr0, hc1 - hc0
    win = Window(hc0, hr0, w, h)
    return win, rasterio.windows.transform(win, T), h, w, f'{r0:06d}_{c0:06d}'


def prefetch_wc(todo):
    """WorldCover для всех тайлов заранее, в потоках (GEE — сетевая задержка), чтобы вычислительные процессы не ходили в GEE."""
    if not USE_WORLDCOVER or not todo:
        return
    def one(tile):
        win, t_halo, h, w, tag = tile_info(tile)
        get_wc_map(t_halo, h, w, tag)
    with ThreadPoolExecutor(4) as ex:
        list(tqdm(ex.map(one, todo), total=len(todo), desc='WorldCover'))


def seg_tile(tile, tdir, exclude=None, ex_tree=None):
    """Сегментация одного тайла. exclude — уже существующие поля (GeoDataFrame в CRS растра): их пиксели не сегментируются
    (так работают этапы 2 и 3 — только по «остатку»)."""
    c0, r0, c1, r1 = tile['core']
    hc0, hr0, hc1, hr1 = tile['halo']
    tag = f'{r0:06d}_{c0:06d}'
    out, empty = tdir / f'tile_{tag}.parquet', tdir / f'tile_{tag}.empty'
    if out.exists() or empty.exists():
        return 'cached'
    h, w = hr1 - hr0, hc1 - hc0
    win = Window(hc0, hr0, w, h)
    t_halo = rasterio.windows.transform(win, T)
    feats, valid = read_window(win)
    if boundary_geom is not None:
        xa, ya = t_halo * (0, 0); xb, yb = t_halo * (w, h)
        part = shapely.intersection(boundary_geom, shp_box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
        if part.is_empty:
            empty.touch(); return 'empty'
        valid &= rasterize([(part, 1)], out_shape=(h, w), transform=t_halo, fill=0, dtype='uint8').astype(bool)
    if exclude is not None and len(exclude):
        xa, ya = t_halo * (0, 0); xb, yb = t_halo * (w, h)
        idx = ex_tree.query(shp_box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)), predicate='intersects')
        if idx.size:
            ex = rasterize([(g_, 1) for g_ in exclude.geometry.values[idx]], out_shape=(h, w), transform=t_halo,
                           fill=0, dtype='uint8').astype(bool)
            valid &= ~binary_dilation(ex, iterations=1)

    if USE_WORLDCOVER:
        wc = get_wc_map(t_halo, h, w, tag)
        crop = np.isin(wc, AGRI_CLASSES)
    else:
        wc, crop = None, np.ones((h, w), bool)
    sl = (slice(r0 - hr0, r1 - hr0), slice(c0 - hc0, c1 - hc0))
    nv = valid[sl].sum()
    if nv == 0 or (crop & valid)[sl].sum() / nv < MIN_TILE_CROP:
        empty.touch(); return 'no cropland'
    lab, _ = segment_regions(feats, valid, crop, wc)
    gdf = extract_polygons(lab, feats, crop, (hc0, hr0), tile['core'], REGION, tile_id=int(hashlib.md5(tag.encode()).hexdigest()[:7], 16))
    if gdf is None or gdf.empty:
        empty.touch(); return 'no fields'
    gdf.to_parquet(out)
    return f'{len(gdf)} polygons'


def seam_jump_fn(gA, gB):
    """Средний скачок значений (как JUMP_THR внутри тайла) на общей границе фрагментов через шов тайла.
    gA, gB — массивы полигонов в пиксельных координатах растра; шов — вертикальная/горизонтальная прямая с целой координатой.
    Возвращает массив (nan — посчитать не удалось, такая пара не блокируется)."""
    out = np.full(len(gA), np.nan)
    inter = shapely.intersection(gA, gB)
    pad = 3
    for i, g_ in enumerate(inter):
        tot, cnt = 0.0, 0
        for part in shapely.get_parts(g_):
            if part.geom_type != 'LineString':
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in part.bounds]
            vert = x0 == x1 and y1 > y0
            if not (vert or (y0 == y1 and x1 > x0)):
                continue
            # окно с запасом pad для сглаживания; sl — индексы вдоль шва, k — координата шва
            ca, cb, ra, rb = (x0 - pad - 1, x0 + pad, y0 - pad, y1 + pad) if vert else (x0 - pad, x1 + pad, y0 - pad - 1, y0 + pad)
            ca, ra, cb, rb = max(ca, 0), max(ra, 0), min(cb, W), min(rb, H)
            if cb <= ca or rb <= ra:
                continue
            f, _ = read_window(Window(ca, ra, cb - ca, rb - ra))
            f = smooth_feats(f, SMOOTH_SIGMA)
            if vert:
                if not (ca <= x0 - 1 and x0 < cb):
                    continue
                d = np.sqrt(((f[y0 - ra:y1 - ra, x0 - 1 - ca] - f[y0 - ra:y1 - ra, x0 - ca]) ** 2).sum(-1))
            else:
                if not (ra <= y0 - 1 and y0 < rb):
                    continue
                d = np.sqrt(((f[y0 - 1 - ra, x0 - ca:x1 - ca] - f[y0 - ra, x0 - ca:x1 - ca]) ** 2).sum(-1))
            tot += float(d.sum()); cnt += d.size
        if cnt:
            out[i] = tot / cnt
    return out


_JOB = {}


def _seg_job(tile):
    return seg_tile(tile, _JOB['tdir'], _JOB['exclude'], _JOB['ex_tree'])


def seg_raw(name, overrides, exclude=None):
    """Один проход по тайлам: сегментация → «сырые» регионы (все тона и размеры) → склейка швов.
    Результат кэшируется по тайлам; тайлы считаются параллельно (N_WORKERS процессов)."""
    set_params(*overrides)
    POST_ONLY = ('SIMPLIFY_PX', 'MIN_FIELD_HA', 'FILL_HOLES_HA', 'MIN_WIDTH_PX', 'SMOOTH_LABELS')    # влияют только на make_fields — сегментацию не пересчитывают
    p = {k: globals()[k] for k in BASE if k not in POST_ONLY}
    p.update(TILE=TILE, HALO=HALO, REGION=REGION, AGRI=AGRI_CLASSES, WC=USE_WORLDCOVER, EX=layer_hash(exclude), V=5)
    h = hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]
    tdir = Path(OUT_DIR) / f'tiles5_{RUN_MODE}_{name}_{h}'
    tdir.mkdir(parents=True, exist_ok=True)
    todo = [t_ for t_ in tiles if not ((tdir / f"tile_{t_['core'][1]:06d}_{t_['core'][0]:06d}.parquet").exists()
                                       or (tdir / f"tile_{t_['core'][1]:06d}_{t_['core'][0]:06d}.empty").exists())]
    if todo:
        prefetch_wc(todo)
        _JOB.update(tdir=tdir, exclude=exclude, ex_tree=STRtree(exclude.geometry.values) if exclude is not None and len(exclude) else None)
        if N_WORKERS > 1 and len(todo) > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(min(N_WORKERS, len(todo))) as pool:
                for _ in tqdm(pool.imap_unordered(_seg_job, todo), total=len(todo), desc=name):
                    pass
        else:
            for tile in tqdm(todo, desc=name):
                _seg_job(tile)
    files = sorted(tdir.glob('tile_*.parquet'))
    if not files:
        return None
    raw = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True), geometry='geometry', crs=None)
    if not SEAM_CONSTRAIN:
        return merge_across_seams_legacy(raw, SEAM_THR, MIN_CONTACT)
    max_px = int(max(SEAM_MAX_REGION_HA, MAX_REGION_HA) / px_area_ha) if MAX_REGION_HA else 0
    cmin_px = int(COMPACT_MIN_HA / px_area_ha) if COMPACT_MIN_HA else 0
    return merge_across_seams(raw, SEAM_THR, MIN_CONTACT, seam_jump_fn, JUMP_THR, max_px, MIN_COMPACT, cmin_px)


WC_REGION = None


def wc_region():
    """WorldCover на сетке растра для всей области (np.memmap uint8; собирается из кэша тайлов). None — без WorldCover."""
    global WC_REGION
    if not USE_WORLDCOVER:
        return None
    if WC_REGION is None:
        path = Path(LOCAL_DIR) / f'wc_region_{RUN_MODE}_{X0}_{Y0}_{X1}_{Y1}.u8'
        mm = np.memmap(path, dtype=np.uint8, mode='w+', shape=(Y1 - Y0, X1 - X0))
        for tile in tiles:
            win, t_halo, h, w, tag = tile_info(tile)
            wc = get_wc_map(t_halo, h, w, tag)
            c0, r0, c1, r1 = tile['core']; hc0, hr0 = tile['halo'][:2]
            mm[r0 - Y0:r1 - Y0, c0 - X0:c1 - X0] = wc[r0 - hr0:r1 - hr0, c0 - hc0:c1 - hc0]
        mm.flush()
        WC_REGION = mm
    return WC_REGION


def polygon_crop_frac(gdf):
    """Доля пашни (AGRI_CLASSES) WorldCover внутри каждого полигона (по пикселям растра)."""
    mm = wc_region()
    out = np.ones(len(gdf))
    if mm is None:
        return out
    for i, g_ in enumerate(gdf.geometry.values):
        win, _ = field_window(g_, margin=0)
        if win is None:
            out[i] = 0; continue
        ca, ra = max(int(win.col_off), X0), max(int(win.row_off), Y0)
        cb, rb = min(int(win.col_off + win.width), X1), min(int(win.row_off + win.height), Y1)
        if cb <= ca or rb <= ra:
            out[i] = 0; continue
        wt = rasterio.windows.transform(Window(ca, ra, cb - ca, rb - ra), T)
        m = rasterize([(g_, 1)], out_shape=(rb - ra, cb - ca), transform=wt, fill=0, dtype='uint8').astype(bool)
        if m.any():
            out[i] = float(np.isin(mm[ra - Y0:rb - Y0, ca - X0:cb - X0], AGRI_CLASSES)[m].mean())
    return out


def crop_filter(gdf, min_frac=None, label=''):
    """Оставляет полигоны, на которых пашни WorldCover ≥ min_frac (по умолчанию CROP_POLY_MIN); записывает crop_wc."""
    min_frac = CROP_POLY_MIN if min_frac is None else min_frac
    if gdf is None or len(gdf) == 0 or not min_frac or not USE_WORLDCOVER:
        return gdf
    fr = polygon_crop_frac(gdf)
    gdf = gdf.copy(); gdf['crop_wc'] = fr.round(3)
    keep = fr >= min_frac
    if (~keep).any():
        by = gdf[~keep]['kind'].value_counts().to_dict() if 'kind' in gdf.columns else {}
        print(f'  пашня WorldCover < {min_frac:.0%}: убрано {int((~keep).sum())} из {len(gdf)} {label}{by}')
    return gdf[keep].reset_index(drop=True)


def make_fields(merged, tone, min_ha, fill_ha, width_px, min_solidity=0.0, kind='main', crop_min=0.0, simplify_px=None):
    """Из «сырых» регионов — поля нужного типа. tone: 'pd' — розовые/тёмные (mean_min ≤ YELLOW_MIN_THR),
    'y' — «жёлтые» (mean_min > YELLOW_MIN_THR), 'any'. Это дешёвая постобработка: сегментация не повторяется."""
    if merged is None or len(merged) == 0:
        return None
    mm = (merged['s1'] / merged['n_px']).values
    sel = np.ones(len(merged), bool)
    if tone == 'pd':
        sel &= mm <= YELLOW_MIN_THR
    elif tone == 'y':
        sel &= mm > YELLOW_MIN_THR
    sel &= (merged['crop_px'] / merged['n_px']).values >= crop_min
    if not sel.any():
        return None
    simplify_px = SIMPLIFY_PX if simplify_px is None else simplify_px
    fl = finalize(merged[sel].copy(), T, CRS, px_area_ha, min_ha, simplify_px, width_px, fill_ha)
    fl = solidity_filter(fl.reset_index(drop=True), min_solidity)
    fl['kind'] = kind
    return fl


def run_seg_stage(name, overrides, exclude=None, tone='pd', min_solidity=0.0, kind='main'):
    """Отдельный проход (используется для сравнения пресетов и для тонкого прохода этапа 2)."""
    merged = seg_raw(name, overrides, exclude)
    return make_fields(merged, tone, MIN_FIELD_HA, FILL_HOLES_HA, MIN_WIDTH_PX, min_solidity, kind, 0.0, SIMPLIFY_PX)


def renumber(gdf):
    gdf = gdf.reset_index(drop=True).copy()
    gdf['id'] = np.arange(1, len(gdf) + 1)
    return gdf


def compare_window(size=1400, center=(0.5, 0.5)):
    cx, cy = int(X0 + (X1 - X0) * center[0]), int(Y0 + (Y1 - Y0) * center[1])
    c0, r0 = min(max(cx - size // 2, X0), max(X1 - size, X0)), min(max(cy - size // 2, Y0), max(Y1 - size, Y0))
    return Window(c0, r0, min(size, X1 - c0), min(size, Y1 - r0))


CLASS_COLORS = {'засеянное': 'yellow', 'пар': 'cyan', 'неиспользуемое': 'red'}


def plot_compare(results, win=None, cols=3, panel_px=800, path=None, color_by_class=False):
    """Одно и то же окно композита (R=max, G=min, B=diff) + контуры каждого набора (обрезаны по окну: гигантский полигон
    не растягивает оси). color_by_class: цвет контура по классу поля (засеянное — жёлтый, пар — голубой, неиспользуемое — красный)."""
    win = win or compare_window(COMPARE_WIN_PX)
    rgb = np.moveaxis(src.read([1, 2, 3], window=win, out_shape=(3, panel_px, panel_px)), 0, -1).astype(np.float32)
    rgb = np.clip(rgb / 255, 0, 1)
    wt = rasterio.windows.transform(win, T)
    xa, ya = wt * (0, 0); xb, yb = wt * (win.width, win.height)
    x0, x1, y0, y1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
    names = list(results)
    rows = math.ceil(len(names) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, n in zip(axs.ravel(), names):
        ax.imshow(rgb, extent=(xa, xb, yb, ya))
        gdf = results[n]
        gs = gdf.geometry.clip_by_rect(x0, y0, x1, y1)
        ok = (~gs.is_empty) & gs.geom_type.isin(['Polygon', 'MultiPolygon'])
        if color_by_class and 'cls_main_name' in gdf.columns:
            for cname, col in CLASS_COLORS.items():
                sel = ok & (gdf['cls_main_name'] == cname)
                if sel.any():
                    gs[sel].boundary.plot(ax=ax, color=col, linewidth=0.8)
        else:
            gs[ok].boundary.plot(ax=ax, color='yellow', linewidth=0.7)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f'{n}  (полей в окне: {int(ok.sum())}, всего: {len(gdf)})', fontsize=11)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=80)
    plt.show()


# %% 10_stage1_main_fields
SUMMARY = []
RAW1 = None          # «сырые» регионы единственного прохода сегментации — из них делаются этапы 1, 2 и 3


def get_raw1():
    """Единственный проход сегментации (пресет этапа 1). Повторный вызов — из памяти; после перезапуска — из кэша тайлов."""
    global RAW1
    if RAW1 is None:
        RAW1 = seg_raw(f's1_{STAGE1_PRESET}', [PRESETS[STAGE1_PRESET]])
        set_params(PRESETS[STAGE1_PRESET])
    return RAW1


def s1_param(k):
    return {**BASE, **PRESETS[STAGE1_PRESET]}[k]


def do_stage1():
    global FIELDS
    r = make_fields(get_raw1(), 'pd', s1_param('MIN_FIELD_HA'), s1_param('FILL_HOLES_HA'), s1_param('MIN_WIDTH_PX'), 0.0, 'main',
                    simplify_px=s1_param('SIMPLIFY_PX'))
    if r is None or r.empty:
        raise RuntimeError('Этап 1: полей нет')
    FIELDS = renumber(crop_filter(r, label='(этап 1) '))
    save_stage(1, FIELDS)
    d = describe(FIELDS)
    print(f'  этап 1: максимум {d["max_ha"]} га, полигонов > {GIANT_HA} га: {d["n_giant"]} ({d["giant_share"]:.0%} площади)')
    return FIELDS


STAGE_FUNCS[1] = do_stage1
if not COMPARE:
    do_stage1() if STAGES[1] else print('Этап 1 пропущен (STAGES[1]=0)')

# %% 10b_stage1b_split
# ───────────── Этап 1б: разбивка слипшихся полигонов ─────────────
# Крупные полигоны, в которые слиплось несколько полей, режутся по длинным тонким линиям (лесополосы, дороги, межи)
# и по резким ступенькам тона. Линии ищутся ориентированными фильтрами (усреднение ВДОЛЬ линии, поэтому ловятся
# слабые линии, которые top-hat барьеры этапа 1 пропускают), затем отсеиваются по длине. Работает только внутри
# выбранных полигонов, растр читается окнами — это дёшево и не требует повторной сегментации.
from scipy.ndimage import binary_erosion, binary_dilation, label as ndi_label3, distance_transform_edt as _edt


def split_labels(feats, fm, P):
    """Метки кусков внутри маски поля fm (0 — вне поля) или None, если разбивать нечем."""
    R = split_line_response(feats, P['SENS'], P['SIGMA_ALONG'], edge_sens=P.get('EDGE_SENS'))
    core = binary_erosion(fm, iterations=P['EDGE_PX'])
    cand = apply_hysteresis_threshold(np.where(core, R, 0), 0.6, 1.0)
    lab, n = ndi_label3(skeletonize(cand), structure=np.ones((3, 3)))
    if n == 0:
        return None
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sz >= P['LINE_MIN_PX']); keep = keep[keep > 0]
    if not len(keep):
        return None
    lines = binary_dilation(np.isin(lab, keep), iterations=P['LINE_DIL'])
    pieces, m = ndi_label3(fm & ~lines, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if m < 2:
        return None
    ps = np.bincount(pieces.ravel(), minlength=m + 1)
    big = np.flatnonzero(ps * px_area_ha >= P['PIECE_MIN_HA']); big = big[big > 0]
    if len(big) < 2:
        return None
    seeds = np.where(np.isin(pieces, big), pieces, 0)               # мелкие куски и сами линии отдаются ближайшему крупному
    idx = _edt(seeds == 0, return_distances=False, return_indices=True)
    return np.where(fm, seeds[idx[0], idx[1]], 0)


_SPJOB = {}


def _split_job(i):
    """Один полигон → список кусков (shapely) или None (не разрезан / пропущен), 'skip' если окно слишком велико."""
    P, gm = _SPJOB['P'], _SPJOB['geoms'][i]
    win, wt = field_window(gm, margin=4)
    if win is None:
        return None
    if win.width * win.height > P['MAX_WINDOW_PX']:
        return 'skip'
    feats, valid = read_window(win)
    h, w = int(win.height), int(win.width)
    fm = rasterize([(gm, 1)], out_shape=(h, w), transform=wt, fill=0, dtype='uint8').astype(bool) & valid
    lab = split_labels(feats, fm, P) if fm.any() else None
    if lab is None:
        return None
    polys = [shp_shape(g_) for g_, v in _shapes_px(lab)]
    try:
        polys = list(shapely.coverage_simplify(np.array(polys, dtype=object), s1_param('SIMPLIFY_PX')))
    except Exception:
        polys = [p_.simplify(s1_param('SIMPLIFY_PX'), preserve_topology=True) for p_ in polys]
    Mx = [wt.a, wt.b, wt.d, wt.e, wt.c, wt.f]
    out = []
    for p_ in polys:
        p_ = shapely.make_valid(affine_transform(p_, Mx)).intersection(gm)        # внешний контур и дыры — как у исходного поля
        out += [q_ for q_ in shapely.get_parts(p_) if q_.geom_type == 'Polygon' and not q_.is_empty]
    return out


def split_fields(fin, P=None, verbose=True):
    """Режет крупные полигоны слипшихся полей. Возвращает (слой, число разрезанных полигонов). Полигоны считаются параллельно (N_WORKERS)."""
    P = {**SPLIT, **(P or {})}
    fin = fin.reset_index(drop=True)
    kinds = P['KINDS']
    elig = [i for i in range(len(fin)) if (kinds is None or fin['kind'].values[i] in kinds) and fin['area_ha'].values[i] >= P['MIN_HA']]
    _SPJOB.update(P=P, geoms=list(fin.geometry.values))
    res = {}
    if N_WORKERS > 1 and len(elig) > 3:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(N_WORKERS, len(elig))) as pool:
            for i, r in zip(elig, tqdm(pool.imap(_split_job, elig, chunksize=1), total=len(elig), desc='разбивка', disable=not verbose)):
                res[i] = r
    else:
        for i in tqdm(elig, desc='разбивка', disable=not verbose):
            res[i] = _split_job(i)
    rows, n_split, n_skip = [], 0, 0
    cols = fin.columns.drop('geometry')
    for i in range(len(fin)):
        rec = {c: fin[c].values[i] for c in cols}
        r = res.get(i)
        if r == 'skip':
            n_skip += 1
        if not isinstance(r, list) or not r:
            rows.append((fin.geometry.values[i], rec)); continue
        n_split += 1
        rows += [(q_, dict(rec, split_from=rec.get('id', i))) for q_ in r]
    out = gpd.GeoDataFrame([r for _, r in rows], geometry=[g_ for g_, _ in rows], crs=CRS)
    out = add_area(out)
    out = out[out.area_ha >= 0.05].reset_index(drop=True)
    if verbose:
        print(f'разбивка: разрезано {n_split} полигонов → {len(out)} полей (было {len(fin)})' + (f'; пропущено из-за размера окна: {n_skip}' if n_skip else ''))
    return out, n_split


def _shapes_px(lab):
    """Векторизация меток в пиксельных координатах окна (для coverage_simplify и общего transform)."""
    from rasterio.features import shapes as _sh
    for g_, v in _sh(lab.astype(np.int16), mask=lab > 0, transform=Affine.identity(), connectivity=4):
        yield g_, v


def do_stage1b():
    global FIELDS
    FIELDS = renumber(crop_filter(split_fields(get_input('1b'), SPLIT_OVER)[0], label='(куски 1b) '))      # SPLIT_OVER — перекрытия SPLIT из пресета
    save_stage('1b', FIELDS)
    return FIELDS


STAGE_FUNCS['1b'] = do_stage1b
if not COMPARE:
    do_stage1b() if STAGES['1b'] else print('Этап 1б (разбивка) пропущен (STAGES["1b"]=0)')

# %% 11_stage2_small_fields
# Мелкие поля берутся из тех же регионов единственного прохода (get_raw1), без повторной сегментации:
# фильтры — тон, площадь ≥ S2['MIN_FIELD_HA'], форма (толщина, выпуклость), доля пашни. Если задан свой слой (IN_LAYERS[2]),
# новые полигоны просто вычитаются из него. S2_FINE_PASS=True — дополнительный проход более мелкой сегментацией остатка (долго).
def do_stage2():
    global FIELDS
    base = get_input(2)
    n_new = 0
    if S2_FINE_PASS:
        new = run_seg_stage('s2', [S2_FINE], exclude=base, tone='pd', min_solidity=S2_MIN_SOLIDITY, kind='small')
    else:
        new = make_fields(get_raw1(), 'pd', S2['MIN_FIELD_HA'], S2['FILL_HOLES_HA'], S2['MIN_WIDTH_PX'], S2_MIN_SOLIDITY, 'small',
                          crop_min=S2['CROP_FRAC_MIN'], simplify_px=s1_param('SIMPLIFY_PX'))
    if new is not None and len(new):
        new = subtract_existing(new, base, S2['MIN_FIELD_HA'])
        new = shape_filter(new, None, S2['MAX_HOLE_FRAC'])
        new = crop_filter(new, label='(мелкие) ')
        new['kind'] = np.where(new.area_ha >= s1_param('MIN_FIELD_HA'), 'main', 'small')
        new = new[new.kind == 'small'] if not S2_FINE_PASS else new
        n_new = len(new)
        base = pd.concat([base, new], ignore_index=True)
    FIELDS = renumber(resolve_nested(gpd.GeoDataFrame(base, geometry='geometry', crs=CRS)))
    print(f'Этап 2: добавлено {n_new} мелких полей')
    save_stage(2, FIELDS)
    return FIELDS


STAGE_FUNCS[2] = do_stage2
if not COMPARE:
    do_stage2() if STAGES[2] else print('Этап 2 пропущен (STAGES[2]=0)')

# %% 12_stage3_yellow_fields
# «Жёлтые» поля — из тех же регионов (mean_min > YELLOW_MIN_THR), строже по форме и по WorldCover.
def do_stage3():
    global FIELDS
    base = get_input(3)
    new = make_fields(get_raw1(), 'y', S3['MIN_FIELD_HA'], S3['FILL_HOLES_HA'], S3['MIN_WIDTH_PX'], S3_MIN_SOLIDITY, 'yellow',
                      crop_min=S3['CROP_FRAC_MIN'], simplify_px=s1_param('SIMPLIFY_PX'))
    n_new = 0
    if new is not None and len(new):
        new = subtract_existing(new, base, S3['MIN_FIELD_HA'])
        new = shape_filter(new, S3['MAX_FIELD_HA'], S3['MAX_HOLE_FRAC'])
        new = crop_filter(new, label='(жёлтые) ')
        n_new = len(new)
        base = pd.concat([base, new], ignore_index=True)
    FIELDS = renumber(resolve_nested(gpd.GeoDataFrame(base, geometry='geometry', crs=CRS)))
    print(f'Этап 3: добавлено {n_new} «жёлтых» полей')
    save_stage(3, FIELDS)
    return FIELDS


STAGE_FUNCS[3] = do_stage3
if not COMPARE:
    do_stage3() if STAGES[3] else print('Этап 3 пропущен (STAGES[3]=0)')

# %% 13_field_pass_funcs
from scipy.ndimage import (binary_erosion, binary_opening, uniform_filter, label as ndi_label2, minimum as ndi_min_labels)
from rasterio.features import shapes as _shapes

CLASS_TREE = None


def find_spots(f, fm, wt):
    """Пятна внутри поля, резко отличающиеся по тону от основного тона поля.
    f — сглаженные каналы окна, fm — маска поля. Возвращает список (маска пятна, тип, средние каналы)."""
    npx = int(fm.sum())
    if npx < 50:
        return []
    core = binary_erosion(fm, iterations=SPOT_EDGE_PX)
    ref = fm if core.sum() < 30 else core
    med_max, med_min = np.median(f[ref, 0]), np.median(f[ref, 1])
    d = np.hypot(f[..., 0] - med_max, f[..., 1] - med_min)
    if med_max < DARK_MAX_THR:                                    # тёмное поле: только более светлые пятна (розовые/жёлтые)
        cand = (d >= SPOT_DIST_DARK) & (f[..., 0] - med_max >= 0.5 * SPOT_DIST_DARK)
    else:                                                         # розовое поле: жёлтые (min выше) и тёмные (max ниже)
        cand = (d >= SPOT_DIST_PINK) & ((f[..., 1] - med_min >= 0.4 * SPOT_DIST_PINK) | (med_max - f[..., 0] >= 0.6 * SPOT_DIST_PINK))
    cand &= fm
    cand = binary_opening(cand, structure=np.ones((3, 3), bool))
    if not cand.any():
        return []
    lab, n = ndi_label2(cand, structure=np.ones((3, 3), int))
    dist_out = distance_transform_edt(fm)                         # расстояние до внешней стороны поля
    min_px = SPOT_MIN_HA / px_area_ha
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    dmin = np.asarray(ndi_min_labels(dist_out, lab, index=np.arange(1, n + 1)))
    out = []
    for i in range(1, n + 1):
        if sizes[i] >= min_px and dmin[i - 1] >= SPOT_EDGE_PX and sizes[i] <= SPOT_MAX_FRAC * npx:
            m = binary_dilation(lab == i, iterations=1) & fm
            mm = f[lab == i].mean(0)
            typ = 'yellow' if mm[1] > YELLOW_MIN_THR else ('dark' if mm[0] < DARK_MAX_THR else 'pink')
            out.append((m, typ, mm))
    return out


def fit_class_tree():
    """CLASS_MODE='tree': дерево решений по пикселям (max, min, diff) на ваших размеченных полях."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    tr = read_any(CLASS_TRAIN_LAYER)
    assert CLASS_TRAIN_COL in tr.columns, f'нет колонки {CLASS_TRAIN_COL}'
    tr = tr[tr[CLASS_TRAIN_COL].notna()].reset_index(drop=True)
    rng = np.random.default_rng(0)
    X, y = [], []
    for geom, lab_ in tqdm(zip(tr.geometry.values, tr[CLASS_TRAIN_COL].values), total=len(tr), desc='обучение'):
        win, wt = field_window(geom)
        if win is None:
            continue
        feats, valid = read_window(win)
        m = rasterize([(geom, 1)], out_shape=feats.shape[:2], transform=wt, fill=0, dtype='uint8').astype(bool) & valid
        m = binary_erosion(m, iterations=2) if m.sum() > 100 else m
        px = feats[m]
        if len(px):
            px = px[rng.choice(len(px), min(len(px), 400), replace=False)]
            X.append(px); y += [lab_] * len(px)
    X, y = np.concatenate(X), np.array(y)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight='balanced', random_state=0).fit(X, y)
    print(export_text(clf, feature_names=['max', 'min', 'diff']))
    print('точность на обучающих пикселях:', round(clf.score(X, y), 3))
    return clf


def pixel_classes(f, allow_fallow=True):
    """f (h,w,3) сглаженные каналы → классы 1..N (rules: 1 засеянное, 2 пар, 3 неиспользуемое).
    allow_fallow=False (мелкие поля): пиксели «пар» относятся к «засеянному»."""
    if CLASS_MODE == 'tree':
        cls = (np.searchsorted(CLASS_TREE.classes_, CLASS_TREE.predict(f.reshape(-1, 3))) + 1).reshape(f.shape[:2]).astype(np.int8)
        if not allow_fallow:
            names = {v: k for k, v in CLASS_NAMES.items()}
            if 'пар' in names and 'засеянное' in names:
                cls[cls == names['пар']] = names['засеянное']
        return cls
    cls = np.ones(f.shape[:2], np.int8)                 # засеянное
    if allow_fallow:
        cls[f[..., 0] < FALLOW_MAX_THR] = 2             # пар: растительность за сезон не развилась
    cls[f[..., 1] >= UNUSED_MIN_THR] = 3                # неиспользуемое: min NDVI не падает (травостой/залежь/кустарник)
    if UNUSED_DIFF_MAX:                                 # … или слабый размах NDVI за сезон при развитой растительности (залежь)
        cls[(f[..., 2] < UNUSED_DIFF_MAX) & (f[..., 0] >= FALLOW_MAX_THR)] = 3
    return cls


def majority(cls, valid, win, ids):
    if win <= 1:
        return np.where(valid, cls, 0).astype(np.int8)
    st = np.stack([uniform_filter(((cls == c) & valid).astype(np.float32), win) for c in ids], 0)
    out = np.asarray(ids, np.int8)[st.argmax(0)]
    out[~valid] = 0
    return out


def split_polygons(cm, wt, fm, geom, ids):
    """Куски поля по классам: мелкие компоненты (< SPLIT_MIN_HA) поглощаются ближайшим классом."""
    lab = cm.copy()
    min_px = SPLIT_MIN_HA / px_area_ha
    for c in ids:
        cc, n = ndi_label2(lab == c)
        if n:
            sz = np.bincount(cc.ravel(), minlength=n + 1)
            bad = np.flatnonzero(sz < min_px); bad = bad[bad > 0]
            lab[np.isin(cc, bad)] = 0
    if (lab == 0)[fm].all():
        return []
    hole = (lab == 0) & fm
    if hole.any():
        idx = distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(hole, lab[idx[0], idx[1]], lab)
    out = []
    for g_, v in _shapes(lab.astype(np.int16), mask=fm & (lab > 0), transform=wt, connectivity=4):
        p_ = shp_shape(g_).intersection(geom)
        if not p_.is_empty:
            out.append((p_, int(v)))
    return out


def smooth3(feats, sigma):
    return np.stack([gaussian_filter(feats[..., k], sigma) for k in range(3)], -1)


def field_pass(fin, spots=True, classify=True):
    """Один проход по полям: окно растра читается и сглаживается ОДИН раз, в нём
    (4) ищутся и вырезаются пятна, затем (6) на оставшейся маске считаются классы пикселей.
    Возвращает (поля с вырезанными пятнами, поля с классами или None, слой пятен, слой разбивки или None)."""
    global CLASS_TREE, CLASS_NAMES
    fin = fin.reset_index(drop=True).copy()
    n = len(fin)
    if classify:
        if CLASS_MODE == 'tree':
            CLASS_TREE = fit_class_tree()
            CLASS_NAMES = {i + 1: str(c) for i, c in enumerate(CLASS_TREE.classes_)}
        ids = sorted(CLASS_NAMES)
    elig = np.zeros(n, bool)
    if spots:
        elig = np.ones(n, bool) if SPOT_KINDS is None else np.array(fin['kind'].isin(SPOT_KINDS).values, dtype=bool)
        elig = elig & np.asarray((fin.area_ha >= SPOT_MIN_FIELD_HA).values)
        print(f'пятна: полей к проверке {int(elig.sum())} из {n}')
    geoms = list(fin.geometry.values)
    n_spots, spots_ha = np.zeros(n, int), np.zeros(n)
    spot_rows, split_rows = [], []
    if classify:
        fr = {c: np.zeros(n) for c in ids}
        main_, minor_ = np.zeros(n, int), np.zeros(n, int)
        main_f, minor_f = np.zeros(n), np.zeros(n)
    for i in tqdm(range(n), desc='поля'):
        if not (elig[i] or classify):
            continue
        geom = geoms[i]
        win, wt = field_window(geom)
        if win is None:
            continue
        h, w = int(win.height), int(win.width)
        feats, valid = read_window(win)
        fm = rasterize([(geom, 1)], out_shape=(h, w), transform=wt, fill=0, dtype='uint8').astype(bool) & valid
        if fm.sum() == 0:
            continue
        fs = smooth3(feats, SPOT_SIGMA) if (elig[i] or CLASS_SIGMA == SPOT_SIGMA) else None
        fm_c = fm
        if elig[i]:
            sp = find_spots(fs, fm, wt)
            if sp:
                n_spots[i] = len(sp)
                for m, typ, mm in sp:
                    a_ha = float(m.sum()) * px_area_ha
                    spots_ha[i] += a_ha
                    for g_, v in _shapes(m.astype(np.uint8), mask=m, transform=wt, connectivity=8):
                        spot_rows.append(dict(field_id=int(fin['id'].values[i]) if 'id' in fin.columns else i, type=typ,
                                              area_ha=round(a_ha, 3), mean_max=round(mm[0], 1), mean_min=round(mm[1], 1),
                                              mean_diff=round(mm[2], 1), geometry=shp_shape(g_)))
                if SPOT_CUT:
                    allm = np.zeros((h, w), bool)
                    for m, _, _ in sp:
                        allm |= m
                    hp = [shp_shape(g_) for g_, v in _shapes(allm.astype(np.uint8), mask=allm, transform=wt, connectivity=8)]
                    geoms[i] = geom.difference(shapely.union_all(hp))
                    fm_c = fm & ~allm
        if classify:
            fc = fs if (fs is not None and CLASS_SIGMA == SPOT_SIGMA) else smooth3(feats, CLASS_SIGMA)
            cm = majority(pixel_classes(fc, allow_fallow=bool(fin['area_ha'].values[i] >= FALLOW_MIN_FIELD_HA)), fm_c, CLASS_SMOOTH_WIN, ids)
            cnt = np.bincount(cm[fm_c], minlength=max(ids) + 1)[ids].astype(float)
            if cnt.sum() == 0:
                continue
            frac = cnt / cnt.sum()
            order = np.argsort(-frac)
            main_[i], main_f[i] = ids[order[0]], frac[order[0]]
            if len(ids) > 1 and frac[order[1]] >= CLASS_MINOR_FRAC:
                minor_[i], minor_f[i] = ids[order[1]], frac[order[1]]
            for j, c in enumerate(ids):
                fr[c][i] = frac[j]
            base_row = {k: fin[k].values[i] for k in ('id', 'kind') if k in fin.columns}
            if SPLIT_BY_CLASS and minor_[i]:
                for p_, v in split_polygons(cm, wt, fm_c, geoms[i], [main_[i], minor_[i]] if len(ids) > 2 else ids):
                    split_rows.append(dict(base_row, field_id=base_row.get('id', i), cls=v, geometry=p_))
            else:
                split_rows.append(dict(base_row, field_id=base_row.get('id', i), cls=main_[i], geometry=geoms[i]))
    cut = fin.copy()
    cut['geometry'] = geoms
    if spots:
        cut['n_spots'], cut['spots_ha'] = n_spots, spots_ha.round(3)
    cut = add_area(cut)
    SPOTS = gpd.GeoDataFrame(spot_rows, geometry='geometry', crs=CRS) if spot_rows else gpd.GeoDataFrame(
        {'field_id': [], 'type': [], 'area_ha': []}, geometry=[], crs=CRS)
    if not classify:
        return cut, None, SPOTS, None
    cl = cut.copy()
    cl['cls_main'], cl['cls_main_name'], cl['cls_main_frac'] = main_, [CLASS_NAMES.get(v, '') for v in main_], main_f.round(3)
    cl['cls_minor'], cl['cls_minor_name'], cl['cls_minor_frac'] = minor_, [CLASS_NAMES.get(v, '') for v in minor_], minor_f.round(3)
    for c in ids:
        cl[f'frac_{c}'] = fr[c].round(3)
    SPLIT = None
    if SPLIT_BY_CLASS and split_rows:
        SPLIT = gpd.GeoDataFrame(split_rows, geometry='geometry', crs=CRS)
        SPLIT['cls_name'] = SPLIT['cls'].map(CLASS_NAMES)
        SPLIT = add_area(SPLIT)
    return cut, cl, SPOTS, SPLIT


def save_class_outputs(cl, split):
    global CLASS_SPLIT
    CLASS_SPLIT = split
    save_stage(6, cl)
    if split is not None and (SAVE_STAGES is None or 6 in SAVE_STAGES):
        split.to_file(stage_file(6), driver='GPKG', layer='fields_split')
        print(f'  → слой fields_split: {len(split)} кусков')
    print(cl.groupby('cls_main_name').area_ha.agg(['count', 'sum']).round(0))


# %% 14_stage4_6_run
# Этапы 4 и 6 — один проход по полям (окно читается один раз). Если включён этап 5 и CLASSIFY_AFTER_REFINE=True,
# классификация делается отдельным проходом уже после уточнения границ.
def fused_4_6():
    return bool(stage_on(4) and stage_on(6) and not (stage_on(5) and CLASSIFY_AFTER_REFINE))


def do_stage4():
    global FIELDS, SPOTS_LAYER
    fuse = fused_4_6()
    cut, cl, spots_gdf, split = field_pass(get_input(4), spots=True, classify=fuse)
    SPOTS_LAYER = spots_gdf
    if len(spots_gdf):
        print('найдено пятен:', len(spots_gdf), '| по типам:', spots_gdf['type'].value_counts().to_dict(), f'| {spots_gdf.area_ha.sum():.0f} га')
        if SAVE_STAGES is None:
            spots_gdf.to_file(Path(OUT_DIR) / f'spots_{RUN_MODE}{TAG}.gpkg', driver='GPKG', layer='spots')
    FIELDS = cut
    save_stage(4, FIELDS)
    if fuse:
        FIELDS = cl
        save_class_outputs(cl, split)
    return FIELDS


STAGE_FUNCS[4] = do_stage4
if not COMPARE:
    do_stage4() if STAGES[4] else print('Этап 4 пропущен (STAGES[4]=0)')

# %% 14b_refine_funcs
_ORIGIN = 20037508.342789244
WORK_CRS = None            # метрическая CRS для этапа 2 (задаётся в ячейке 11)


class BasemapBlocked(RuntimeError):
    pass


def _fetch_tile(args):
    z, x, y = args
    p = Path(BASEMAP_CACHE) / str(z) / str(x) / f'{y}.jpg'
    if p.exists() and p.stat().st_size > 200:
        return p
    import requests
    p.parent.mkdir(parents=True, exist_ok=True)
    url = BASEMAP_URL.format(x=x, y=y, z=z)
    blocked = 0
    for k in range(4):
        try:
            r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200 and len(r.content) > 200:
                p.write_bytes(r.content)
                return p
            if r.status_code in (403, 429):
                blocked += 1
                time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(1 + k)
    if blocked >= 3:
        raise BasemapBlocked(f'сервер тайлов отвечает 403/429 ({url}) — остановка, чтобы не нагружать его')
    return None


def basemap_tile_range(bounds_work, z=None):
    z = z or BASEMAP_ZOOM
    mb = rasterio.warp.transform_bounds(WORK_CRS, 'EPSG:3857', *bounds_work, densify_pts=21)
    size = 2 * _ORIGIN / 2 ** z
    tx0, tx1 = int((mb[0] + _ORIGIN) // size), int((mb[2] + _ORIGIN) // size)
    ty0, ty1 = int((_ORIGIN - mb[3]) // size), int((_ORIGIN - mb[1]) // size)
    return z, size, tx0, tx1, ty0, ty1


def basemap_rgb(Tb, shape):
    """RGB подложки на сетке Tb (метрическая WORK_CRS), shape=(h,w). None — если тайлы не скачались."""
    from PIL import Image
    h, w = shape
    bounds = (Tb.c, Tb.f + h * Tb.e, Tb.c + w * Tb.a, Tb.f)
    z, size, tx0, tx1, ty0, ty1 = basemap_tile_range(bounds)
    coords = [(z, x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(BASEMAP_WORKERS) as ex:
        paths = list(ex.map(_fetch_tile, coords))
    if sum(p is None for p in paths) > 0.2 * len(paths):
        return None
    mos = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), 128, np.uint8)
    for (zz, x, y), p in zip(coords, paths):
        if p is not None:
            mos[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = \
                np.asarray(Image.open(p).convert('RGB'))
    res_z = size / 256
    Ts = Affine(res_z, 0, -_ORIGIN + tx0 * size, 0, -res_z, _ORIGIN - ty0 * size)
    dst = np.zeros((3, h, w), np.uint8)
    reproject(source=np.moveaxis(mos, -1, 0), destination=dst, src_transform=Ts, src_crs='EPSG:3857',
              dst_transform=Tb, dst_crs=WORK_CRS, resampling=Resampling.bilinear)
    return np.moveaxis(dst, 0, -1)


def _color_gradient(rgb, sigma):
    s = np.stack([gaussian_filter(rgb[..., k].astype(np.float32), sigma) for k in range(3)], -1)
    return np.sqrt(sum(sobel(s[..., k]) ** 2 for k in range(3))).astype(np.float32)


def refine_block(rgb, L, passes, compactness=0.0, prior=0.0):
    """Маркерный watershed по градиенту подложки. L — метки полей (0 — не поле).
    Маркеры: ядро каждого поля (сжатое на e_field), фон (сжатый на e_bg); неопределённая зона между ними
    заливается по градиенту → граница садится на сильнейший край. Проходы идут от грубого масштаба к тонкому."""
    L = L.astype(np.int32)
    n_lab = int(L.max())
    BG = n_lab + 1
    for ps in passes:
        ef, eb = float(ps['e_field']), float(ps['e_bg'])
        bnd = np.zeros(L.shape, bool)
        d = L[:, :-1] != L[:, 1:]; bnd[:, :-1] |= d; bnd[:, 1:] |= d
        d = L[:-1, :] != L[1:, :]; bnd[:-1, :] |= d; bnd[1:, :] |= d
        if not bnd.any():
            break
        dist = distance_transform_edt(~bnd)
        maxd = np.asarray(ndi_maximum(dist, L, index=np.arange(n_lab + 1)), dtype=np.float64)
        thr_f = np.minimum(ef, 0.5 * maxd)                    # у тонких полигонов маркер всё равно остаётся
        lim = max(ef, eb) + 2
        mask = dist <= lim                                    # неопределённая зона + кольцо маркеров
        core_f = (L > 0) & (dist > thr_f[L])
        core_b = (L == 0) & (dist > eb)
        markers = np.zeros(L.shape, np.int32)
        markers[core_f & mask] = L[core_f & mask]
        markers[core_b & mask] = BG
        g = _color_gradient(rgb, ps['sigma'])
        if prior:
            g = g + prior * dist.astype(np.float32)
        ws = watershed(g, markers, mask=mask, connectivity=1, compactness=compactness)
        newL = np.where(mask & (ws > 0), ws, L)
        newL[newL == BG] = 0
        L = newL.astype(np.int32)
    return L


def refine_fields(gdf, get_rgb, res=1.0, block_px=2000, margin_px=60, passes=None, compactness=0.0, prior=0.0,
                  max_area_change=0.4, simplify_m=0.7, log=print):
    """Уточнение полигонов (метрическая CRS) по блокам. get_rgb(Tb, (h,w)) → RGB (h,w,3) или None.
    Глобальная пиксельная сетка с целыми координатами → фрагменты поля из соседних блоков стыкуются точно."""
    passes = passes or REFINE_PASSES
    geoms = list(gdf.geometry.values)
    minx, miny, maxx, maxy = gdf.total_bounds
    ox, oy = math.floor(minx / res) * res, math.ceil(maxy / res) * res
    ncols, nrows = math.ceil((maxx - ox) / res), math.ceil((oy - miny) / res)
    gpx = [affine_transform(g, [1 / res, 0, 0, -1 / res, -ox / res, oy / res]) for g in geoms]
    tree = STRtree(gpx)
    nbx, nby = math.ceil(ncols / block_px), math.ceil(nrows / block_px)
    frags, failed, done = defaultdict(list), 0, 0
    for by in range(nby):
        for bx in range(nbx):
            c0, r0 = bx * block_px, by * block_px
            c1, r1 = min(c0 + block_px, ncols), min(r0 + block_px, nrows)
            gc0, gr0, gc1, gr1 = c0 - margin_px, r0 - margin_px, c1 + margin_px, r1 + margin_px
            idx = tree.query(shp_box(gc0, gr0, gc1, gr1), predicate='intersects')
            if idx.size == 0:
                continue
            h, w = gr1 - gr0, gc1 - gc0
            Tb = Affine(res, 0, ox + gc0 * res, 0, -res, oy - gr0 * res)
            rgb = get_rgb(Tb, (h, w))
            if rgb is None:
                failed += 1; continue
            L = rasterize([(gpx[i], k + 1) for k, i in enumerate(idx)], out_shape=(h, w),
                          transform=Affine(1, 0, gc0, 0, 1, gr0), fill=0, dtype='int32')
            L2 = refine_block(rgb, L, passes, compactness, prior)
            Lc = L2[margin_px:margin_px + (r1 - r0), margin_px:margin_px + (c1 - c0)]
            for geom, v in rio_shapes(Lc, mask=Lc > 0, transform=Affine(1, 0, c0, 0, 1, r0), connectivity=4):
                frags[int(idx[int(v) - 1])].append(shp_shape(geom))
            done += 1
            log(f'блок {done}: x{bx} y{by}, полигонов {idx.size}')
    out = list(geoms)
    refined = np.zeros(len(geoms), bool)
    for i, fr in frags.items():
        u = shapely.union_all(fr)
        if u.geom_type == 'MultiPolygon':
            big = max(p.area for p in u.geoms)
            u = MultiPolygon([p for p in u.geoms if p.area >= max(0.02 * big, 50)])
        gm = affine_transform(u, [res, 0, 0, -res, ox, oy])
        if abs(gm.area / geoms[i].area - 1) <= max_area_change:
            out[i] = gm; refined[i] = True
    if simplify_m > 0 and refined.any():
        try:
            out = list(shapely.coverage_simplify(np.array(out, dtype=object), simplify_m))
        except Exception:
            out = [shapely.simplify(g, simplify_m, preserve_topology=True) for g in out]
    res_gdf = gdf.copy()
    res_gdf['geometry'] = gpd.GeoSeries(out, index=gdf.index, crs=gdf.crs)
    res_gdf['refined'] = refined
    log(f'уточнено полигонов: {int(refined.sum())} из {len(geoms)}; блоков без подложки: {failed}')
    return res_gdf


# %% 15_stage5_refine
def do_stage5():
    global FIELDS, WORK_CRS
    f0 = get_input(5)
    crs0 = f0.crs
    if crs0.is_projected:
        WORK_CRS = crs0
    else:
        lon0 = f0.to_crs(4326).total_bounds[[0, 2]].mean()
        WORK_CRS = rasterio.crs.CRS.from_epsg(32600 + int((lon0 + 180) // 6) + 1)
    fw = f0.to_crs(WORK_CRS)
    sel = np.ones(len(fw), bool) if REFINE_KINDS is None else np.array(fw['kind'].isin(REFINE_KINDS).values, dtype=bool)
    if REFINE_AOI_M and RUN_MODE == 'test':
        cx, cy = [(a + b) / 2 for a, b in zip(fw.total_bounds[:2], fw.total_bounds[2:])]
        aoi = shp_box(cx - REFINE_AOI_M / 2, cy - REFINE_AOI_M / 2, cx + REFINE_AOI_M / 2, cy + REFINE_AOI_M / 2)
        sel = sel & np.asarray(fw.geometry.within(aoi).values)
    sub = fw[sel].copy()
    print(f'Этап 5: полигонов к уточнению {len(sub)} из {len(fw)}')
    z, size, tx0, tx1, ty0, ty1 = basemap_tile_range(tuple(sub.total_bounds))
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f'zoom {z}: ~{n_tiles} тайлов (≈{n_tiles * 20 / 1024:.0f} МБ при 20 КБ/тайл)')
    assert n_tiles <= REFINE_MAX_TILES, f'{n_tiles} тайлов > REFINE_MAX_TILES — уменьшите область/zoom или поднимите лимит'
    ref = refine_fields(sub, basemap_rgb, res=REFINE_RES_M, block_px=REFINE_BLOCK_PX, margin_px=REFINE_MARGIN_PX,
                        passes=REFINE_PASSES, compactness=REFINE_COMPACT, prior=REFINE_PRIOR,
                        max_area_change=REFINE_MAX_AREA_CHANGE)
    full = pd.concat([fw[~sel].assign(refined=False), ref], ignore_index=True)
    full = gpd.GeoDataFrame(full, geometry='geometry', crs=WORK_CRS).to_crs(CRS)
    FIELDS = add_area(renumber(full))
    save_stage(5, FIELDS)
    # до/после на подложке (центральный участок)
    b = sub.total_bounds
    cxm, cym = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = 900
    px = int(2 * half / REFINE_RES_M)
    Tv = Affine(REFINE_RES_M, 0, cxm - half, 0, -REFINE_RES_M, cym + half)
    img = basemap_rgb(Tv, (px, px))
    if img is not None:
        fig, axs = plt.subplots(1, 2, figsize=(18, 9))
        for ax, g_, ttl in zip(axs, (sub, ref), ('до (10 м)', 'после уточнения по подложке')):
            ax.imshow(img, extent=(cxm - half, cxm + half, cym - half, cym + half))
            g_.cx[cxm - half:cxm + half, cym - half:cym + half].boundary.plot(ax=ax, color='yellow', linewidth=0.8)
            ax.set_title(ttl); ax.set_axis_off()
        plt.tight_layout(); plt.savefig(f'{OUT_DIR}/refine_before_after.png', dpi=80); plt.show()
    return FIELDS


STAGE_FUNCS[5] = do_stage5
if not COMPARE:
    do_stage5() if STAGES[5] else print('Этап 5 пропущен (STAGES[5]=0)')

# %% 16_stage6_after_refine
# Классификация после уточнения границ (этап 5) — отдельный проход по полям. Если этапы 4 и 6 уже выполнены одним проходом
# (fused_4_6), здесь ничего не делается. Если этап 4 выключен, а 6 включён — классификация идёт только здесь.
def do_stage6():
    global FIELDS
    if fused_4_6():
        return FIELDS
    _, cl, _, split = field_pass(get_input(6), spots=False, classify=True)
    FIELDS = cl
    save_class_outputs(cl, split)
    return FIELDS


STAGE_FUNCS[6] = do_stage6
if not COMPARE:
    do_stage6() if STAGES[6] else print('Этап 6 пропущен (STAGES[6]=0)')

# %% 17_final
def final_outputs(overview=True):
    """Итог: fields_final_<режим><метка>.gpkg (слой fields; fields_split — разбивка по классам; spots — пятна) + картинка обзора."""
    global FIELDS, CLASS_SPLIT
    final_path = Path(OUT_DIR) / f'fields_final_{RUN_MODE}{TAG}.gpkg'
    FIELDS = regularize_boundaries(FIELDS)
    CLASS_SPLIT = regularize_boundaries(CLASS_SPLIT)
    FIELDS.to_file(final_path, driver='GPKG', layer='fields')
    if CLASS_SPLIT is not None:
        CLASS_SPLIT.to_file(final_path, driver='GPKG', layer='fields_split')
    if SPOTS_LAYER is not None and len(SPOTS_LAYER):
        SPOTS_LAYER.to_file(final_path, driver='GPKG', layer='spots')
    print('итог:', final_path, '|', len(FIELDS), 'полей |', FIELDS.kind.value_counts().to_dict() if 'kind' in FIELDS.columns else '')
    if overview:
        win = compare_window(1800)
        plot_compare({'итог': FIELDS}, win=win, cols=1, panel_px=1200, path=Path(OUT_DIR) / f'overview_{RUN_MODE}{TAG}.png', color_by_class=True)


if not COMPARE and FIELDS is not None:
    final_outputs()

# %% 18_compare
# Режим сравнения пресетов (COMPARE_PRESETS=True и RUN_MODE='test'): КАЖДЫЙ пресет проходит все включённые этапы; на диск пишутся
# только слой после этапа 1 (fields_after_stage1_test_<пресет>.gpkg) и финал (fields_final_test_<пресет>.gpkg: fields, fields_split, spots).
# Сегментация кэшируется по параметрам, поэтому пресеты, отличающиеся только этапами (SPLIT, NO_STAGES), считаются почти мгновенно.
COMPARE_ROWS, COMPARE_RES = [], {}


def run_all(nm):
    """Один пресет: этапы 1…6 подряд (по STAGES и перекрытиям пресета), финал → fields_final_<режим>_<пресет>.gpkg."""
    global STAGE1_PRESET, RAW1, FIELDS, TAG, SAVE_STAGES, SPLIT_OVER, NO_STAGES_CUR, CLASS_SPLIT, SPOTS_LAYER
    t0 = time.time()
    P = PRESETS[nm]
    STAGE1_PRESET, RAW1, FIELDS, TAG = nm, None, None, f'_{nm}'
    SAVE_STAGES = {1}
    SPLIT_OVER, NO_STAGES_CUR = P.get('SPLIT', {}), set(P.get('NO_STAGES', []))
    CLASS_SPLIT = SPOTS_LAYER = None
    row, s1 = dict(preset=nm), None
    for k in ORDER:
        if stage_on(k) and k in STAGE_FUNCS:
            STAGE_FUNCS[k]()
            if k == 1:
                s1 = FIELDS.copy()
                d = describe(s1)
                row.update(n_s1=d['n'], max_ha_s1=d['max_ha'], giants_s1=d['n_giant'])
    final_outputs(overview=False)
    d = describe(FIELDS)
    row.update(n=d['n'], median_ha=d['median_ha'], p90_ha=d['p90_ha'], max_ha=d['max_ha'], giants=d['n_giant'], giant_share=d['giant_share'])
    if 'cls_main_name' in FIELDS.columns:
        sh = FIELDS.groupby('cls_main_name').area_ha.sum() / FIELDS.area_ha.sum()
        row.update({f'%{c}': round(100 * float(sh.get(c, 0)), 1) for c in CLASS_NAMES.values()})
    row['min'] = round((time.time() - t0) / 60, 1)
    COMPARE_ROWS.append(row)
    COMPARE_RES[nm] = (s1, FIELDS.copy())
    return row


if COMPARE:
    names = COMPARE_NAMES or [n_ for n_ in PRESETS if not n_.startswith('OLD')]
    for nm in names:
        print(f'\n══════ пресет {nm} ══════')
        try:
            run_all(nm)
        except Exception as e_:
            import traceback; traceback.print_exc()
            COMPARE_ROWS.append(dict(preset=nm, error=str(e_)[:80]))
    table = pd.DataFrame(COMPARE_ROWS)
    print('\n', table.to_string(index=False))
    table.to_csv(Path(OUT_DIR) / f'compare_summary_{RUN_MODE}.csv', index=False)
    ok_ = {n_: v for n_, v in COMPARE_RES.items()}
    for ci, ctr in enumerate(COMPARE_CENTERS):
        win = compare_window(COMPARE_WIN_PX, ctr)
        plot_compare({n_: v[0] for n_, v in ok_.items()}, win=win, path=f'{OUT_DIR}/compare_stage1_{ci}.png')
        plot_compare({n_: v[1] for n_, v in ok_.items()}, win=win, path=f'{OUT_DIR}/compare_final_{ci}.png', color_by_class=True)
