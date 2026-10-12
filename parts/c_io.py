
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
