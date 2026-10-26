
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


_EXT = {}


def ext_window(t_halo, h, w):
    """Внешняя карта границ на сетке окна тайла → bool (h,w) или None.
    EXT_TIF — путь к GeoTIFF (тогда канал EXT_BAND, порог EXT_THR) или список источников
    [dict(TIF=…, BAND=1, THR=0.3), …] — линии всех источников объединяются (например, FTW + многодатные границы).
    Растр может быть на любой сетке: перепроецируется максимумом (тонкие линии не теряются); затем порог и
    утончение до 1 px (EXT_SKEL)."""
    if not EXT_TIF:
        return None
    from rasterio.warp import reproject, Resampling
    srcs = EXT_TIF if isinstance(EXT_TIF, (list, tuple)) else [dict(TIF=EXT_TIF, BAND=EXT_BAND, THR=EXT_THR)]
    m = np.zeros((h, w), bool)
    for d in srcs:
        pth = Path(d['TIF']) if os.path.isabs(str(d['TIF'])) else Path(DRIVE_DIR) / d['TIF']
        key = (os.getpid(), str(pth))
        if key not in _EXT:
            _EXT[key] = rasterio.open(pth)
        s = _EXT[key]; band = int(d.get('BAND', 1))
        a = np.zeros((h, w), np.float32)
        reproject(rasterio.band(s, band), a, dst_transform=t_halo, dst_crs=CRS, dst_nodata=0, resampling=Resampling.max)
        a = a / 255.0 if s.dtypes[band - 1] == 'uint8' else a
        m |= a >= float(d.get('THR', EXT_THR))
    if EXT_SKEL and m.any():
        m = skeletonize(m)
    return m


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
    lab, _ = segment_regions(feats, valid, crop, wc, ext_window(t_halo, h, w))
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
