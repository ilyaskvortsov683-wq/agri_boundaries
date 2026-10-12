
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


_JOB = {}


def _seg_job(tile):
    return seg_tile(tile, _JOB['tdir'], _JOB['exclude'], _JOB['ex_tree'])


def seg_raw(name, overrides, exclude=None):
    """Один проход по тайлам: сегментация → «сырые» регионы (все тона и размеры) → склейка швов.
    Результат кэшируется по тайлам; тайлы считаются параллельно (N_WORKERS процессов)."""
    set_params(*overrides)
    p = {k: globals()[k] for k in BASE}
    p.update(TILE=TILE, HALO=HALO, REGION=REGION, AGRI=AGRI_CLASSES, WC=USE_WORLDCOVER, EX=layer_hash(exclude), V=3)
    h = hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]
    tdir = Path(OUT_DIR) / f'tiles3_{RUN_MODE}_{name}_{h}'
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
    return merge_across_seams(raw, SEAM_THR, MIN_CONTACT)


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


def compare_window(size=1400):
    cx, cy = (X0 + X1) // 2, (Y0 + Y1) // 2
    c0, r0 = max(cx - size // 2, X0), max(cy - size // 2, Y0)
    return Window(c0, r0, min(size, X1 - c0), min(size, Y1 - r0))


def plot_compare(results, win=None, cols=3, panel_px=800, path=None):
    """Одно и то же окно композита (R=max, G=min, B=diff) + контуры каждого набора. Для скриншотов."""
    win = win or compare_window()
    rgb = np.moveaxis(src.read([1, 2, 3], window=win, out_shape=(3, panel_px, panel_px)), 0, -1).astype(np.float32)
    rgb = np.clip(rgb / 255, 0, 1)
    wt = rasterio.windows.transform(win, T)
    xa, ya = wt * (0, 0); xb, yb = wt * (win.width, win.height)
    names = list(results)
    rows = math.ceil(len(names) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, n in zip(axs.ravel(), names):
        ax.imshow(rgb, extent=(xa, xb, yb, ya))
        sub = results[n].cx[min(xa, xb):max(xa, xb), min(ya, yb):max(ya, yb)]
        sub.boundary.plot(ax=ax, color='yellow', linewidth=0.7)
        ax.set_title(f'{n}  (полей: {len(sub)})', fontsize=11)
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


if STAGES[1]:
    if COMPARE_PRESETS and RUN_MODE == 'test':
        S1_RESULTS = {}
        for nm in PRESETS:
            t0 = time.time()
            r = run_seg_stage(f's1_{nm}', [PRESETS[nm]], tone='pd', kind='main')
            if r is not None:
                S1_RESULTS[nm] = r
                a = r.area_ha
                SUMMARY.append(dict(preset=nm, fields=len(r), median_ha=round(a.median(), 1), p10_ha=round(a.quantile(.1), 1),
                                    p90_ha=round(a.quantile(.9), 1), total_ha=int(a.sum()), minutes=round((time.time() - t0) / 60, 1)))
        print(pd.DataFrame(SUMMARY).to_string(index=False))
        plot_compare(S1_RESULTS, path=f'{OUT_DIR}/compare_stage1.png')
    r = make_fields(get_raw1(), 'pd', s1_param('MIN_FIELD_HA'), s1_param('FILL_HOLES_HA'), s1_param('MIN_WIDTH_PX'), 0.0, 'main',
                    simplify_px=s1_param('SIMPLIFY_PX'))
    if r is None or r.empty:
        raise RuntimeError('Этап 1: полей нет')
    FIELDS = renumber(r)
    save_stage(1, FIELDS)
else:
    print('Этап 1 пропущен (STAGES[1]=0)')

# %% 11_stage2_small_fields
# Мелкие поля берутся из тех же регионов единственного прохода (get_raw1), без повторной сегментации:
# фильтры — тон, площадь ≥ S2['MIN_FIELD_HA'], форма (толщина, выпуклость), доля пашни. Если задан свой слой (IN_LAYERS[2]),
# новые полигоны просто вычитаются из него. S2_FINE_PASS=True — дополнительный проход более мелкой сегментацией остатка (долго).
if STAGES[2]:
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
        new['kind'] = np.where(new.area_ha >= s1_param('MIN_FIELD_HA'), 'main', 'small')
        new = new[new.kind == 'small'] if not S2_FINE_PASS else new
        n_new = len(new)
        base = pd.concat([base, new], ignore_index=True)
    FIELDS = renumber(resolve_nested(gpd.GeoDataFrame(base, geometry='geometry', crs=CRS)))
    print(f'Этап 2: добавлено {n_new} мелких полей')
    save_stage(2, FIELDS)
else:
    print('Этап 2 пропущен (STAGES[2]=0)')

# %% 12_stage3_yellow_fields
# «Жёлтые» поля — из тех же регионов (mean_min > YELLOW_MIN_THR), строже по форме и по WorldCover.
if STAGES[3]:
    base = get_input(3)
    new = make_fields(get_raw1(), 'y', S3['MIN_FIELD_HA'], S3['FILL_HOLES_HA'], S3['MIN_WIDTH_PX'], S3_MIN_SOLIDITY, 'yellow',
                      crop_min=S3['CROP_FRAC_MIN'], simplify_px=s1_param('SIMPLIFY_PX'))
    n_new = 0
    if new is not None and len(new):
        new = subtract_existing(new, base, S3['MIN_FIELD_HA'])
        new = shape_filter(new, S3['MAX_FIELD_HA'], S3['MAX_HOLE_FRAC'])
        n_new = len(new)
        base = pd.concat([base, new], ignore_index=True)
    FIELDS = renumber(resolve_nested(gpd.GeoDataFrame(base, geometry='geometry', crs=CRS)))
    print(f'Этап 3: добавлено {n_new} «жёлтых» полей')
    save_stage(3, FIELDS)
else:
    print('Этап 3 пропущен (STAGES[3]=0)')
