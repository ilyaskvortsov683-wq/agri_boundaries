# %% 10b_stage1b_split
# ───────────── Этап 1б: разбивка слипшихся полигонов ─────────────
# Крупные полигоны, в которые слиплось несколько полей, режутся по длинным тонким линиям (лесополосы, дороги, межи)
# и по резким ступенькам тона. Линии ищутся ориентированными фильтрами (усреднение ВДОЛЬ линии, поэтому ловятся
# слабые линии, которые top-hat барьеры этапа 1 пропускают), затем отсеиваются по длине. Работает только внутри
# выбранных полигонов, растр читается окнами — это дёшево и не требует повторной сегментации.
from scipy.ndimage import binary_erosion, binary_dilation, label as ndi_label3, distance_transform_edt as _edt


def split_labels(feats, fm, P):
    """Метки кусков внутри маски поля fm (0 — вне поля) или None, если разбивать нечем."""
    R = split_line_response(feats, P['SENS'], P['SIGMA_ALONG'])
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
    FIELDS = renumber(split_fields(get_input('1b'), SPLIT_OVER)[0])      # SPLIT_OVER — перекрытия SPLIT из пресета
    save_stage('1b', FIELDS)
    return FIELDS


STAGE_FUNCS['1b'] = do_stage1b
if not COMPARE:
    do_stage1b() if STAGES['1b'] else print('Этап 1б (разбивка) пропущен (STAGES["1b"]=0)')
