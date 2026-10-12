# %% 10b_stage1b_split
# ───────────── Этап 1б: разбивка слипшихся полигонов ─────────────
# Крупные полигоны, в которые слиплось несколько полей, режутся по длинным тонким линиям (лесополосы, дороги, межи)
# и по резким ступенькам тона. Линии ищутся ориентированными фильтрами (усреднение ВДОЛЬ линии, поэтому ловятся
# слабые линии, которые top-hat барьеры этапа 1 пропускают), затем отсеиваются по длине. Работает только внутри
# выбранных полигонов, растр читается окнами — это дёшево и не требует повторной сегментации.
from scipy.signal import fftconvolve
from scipy.ndimage import binary_erosion, binary_dilation, label as ndi_label3, distance_transform_edt as _edt
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold

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


def _line_resp(feats, sens, s_along):
    s = np.stack([gaussian_filter(feats[..., k], 0.7) for k in range(2)], -1).astype(np.float32)
    mx, mn = s[..., 0], s[..., 1]
    b = _SPLIT_BASE
    return np.maximum.reduce([_oriented(mn, 'edge', 1.3, s_along) * sens / b['E_MIN'],
                              _oriented(mn, 'ridge', 1.3, s_along) * sens / b['R_MIN'],
                              _oriented(mx, 'edge', 1.3, s_along) * sens / b['E_MAX']])


def split_line_response(feats, sens, s_along, block=2500, halo=24):
    """Нормированный отклик «линия/ступенька» (≥ 1 — уверенная линия). Каналы: min (ступенька и тонкая линия), max (ступенька).
    Окна больше block×block считаются блоками с перекрытием (экономия памяти для полигонов-гигантов)."""
    h, w = feats.shape[:2]
    if h * w <= 1.5 * block * block:
        return _line_resp(feats, sens, s_along)
    R = np.zeros((h, w), np.float32)
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            ra, rb, ca, cb = max(r0 - halo, 0), min(r0 + block + halo, h), max(c0 - halo, 0), min(c0 + block + halo, w)
            Rb = _line_resp(feats[ra:rb, ca:cb], sens, s_along)
            R[r0:r0 + block, c0:c0 + block] = Rb[r0 - ra:r0 - ra + min(block, h - r0), c0 - ca:c0 - ca + min(block, w - c0)]
    return R


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


if STAGES['1b']:
    FIELDS = renumber(split_fields(get_input('1b'))[0])
    save_stage('1b', FIELDS)
else:
    print('Этап 1б (разбивка) пропущен (STAGES["1b"]=0)')
