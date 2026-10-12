
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


def save_class_outputs(cl, SPLIT):
    save_stage(6, cl)
    if SPLIT is not None:
        SPLIT.to_file(stage_file(6), driver='GPKG', layer='fields_split')
        print(f'  → слой fields_split: {len(SPLIT)} кусков')
    print(cl.groupby('cls_main_name').area_ha.agg(['count', 'sum']).round(0))


# %% 14_stage4_6_run
# Этапы 4 и 6 — один проход по полям (окно читается один раз). Если включён этап 5 и CLASSIFY_AFTER_REFINE=True,
# классификация делается отдельным проходом уже после уточнения границ.
FUSE_4_6 = bool(STAGES[4] and STAGES[6] and not (STAGES[5] and CLASSIFY_AFTER_REFINE))
if STAGES[4]:
    cut, cl, SPOTS, SPLIT = field_pass(get_input(4), spots=True, classify=FUSE_4_6)
    if len(SPOTS):
        print('найдено пятен:', len(SPOTS), '| по типам:', SPOTS['type'].value_counts().to_dict(), f'| {SPOTS.area_ha.sum():.0f} га')
        SPOTS.to_file(Path(OUT_DIR) / f'spots_{RUN_MODE}.gpkg', driver='GPKG', layer='spots')
    FIELDS = cut
    save_stage(4, FIELDS)
    if FUSE_4_6:
        FIELDS = cl
        save_class_outputs(cl, SPLIT)
else:
    print('Этап 4 пропущен (STAGES[4]=0)')
