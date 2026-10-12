
# %% 13_stage4_spots
from scipy.ndimage import (binary_erosion, binary_opening, uniform_filter, label as ndi_label2)
from rasterio.features import shapes as _shapes


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


def detect_spots(geom, area_ha):
    """Пятна внутри поля, резко отличающиеся по тону от основного тона поля. Возвращает список (polygon, тип, средние)."""
    win, wt = field_window(geom)
    if win is None:
        return []
    h, w = int(win.height), int(win.width)
    feats, valid = read_window(win)
    fm = rasterio.features.rasterize([(geom, 1)], out_shape=(h, w), transform=wt, fill=0, dtype='uint8').astype(bool) & valid
    npx = int(fm.sum())
    if npx < 50:
        return []
    f = np.stack([gaussian_filter(feats[..., k], SPOT_SIGMA) for k in range(3)], -1)
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
    dmin = np.asarray(ndi_minimum_labels(dist_out, lab, n))
    keep = [i for i in range(1, n + 1) if sizes[i] >= min_px and dmin[i - 1] >= SPOT_EDGE_PX and sizes[i] <= SPOT_MAX_FRAC * npx]
    out = []
    for i in keep:
        m = binary_dilation(lab == i, iterations=1) & fm
        mm = f[lab == i].mean(0)
        typ = 'yellow' if mm[1] > YELLOW_MIN_THR else ('dark' if mm[0] < DARK_MAX_THR else 'pink')
        polys = [shp_shape(g_) for g_, v in _shapes(m.astype(np.uint8), mask=m, transform=wt, connectivity=8)]
        for p_ in polys:
            out.append((p_, typ, mm))
    return out


def ndi_minimum_labels(arr, lab, n):
    from scipy.ndimage import minimum as _min
    return _min(arr, lab, index=np.arange(1, n + 1))


if STAGES[4]:
    fin = get_input(4)
    elig = np.ones(len(fin), bool) if SPOT_KINDS is None else np.array(fin['kind'].isin(SPOT_KINDS).values, dtype=bool)
    elig &= (fin.area_ha >= SPOT_MIN_FIELD_HA).values
    print(f'Этап 4: полей к проверке {int(elig.sum())} из {len(fin)}')
    spot_rows, new_geoms = [], list(fin.geometry.values)
    n_spots = np.zeros(len(fin), int); spots_ha = np.zeros(len(fin))
    for i in tqdm(np.flatnonzero(elig), desc='пятна'):
        sp = detect_spots(fin.geometry.values[i], fin.area_ha.values[i])
        if not sp:
            continue
        holes = shapely.union_all([p_ for p_, _, _ in sp])
        for p_, typ, mm in sp:
            a_ha = (gpd.GeoSeries([p_], crs=CRS).to_crs('EPSG:6933').area.iloc[0] / 1e4) if not CRS.is_projected else p_.area / 1e4
            spot_rows.append(dict(field_id=int(fin['id'].values[i]) if 'id' in fin.columns else i, type=typ, area_ha=round(a_ha, 3),
                                  mean_max=round(mm[0], 1), mean_min=round(mm[1], 1), mean_diff=round(mm[2], 1), geometry=p_))
        n_spots[i] = len(sp); spots_ha[i] = sum(r['area_ha'] for r in spot_rows[-len(sp):])
        if SPOT_CUT:
            g2 = new_geoms[i].difference(holes)
            new_geoms[i] = g2
    fin = fin.copy()
    fin['geometry'] = new_geoms
    fin['n_spots'] = n_spots; fin['spots_ha'] = spots_ha.round(3)
    SPOTS = gpd.GeoDataFrame(spot_rows, geometry='geometry', crs=CRS) if spot_rows else gpd.GeoDataFrame(
        {'field_id': [], 'type': [], 'area_ha': []}, geometry=[], crs=CRS)
    if len(SPOTS):
        print('найдено пятен:', len(SPOTS), '| по типам:', SPOTS['type'].value_counts().to_dict(), f'| {SPOTS.area_ha.sum():.0f} га')
        SPOTS.to_file(Path(OUT_DIR) / f'spots_{RUN_MODE}.gpkg', driver='GPKG', layer='spots')
    FIELDS = add_area(fin)
    save_stage(4, FIELDS)
else:
    print('Этап 4 пропущен (STAGES[4]=0)')
