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


def detect_line_barriers(feats, valid, sens, s_along, min_px, edge_px=0):
    """Длинные прямые тонкие линии и ступеньки тона (межи, дороги, слабые лесополосы, границы соседних культур).
    Ориентированные фильтры усредняют ВДОЛЬ линии, поэтому ловят линии, невидимые для слияния по средним (скачок < JUMP_THR).
    Результат: скелет линий (1 px) длиной ≥ min_px. Операция локальна: не зависит от разбиения на тайлы (при достаточном HALO)."""
    R = split_line_response(feats, sens, s_along)
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
        line_b = detect_line_barriers(feats, valid, LINE_SENS, LINE_SIGMA_ALONG, LINE_MIN_PX)
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

# %% 10_refine_funcs
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
