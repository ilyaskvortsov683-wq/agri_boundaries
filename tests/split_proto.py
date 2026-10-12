exec(open('gt_eval.py').read())
import json, sys
from scipy.signal import fftconvolve
from scipy.ndimage import (gaussian_filter, binary_erosion, binary_dilation, label as nlabel, distance_transform_edt)
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold
from rasterio.features import shapes as _shapes
from shapely.geometry import shape as shp_shape
import shapely

_K = {}
def kern(theta, s_c, s_a, kind):
    key = (round(theta, 4), s_c, s_a, kind)
    if key in _K: return _K[key]
    r = int(3 * max(s_c, s_a)) + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    u = x * np.cos(theta) + y * np.sin(theta); v = -x * np.sin(theta) + y * np.cos(theta)
    gk = np.exp(-u ** 2 / (2 * s_c ** 2) - v ** 2 / (2 * s_a ** 2))
    k = (-u / s_c ** 2 * gk) if kind == 'edge' else ((1 - u ** 2 / s_c ** 2) * gk)
    if kind == 'ridge': k = k - k.mean()
    _K[key] = (k / np.abs(k).sum()).astype(np.float32)
    return _K[key]

def oriented(ch, kind, s_c, s_a, n=12):
    out = np.zeros(ch.shape, np.float32)
    for i in range(n):
        r = fftconvolve(ch, kern(np.pi * i / n, s_c, s_a, kind), mode='same')
        out = np.maximum(out, np.abs(r) if kind == 'edge' else r)
    return out

def line_response(feats, P):
    s = gaussian_filter(feats, (0.7, 0.7, 0))
    mx, mn = s[..., 0], s[..., 1]
    R = np.maximum.reduce([oriented(mn, 'edge', P['SC'], P['SA']) / P['E_MIN'],
                           oriented(mn, 'ridge', P['SC'], P['SA']) / P['R_MIN'],
                           oriented(mx, 'edge', P['SC'], P['SA']) / P['E_MAX']])
    return R

def split_mask(feats, fm, P, dbg=None):
    """возвращает массив меток кусков (0 вне поля) или None"""
    R = line_response(feats, P)
    core = binary_erosion(fm, iterations=P['EDGE_PX'])
    cand = apply_hysteresis_threshold(np.where(core, R, 0), P['HYS_LO'], 1.0)
    sk = skeletonize(cand)
    lab, n = nlabel(sk, structure=np.ones((3, 3)))
    if n == 0: return None
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sz >= P['LINE_MIN_PX']); keep = keep[keep > 0]
    lines = np.isin(lab, keep)
    if dbg is not None: dbg['lines'] = lines; dbg['R'] = R
    if not lines.any(): return None
    lines_d = binary_dilation(lines, iterations=P['LINE_DIL'])
    pieces, m = nlabel(fm & ~lines_d, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if m < 2: return None
    ps = np.bincount(pieces.ravel(), minlength=m + 1)
    big = np.flatnonzero(ps * px_area_ha >= P['PIECE_MIN_HA']); big = big[big > 0]
    if len(big) < 2: return None
    seeds = np.where(np.isin(pieces, big), pieces, 0)
    idx = distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    out = np.where(fm, seeds[idx[0], idx[1]], 0)
    return out

def split_layer(f, P):
    rows = []; nsplit = 0
    for i, gm in enumerate(f.geometry.values):
        k = f.kind.values[i]
        ha = f.area_ha.values[i] if 'area_ha' in f.columns else gm.area
        if k not in ('main', 'input') or ha < P['MIN_HA']:
            rows.append((gm, k, i)); continue
        minx, miny, maxx, maxy = to_px(gpd.GeoDataFrame(geometry=[gm], crs=CRS))[0].bounds
        c0, r0 = max(int(minx) - 4, 0), max(int(miny) - 4, 0); c1, r1 = min(int(maxx) + 5, W), min(int(maxy) + 5, H)
        wt = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), T)
        fm = rasterize([(gm, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt, fill=0, dtype='uint8').astype(bool)
        fe = FEATS[r0:r1, c0:c1]
        lab = split_mask(fe, fm, P)
        if lab is None:
            rows.append((gm, k, i)); continue
        nsplit += 1
        for g_, v in _shapes(lab.astype(np.int16), mask=lab > 0, transform=wt, connectivity=4):
            p_ = shp_shape(g_).intersection(gm)
            for q_ in shapely.get_parts(p_):
                if q_.geom_type == 'Polygon' and not q_.is_empty: rows.append((q_, k, i))
    out = gpd.GeoDataFrame({'kind': [r[1] for r in rows], 'src': [r[2] for r in rows]}, geometry=[r[0] for r in rows], crs=CRS)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out['area_ha'] = [x.area for x in to_px(out)] 
    out['area_ha'] = [pxa * px_area_ha for pxa in out.area_ha]
    return out, nsplit

P0 = dict(SC=1.3, SA=4, E_MIN=9, R_MIN=4.5, E_MAX=13, HYS_LO=0.6, EDGE_PX=2, LINE_MIN_PX=30, LINE_DIL=1, PIECE_MIN_HA=8, MIN_HA=40)
if __name__ == '__main__':
    f = gpd.read_file('v3run/out_F1/fields_after_stage1_full.gpkg')
    f['area_ha'] = [x.area*px_area_ha for x in to_px(f)]
    print('baseline', evaluate(f))
    ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    P = {**P0, **ov}
    t = time.time(); out, ns = split_layer(f, P)
    print('split', ns, 'polygons ->', len(out), 'time', round(time.time() - t, 1))
    print(evaluate(out))
    out.to_file('split_out.gpkg', driver='GPKG')
