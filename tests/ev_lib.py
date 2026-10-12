import re, time, math, sys, pickle, os, glob
import numpy as np
text = open('/tmp/work/field_boundaries_v2.py', encoding='utf-8').read()
parts = re.split(r'^# %% (\S+)(?: \[markdown\])?\n', text, flags=re.M)
cells = {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}
g = globals()
for name in ['2_params', '3_imports', '4_segmentation_funcs', '5_vector_funcs']:
    exec(cells[name], g)
import rasterio
from rasterio.crs import CRS as RCRS
TIF = glob.glob('/root/.claude/uploads/8f56ea04-574c-540f-8aea-f537cd96ba81/aa56e83c*.tif')[0]
src = rasterio.open(TIF)
T = src.transform; CRS = src.crs
H, W = src.height, src.width
lat = (src.bounds.top + src.bounds.bottom) / 2
px_area_ha = abs(T.a) * 111320 * math.cos(math.radians(lat)) * abs(T.e) * 110574 / 1e4
ARR = src.read().astype(np.float32)
FEATS = np.moveaxis(ARR[:3], 0, -1)
VALID = ARR[3] > 0
CACHE = '/tmp/claude-0/-home-claude/8f56ea04-574c-540f-8aea-f537cd96ba81/scratchpad/ev'

def load_presets(path):
    ns = {}
    exec(open(path, encoding='utf-8').read(), ns)
    g['PRESETS'] = ns['PRESETS']
    return ns['PRESETS']

def raw_polys(name, tile=2048, halo=64, tag=''):
    use_preset(name)
    p = {k: g[k] for k in BASE}; key = abs(hash(str(sorted(p.items())))) % 10**8
    f = f'{CACHE}/{name}_{tag}.pkl'
    sig = str(sorted(p.items()))
    if os.path.exists(f):
        s, raw = pickle.load(open(f, 'rb'))
        if s == sig: return raw
    t0 = time.time(); frames = []; tid = 0
    crop = np.ones((H, W), bool)
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            c1, r1 = min(c0 + tile, W), min(r0 + tile, H)
            hc0, hr0, hc1, hr1 = max(c0 - halo, 0), max(r0 - halo, 0), min(c1 + halo, W), min(r1 + halo, H)
            lab, _ = segment_regions(FEATS[hr0:hr1, hc0:hc1], VALID[hr0:hr1, hc0:hc1], crop[hr0:hr1, hc0:hc1], None)
            gd = extract_polygons(lab, FEATS[hr0:hr1, hc0:hc1], crop[hr0:hr1, hc0:hc1], (hc0, hr0), (c0, r0, c1, r1), (0, 0, W, H), tid)
            tid += 1
            if gd is not None: frames.append(gd)
    raw = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry='geometry', crs=None)
    print(f'  {name}: {time.time()-t0:.0f} c', flush=True)
    pickle.dump((sig, raw), open(f, 'wb'))
    return raw

def fields(name, yellow=None, tag=''):
    raw = raw_polys(name, tag=tag)
    use_preset(name)
    fl = finalize(merge_across_seams(raw, g['SEAM_THR'], g['MIN_CONTACT']), T, CRS, px_area_ha, g['MIN_FIELD_HA'], g['SIMPLIFY_PX'], g['MIN_WIDTH_PX'], g['FILL_HOLES_HA'])
    if yellow is not None:
        fl = fl[(fl.mean_min <= yellow)].reset_index(drop=True)
    return fl

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import affine_transform as _at
_inv = ~T
def to_px(fl):
    M = [_inv.a, _inv.b, _inv.d, _inv.e, _inv.c, _inv.f]
    return [_at(x, M) for x in fl.geometry.values]

def overlay(res, win, path, cols=2, size=7, lw=0.9):
    """res: dict name->fields; win=(x0,y0,x1,y1) px"""
    x0, y0, x1, y1 = win
    rgb = np.clip(FEATS[y0:y1, x0:x1] / 255, 0, 1)
    n = len(res); rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(size * cols, size * rows * (y1 - y0) / (x1 - x0)), squeeze=False)
    for ax in axs.ravel(): ax.set_axis_off()
    for ax, (nm, fl) in zip(axs.ravel(), res.items()):
        ax.imshow(rgb, extent=(x0, x1, y1, y0), interpolation='nearest')
        for gm in to_px(fl):
            for pl in (gm.geoms if gm.geom_type == 'MultiPolygon' else [gm]):
                xs, ys = pl.exterior.xy; ax.plot(xs, ys, color='yellow', lw=lw)
                for r in pl.interiors:
                    xs, ys = r.xy; ax.plot(xs, ys, color='cyan', lw=lw)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_title(f'{nm} ({len(fl)})', fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=60); plt.close()

from scipy import ndimage as ndi
from rasterio.features import rasterize as _rz
from skimage.morphology import disk
_sm = ndi.gaussian_filter(FEATS, (1.5, 1.5, 0))
FL = ndi.binary_opening((_sm[..., 1] < 65) & VALID, disk(4))                 # розовые+тёмные
YL = ndi.binary_opening((_sm[..., 1] > 95) & VALID, disk(4))                 # жёлтые
_grad = np.hypot(ndi.sobel(_sm.sum(-1), 0), ndi.sobel(_sm.sum(-1), 1)) / 8
# длинные барьеры-эталон (по min-каналу, top-hat)
_th = ndi.white_tophat(_sm[..., 1], footprint=disk(4))
_ref = ndi.binary_opening(_th > 28, structure=np.ones((1, 1)))
_lb, _n = ndi.label(_ref, structure=np.ones((3, 3)))
_sz = np.bincount(_lb.ravel())
BELT = np.isin(_lb, np.flatnonzero(_sz >= 80)[1:]) & VALID & ~YL & ndi.binary_dilation(FL, iterations=8)

def idmap(fl):
    return _rz([(gm, i + 1) for i, gm in enumerate(to_px(fl))], out_shape=(H, W), fill=0, dtype='int32')

def metrics(fl):
    pid = idmap(fl); cov = pid > 0
    nf = FL.sum(); ny = YL.sum()
    miss = (FL & ~cov).sum() / nf
    ycov = (YL & cov).sum() / ny
    # границы
    b = np.zeros((H, W), bool)
    b[:, :-1] |= pid[:, :-1] != pid[:, 1:]; b[:-1] |= pid[:-1] != pid[1:]
    b &= VALID
    snap = _grad[b].mean(); sharp = (_grad[b] > 10).mean()
    dist = ndi.distance_transform_edt(~b)
    leak = (BELT & cov & (dist > 3)).sum() / max(BELT.sum(), 1)
    # неоднородность внутри (std по max-каналу, пиксели ядра полигона), доля пикселей розовых/тёмных
    A = fl.geometry.values
    gv = np.array([x.convex_hull for x in A], dtype=object)
    rag = float(np.mean(shapely.length(A) / shapely.length(gv)))
    pa = np.bincount(pid.ravel(), minlength=len(fl) + 1)
    return dict(n=len(fl), miss=round(miss, 3), ycov=round(ycov, 3), snap=round(snap, 2), sharp=round(sharp, 3),
                leak=round(leak, 3), rag=round(rag, 2), small=int((fl.area_ha < 40).sum()), med_ha=round(fl.area_ha.median()))
