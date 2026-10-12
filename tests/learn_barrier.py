"""Прототип: обученный детектор границ полей. python3 tests/learn_barrier.py tif wc.npy gt_pieces.gpkg out_prefix
Признаки — сглаженные каналы (σ 0.7/2/4), градиенты, отклик ориентированных линий, локальное std, класс WorldCover.
Цель — граница между эталонными кусками (полоса 3 px) внутри зоны эталона. Обучение на левой половине кадра, проверка на правой.
Результат: <prefix>_prob.npy (вероятность границы), <prefix>_barrier.npy (тонкие линии длиной ≥ 40 px)."""
import sys, re, json, time, numpy as np, geopandas as gpd, rasterio, warnings
from pathlib import Path
from rasterio.features import rasterize
from scipy import ndimage as ndi
warnings.filterwarnings('ignore')
tif, wcp, gtp, pref = sys.argv[1:5]
ROOT = Path('/home/user/agri_boundaries')
parts = re.split(r'^# %% (\S+)\n', (ROOT / 'core/segmentation_core.py').read_text(encoding='utf-8'), flags=re.M); g = globals()
for i in range(1, len(parts), 2):
    if parts[i] in ('3_imports', '4_segmentation_funcs'): exec(parts[i + 1], g)
src = rasterio.open(tif); a = src.read().astype(np.float32); H, W = a.shape[1:]; feats = np.moveaxis(a, 0, -1); T = src.transform
valid = (a > 0).any(0); wc = np.load(wcp)
t0 = time.time()
F, names = [], []
for sg in (0.7, 2.0, 4.0):
    sm = np.stack([ndi.gaussian_filter(a[k], sg) for k in range(3)])
    for k in range(3): F.append(sm[k]); names.append(f'sm{sg}_{k}')
    if sg in (0.7, 2.0):
        for k in range(3): F.append(np.hypot(ndi.sobel(sm[k], 0), ndi.sobel(sm[k], 1)) / 8); names.append(f'grad{sg}_{k}')
for k in range(3):
    m = ndi.uniform_filter(a[k], 7); F.append(np.sqrt(np.maximum(ndi.uniform_filter(a[k] ** 2, 7) - m * m, 0))); names.append(f'std7_{k}')
s2 = np.stack([ndi.gaussian_filter(feats[..., k], 0.7) for k in range(2)], -1)
for sa in (4, 8):
    mx, mn = s2[..., 0], s2[..., 1]
    F.append(_oriented(mn, 'edge', 1.3, sa)); F.append(_oriented(mn, 'ridge', 1.3, sa)); F.append(_oriented(mx, 'edge', 1.3, sa)); names += [f'edge_mn{sa}', f'ridge_mn{sa}', f'edge_mx{sa}']
for c in (10, 30, 40, 50, 80): F.append(ndi.gaussian_filter((wc == c).astype(np.float32), 1.5)); names.append(f'wc{c}')
X = np.stack(F, -1).astype(np.float32); print('признаков', X.shape[-1], 'время %.0f c' % (time.time() - t0))
pcs = gpd.read_file(gtp).to_crs(src.crs)
ids = rasterize([(gm, i + 1) for i, gm in enumerate(pcs.geometry.values)], out_shape=(H, W), transform=T, fill=0, dtype='int32')
b = np.zeros((H, W), bool); b[:, :-1] |= ids[:, :-1] != ids[:, 1:]; b[:-1] |= ids[:-1] != ids[1:]
b &= ndi.binary_dilation(ids > 0, iterations=2)
band = ndi.binary_dilation(b, iterations=1)
dom = ndi.binary_dilation(ids > 0, iterations=3) & valid
col = np.tile(np.arange(W), (H, 1)); tr = dom & (col < W // 2); te = dom & (col >= W // 2)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(0)
def samp(mask, n):
    pos = np.flatnonzero(mask.ravel() & band.ravel()); neg = np.flatnonzero(mask.ravel() & ~band.ravel())
    return np.concatenate([rng.choice(pos, min(n, len(pos)), replace=False), rng.choice(neg, min(3 * n, len(neg)), replace=False)])
idx = samp(tr, 150000); Xf = X.reshape(-1, X.shape[-1]); yf = band.ravel()
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=48, random_state=0).fit(Xf[idx], yf[idx])
ti = samp(te, 100000); p = clf.predict_proba(Xf[ti])[:, 1]
print('hold-out (правая половина): AUC %.3f, AP %.3f (доля границ в выборке %.2f)' % (roc_auc_score(yf[ti], p), average_precision_score(yf[ti], p), yf[ti].mean()))
# ручной признак для сравнения: отклик линий (edge/ridge sa=4)
hr = Xf[ti][:, names.index('edge_mn4')] + Xf[ti][:, names.index('ridge_mn4')]
print('ручной отклик линий (sa=4): AUC %.3f, AP %.3f' % (roc_auc_score(yf[ti], hr), average_precision_score(yf[ti], hr)))
imp = sorted(zip(names, np.zeros(len(names))), key=lambda t: t[1])
prob = np.zeros(H * W, np.float32)
for s0 in range(0, H * W, 1_000_000): prob[s0:s0 + 1_000_000] = clf.predict_proba(Xf[s0:s0 + 1_000_000])[:, 1]
prob = prob.reshape(H, W); np.save(pref + '_prob.npy', prob)
cand = apply_hysteresis_threshold(np.where(valid, prob, 0), 0.35, 0.6)
lab, n = ndi.label(skeletonize(cand), structure=np.ones((3, 3))); keep = np.bincount(lab.ravel(), minlength=n + 1) >= 40; keep[0] = False
bar = keep[lab]; np.save(pref + '_barrier.npy', bar); print('барьер: %.2f %% пикселей, компонент %d' % (100 * bar.mean(), int(keep.sum())))
