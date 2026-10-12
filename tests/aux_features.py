"""Признаки из растров подложек (RGBA, любая CRS, обычно 2 м) на сетку композита (для идеи «подложка в начале, а не на этапе 5»).
python3 tests/aux_features.py <композит.tif> <выход.npz> <подложка_A.tif> [<подложка_B.tif> ...]
Для каждой подложки (считаем на 4 м):
  L_k, G_k   — яркость и «зелёность» (G−R)/(G+R), усреднённые на сетку композита, растянуты в 0–255 (признаки регионов);
  R_k        — отклик «тонкая линия + ступенька» (ориентированные фильтры, как LINE_* в ядре), нормирован на 98-й перцентиль
               (≈1 — уверенная линия), на сетку композита берётся МАКСИМУМ по 2 м-пикселям (барьер-признак).
Совместно: P1, P2 — первые две главные компоненты всех RGB-каналов подложек (цвет без потери контраста), 0–255; BOTH_G — одна серая
(PC1 всех каналов).  Невалидные (альфа = 0) пиксели получают нули."""
import sys, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve

comp, out, bases = sys.argv[1], sys.argv[2], sys.argv[3:]
cs = rasterio.open(comp)
H, W = cs.height, cs.width
FAC = 2                                                     # 2 м → 4 м


def kernel(theta, s_across, s_along, kind):
    r = int(3 * max(s_across, s_along)) + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    u = x * np.cos(theta) + y * np.sin(theta); v = -x * np.sin(theta) + y * np.cos(theta)
    gk = np.exp(-u ** 2 / (2 * s_across ** 2) - v ** 2 / (2 * s_along ** 2))
    k = (-u / s_across ** 2 * gk) if kind == 'edge' else ((1 - u ** 2 / s_across ** 2) * gk)
    if kind == 'ridge':
        k = k - k.mean()
    return (k / np.abs(k).sum()).astype(np.float32)


def oriented(ch, kind, s_across=1.5, s_along=7, n=12):
    out = np.zeros(ch.shape, np.float32)
    for i in range(n):
        r = fftconvolve(ch, kernel(np.pi * i / n, s_across, s_along, kind), mode='same')
        np.maximum(out, np.abs(r) if kind == 'edge' else r, out=out)
    return out


def to_comp(arr, src_tf, src_crs, method):
    dst = np.zeros((H, W), np.float32)
    reproject(arr.astype(np.float32), dst, src_transform=src_tf, src_crs=src_crs, dst_transform=cs.transform, dst_crs=cs.crs,
              resampling=method, src_nodata=None, dst_nodata=0)
    return dst


def stretch(x, valid=None):
    v = x[valid] if valid is not None and valid.any() else x.ravel()
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255


res, rgb_all, valid_all = {}, [], []
for k, f in enumerate(bases):
    s = rasterio.open(f); a = s.read().astype(np.float32)
    ok = a[3] > 0
    rgb = a[:3]
    h4, w4 = (rgb.shape[1] // FAC) * FAC, (rgb.shape[2] // FAC) * FAC
    r4 = rgb[:, :h4, :w4].reshape(3, h4 // FAC, FAC, w4 // FAC, FAC).mean((2, 4))
    ok4 = ok[:h4, :w4].reshape(h4 // FAC, FAC, w4 // FAC, FAC).all((1, 3))
    tf4 = s.transform * s.transform.scale(FAC, FAC)
    for c in range(3):                                      # дыры (альфа = 0) заполняем средним по кадру: краёв они не создают
        r4[c][~ok4] = r4[c][ok4].mean()
    lum = r4[0] * .299 + r4[1] * .587 + r4[2] * .114
    grn = (r4[1] - r4[0]) / (r4[1] + r4[0] + 1.0) * 255
    resp = []
    for ch in (gaussian_filter(lum, 0.7), gaussian_filter(grn, 0.7)):
        for kind in ('ridge', 'edge'):
            r = oriented(ch, kind)
            r[~ok4] = 0
            resp.append(r / np.percentile(r[ok4], 98))
    R = np.maximum.reduce(resp)
    R[gaussian_filter(ok4.astype(np.float32), 8) < 0.99] = 0     # у края кадра фильтры дают артефакт
    # область, по которой валидность на сетку композита
    vcomp = to_comp(ok4.astype(np.float32), tf4, s.crs, Resampling.average) > 0.5
    res[f'R_{k}'] = to_comp(R, tf4, s.crs, Resampling.max) * vcomp
    res[f'L_{k}'] = stretch(to_comp(lum, tf4, s.crs, Resampling.average), vcomp) * vcomp
    res[f'G_{k}'] = stretch(to_comp(grn, tf4, s.crs, Resampling.average), vcomp) * vcomp
    res[f'V_{k}'] = vcomp
    rgb_all += [to_comp(r4[c], tf4, s.crs, Resampling.average) for c in range(3)]
    print(f, 'готово', flush=True)

X = np.stack(rgb_all, -1)                                   # (H, W, 3·nb)
V = np.all([res[f'V_{k}'] for k in range(len(bases))], 0)
Z = X[V]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
u, sv, vt = np.linalg.svd(np.cov(Z.T))
print('доля дисперсии PCA:', (sv / sv.sum()).round(3))
P = ((X - X[V].mean(0)) / (X[V].std(0) + 1e-6)) @ u
for i in range(2):
    res[f'P{i + 1}'] = stretch(P[..., i], V) * V
res['BOTH_G'] = res['P1']
np.savez_compressed(out, **{k: v.astype(np.float32) for k, v in res.items() if not k.startswith('V_')}, V=V)
print('сохранено', out, list(res))
