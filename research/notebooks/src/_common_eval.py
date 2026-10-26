# %% [markdown]
# ### Функции оценки и картинок (общие для всех ноутбуков)
# Ничего менять не нужно — просто выполните ячейку. Метрики те же, что в тестах конвейера
# (`tests/eval_gt.py` в репозитории), поэтому числа можно сравнивать с v6:
# - **IoU** — для каждого эталонного поля (≥ 5 га) берётся лучший по перекрытию полигон, среднее по полям;
# - **IoU≥0.7** — доля эталонных полей, найденных «хорошо»;
# - **frag** — сколько полигонов в среднем покрывают одно эталонное поле (1 — идеально, >1 — дробление);
# - **merged** — сколько полигонов «склеили» 2 и больше эталонных полей;
# - **bF** — точность границ (F-мера, допуск 2 пикселя), **P/R** — точность/полнота линий;
# - **охват** — доля площади эталона, покрытая полигонами; по классам (если в эталоне есть колонка `type`).

# %%
import numpy as np, geopandas as gpd, shapely, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi


def grid_for(gdf, res=10.0):
    """Сетка 10 м в UTM по охвату слоя (если нет вашего композита)."""
    utm = gdf.estimate_utm_crs()
    x0, y0, x1, y1 = gdf.to_crs(utm).total_bounds
    w, h = int(np.ceil((x1 - x0) / res)) + 2, int(np.ceil((y1 - y0) / res)) + 2
    return from_origin(x0 - res, y1 + res, res, res), (h, w), utm


def evaluate(pred, gt, transform=None, shape=None, crs=None, min_ha=5, type_col='type', label=''):
    """pred, gt — GeoDataFrame полигонов (любая СК). transform/shape/crs — сетка для растровых метрик
    (по умолчанию 10 м UTM по эталону). Печатает строку метрик и возвращает dict."""
    if transform is None:
        transform, shape, crs = grid_for(gt)
    H, W = shape
    gt = gt[gt.geometry.notna()].copy(); pred = pred[pred.geometry.notna()].copy()
    gt['geometry'] = shapely.make_valid(gt.geometry.values); pred['geometry'] = shapely.make_valid(pred.geometry.values)
    ea = 6933                                                  # равновеликая СК для площадей
    g6 = gt.to_crs(ea); g6 = g6[g6.area / 1e4 >= min_ha].reset_index(drop=True)
    p6 = pred.to_crs(ea); p6['geometry'] = shapely.make_valid(p6.geometry.values)
    tree = shapely.STRtree(p6.geometry.values)
    ious, frag, merged = [], [], np.zeros(len(p6), int)
    for g in g6.geometry.values:
        best, nfr = 0.0, 0
        for j in tree.query(g, predicate='intersects'):
            pj = p6.geometry.values[j]; inter = g.intersection(pj).area
            if inter >= .15 * g.area: merged[j] += 1
            if inter >= .05 * g.area: nfr += 1
            best = max(best, inter / g.union(pj).area)
        ious.append(best); frag.append(nfr)
    ious = np.array(ious)
    idmap = lambda d: rasterize([(g, i + 1) for i, g in enumerate(d.to_crs(crs).geometry.values)],
                                out_shape=(H, W), transform=transform, fill=0, dtype='int32')
    GT, P = idmap(g6), idmap(p6)
    def bnd(m):
        b = np.zeros(m.shape, bool); b[:, :-1] |= m[:, :-1] != m[:, 1:]; b[:-1] |= m[:-1] != m[1:]; return b
    gb, pb = bnd(GT) & (GT > 0), bnd(P) & (P > 0)
    dgt, dpb = ndi.distance_transform_edt(~gb), ndi.distance_transform_edt(~pb)
    dom = ndi.binary_dilation(GT > 0, iterations=3)
    prec = (dgt[pb & dom] <= 2).mean() if (pb & dom).any() else 0.0
    rec = (dpb[gb] <= 2).mean() if gb.any() else 0.0
    bF = 2 * prec * rec / (prec + rec + 1e-9)
    cov = ((P > 0) & (GT > 0)).sum() / max((GT > 0).sum(), 1)
    res = dict(label=label, gt=len(g6), pred=len(pred), IoU=round(float(ious.mean()), 3) if len(ious) else 0,
               IoU07=round(float(np.mean(ious >= .7)), 2) if len(ious) else 0, frag=round(float(np.mean(frag)), 2),
               merged=int((merged >= 2).sum()), bF=round(float(bF), 3), P=round(float(prec), 2), R=round(float(rec), 2),
               cover=round(float(cov), 2))
    if type_col in gt.columns:
        gtt = gt.to_crs(crs)
        for t in sorted(gtt[type_col].dropna().unique()):
            m = rasterize([(g, 1) for g in gtt[gtt[type_col] == t].geometry.values], out_shape=(H, W),
                          transform=transform, fill=0, dtype='uint8') > 0
            res[f'cover_{t}'] = round(float((P[m] > 0).mean()), 2) if m.any() else None
    print(' | '.join(f'{k} {v}' for k, v in res.items()))
    return res


def show_overlay(img_path, layers, title='', bbox=None, out_png=None, bands=(1, 2, 3), figsize=(11, 11)):
    """Картинка: подложка — ваш растр (3 канала, растяжка 2–98 %), поверх контуры слоёв.
    layers — список (GeoDataFrame, цвет, подпись)."""
    import matplotlib.pyplot as plt
    from rasterio.windows import from_bounds
    with rasterio.open(img_path) as s:
        win = from_bounds(*bbox, s.transform) if bbox is not None else None
        a = s.read(list(bands), window=win, boundless=True).astype(float)
        t = s.window_transform(win) if win is not None else s.transform; crs = s.crs
    for i in range(a.shape[0]):
        lo, hi = np.nanpercentile(a[i][a[i] > 0], [2, 98]) if (a[i] > 0).any() else (0, 1)
        a[i] = np.clip((a[i] - lo) / (hi - lo + 1e-9), 0, 1)
    ext = (t.c, t.c + t.a * a.shape[2], t.f + t.e * a.shape[1], t.f)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.moveaxis(a, 0, -1), extent=ext)
    for gdf, color, name in layers:
        if gdf is not None and len(gdf):
            gdf.to_crs(crs).boundary.plot(ax=ax, color=color, linewidth=.8, label=name)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_title(title); ax.legend(loc='upper right')
    if out_png:
        fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.show()
