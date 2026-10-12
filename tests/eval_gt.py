"""Оценка слоя полей по размеченным полям (тест Донбасс): python3 tests/eval_gt.py fields.gpkg gt_fields.gpkg [gt_pieces.gpkg] [layer_split]
Поля: IoU лучшего полигона к эталонному полю (≥ MIN_HA), доля IoU≥0.7, frag, merged, граница F (допуск 2 px), охват эталона.
Классы (если задан gt_pieces и слой разбивки fields_split): пиксельная точность в пересечении эталона и предсказания."""
import sys, numpy as np, geopandas as gpd, shapely, rasterio, glob, warnings
from rasterio.features import rasterize
from scipy import ndimage as ndi
warnings.filterwarnings('ignore')
MIN_HA = 5
import os
fl = gpd.read_file(sys.argv[1], layer=os.environ.get('LAYER', 'fields') if sys.argv[1].endswith('.gpkg') else None)
gt = gpd.read_file(sys.argv[2])
tif = glob.glob('/root/.claude/uploads/*/*2.tif')[0] if len(sys.argv) < 6 else sys.argv[5]
src = rasterio.open(tif); T = src.transform; H, W = src.height, src.width
gt['geometry'] = shapely.make_valid(gt.geometry.values); gt = gt.to_crs(6933); gt['geometry'] = shapely.make_valid(gt.geometry.values); gt = gt[gt.area / 1e4 >= MIN_HA].reset_index(drop=True); gt_ll = gt.to_crs(4326)
fl['geometry'] = shapely.make_valid(fl.geometry.values); fl6 = fl.to_crs(6933); fl6['geometry'] = shapely.make_valid(fl6.geometry.values)
tree = shapely.STRtree(fl6.geometry.values)
ious, frag, merged = [], [], np.zeros(len(fl6), int)
for g in gt.geometry.values:
    idx = tree.query(g, predicate='intersects'); best = 0; nfr = 0
    for j in idx:
        inter = g.intersection(fl6.geometry.values[j]).area
        if inter >= .15 * g.area: merged[j] += 1
        if inter >= .05 * g.area: nfr += 1
        best = max(best, inter / g.union(fl6.geometry.values[j]).area)
    ious.append(best); frag.append(nfr)
ious = np.array(ious)
# граница
def idmap(gdf):
    return rasterize([(g, i + 1) for i, g in enumerate(gdf.geometry.values)], out_shape=(H, W), transform=T, fill=0, dtype='int32')
GT = idmap(gt_ll); P = idmap(fl.to_crs(4326))
def bnd(m):
    b = np.zeros(m.shape, bool); b[:, :-1] |= m[:, :-1] != m[:, 1:]; b[:-1] |= m[:-1] != m[1:]; return b
gb, pb = bnd(GT) & (GT > 0), bnd(P) & (P > 0)
dgt, dpb = ndi.distance_transform_edt(~gb), ndi.distance_transform_edt(~pb)
dom = ndi.binary_dilation(GT > 0, iterations=3)
prec = (dgt[pb & dom] <= 2).mean() if (pb & dom).any() else 0; rec = (dpb[gb] <= 2).mean()
bF = 2 * prec * rec / (prec + rec + 1e-9)
cov = ((P > 0) & (GT > 0)).sum() / max((GT > 0).sum(), 1)
npoly_in = len({int(v) for v in np.unique(P[GT > 0]) if v})
print(f'GT полей {len(gt)} | полигонов {len(fl)} (в зоне эталона {npoly_in}) | IoU {ious.mean():.3f} | IoU≥0.7 {np.mean(ious >= .7):.2f} | frag {np.mean(frag):.2f} | merged {int((merged >= 2).sum())} | bF {bF:.3f} (P {prec:.2f} R {rec:.2f}) | охват {cov:.2f}')
if len(sys.argv) > 3:
    pcs = gpd.read_file(sys.argv[3]).to_crs(4326)
    cmap = {'Засеянное': 1, 'Пар': 2, 'Залежь': 3}; inv = {'засеянное': 1, 'пар': 2, 'неиспользуемое': 3}
    G = rasterize([(g, cmap[t]) for g, t in zip(pcs.geometry.values, pcs['type'].values)], out_shape=(H, W), transform=T, fill=0, dtype='uint8')
    sp = gpd.read_file(sys.argv[1], layer=sys.argv[4] if len(sys.argv) > 4 else 'fields_split').to_crs(4326)
    Pc = rasterize([(g, inv.get(n, 0)) for g, n in zip(sp.geometry.values, sp['cls_name'].values)], out_shape=(H, W), transform=T, fill=0, dtype='uint8')
    m = (G > 0) & (Pc > 0)
    names = ['засеянное', 'пар', 'неиспользуемое']
    conf = np.zeros((3, 3), int)
    for a in (1, 2, 3):
        for b in (1, 2, 3): conf[a - 1, b - 1] = ((G == a) & (Pc == b) & m).sum()
    acc = np.trace(conf) / conf.sum()
    print(f'классы: пиксельная точность {acc:.3f}; строки — эталон, столбцы — предсказание {names}')
    print(conf)
