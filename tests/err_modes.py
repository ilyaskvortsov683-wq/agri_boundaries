"""Типы ошибок по эталонным полям: python3 tests/err_modes.py fields.gpkg gt_fields.gpkg"""
import sys, numpy as np, geopandas as gpd, shapely, warnings
warnings.filterwarnings('ignore')
fl = gpd.read_file(sys.argv[1], layer='fields').to_crs(6933); gt = gpd.read_file(sys.argv[2]).to_crs(6933)
gt = gt[gt.area / 1e4 >= 5].reset_index(drop=True); tree = shapely.STRtree(fl.geometry.values)
rows = []
for k, g in enumerate(gt.geometry.values):
    idx = tree.query(g, predicate='intersects'); ov = [(j, g.intersection(fl.geometry.values[j]).area) for j in idx]
    cov = sum(a for _, a in ov) / g.area
    best = max(ov, key=lambda t: t[1], default=(None, 0))[0]
    iou = g.intersection(fl.geometry.values[best]).area / g.union(fl.geometry.values[best]).area if best is not None else 0
    nfr = sum(1 for _, a in ov if a >= .1 * g.area)
    # сколько эталонных полей перекрывает лучший полигон
    gtree = shapely.STRtree(gt.geometry.values)
    mer = sum(1 for g2 in gt.geometry.values[gtree.query(fl.geometry.values[best])] if g2.intersection(fl.geometry.values[best]).area >= .25 * g2.area) if best is not None else 0
    rows.append((g.area / 1e4, cov, iou, nfr, mer))
import pandas as pd
d = pd.DataFrame(rows, columns=['ha', 'covg', 'iou', 'nfr', 'mer'])
def mode(r):
    if r.iou >= .7: return 'ok (IoU≥0.7)'
    if r.covg < .5: return 'не найдено (охват<50%)'
    if r.mer >= 2: return 'слито с соседним полем'
    if r.nfr >= 2: return 'раздроблено'
    return 'граница не совпала'
d['mode'] = d.apply(mode, axis=1)
d['size'] = pd.cut(d.ha, [0, 15, 50, 1e9], labels=['5–15 га', '15–50 га', '>50 га'])
print(pd.crosstab(d['size'], d['mode']))
print(d.groupby('size').iou.mean().round(3).to_dict())
