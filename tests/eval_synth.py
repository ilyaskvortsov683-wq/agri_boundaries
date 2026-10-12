"""Оценка слоя полей по эталону синтетической сцены: python3 tests/eval_synth.py fields.gpkg gt.geojson
merged — полигонов, перекрывающих ≥2 эталонных поля (каждое ≥15 % его площади); IoU — среднее лучшее IoU по эталонным полям ≥ 15.5 га."""
import sys, numpy as np, geopandas as gpd, shapely
fl = gpd.read_file(sys.argv[1], layer=sys.argv[3] if len(sys.argv) > 3 else None).to_crs(6933)
gt = gpd.read_file(sys.argv[2]).to_crs(6933)
gt = gt[gt.area / 1e4 >= 15.5].reset_index(drop=True)
tree = shapely.STRtree(fl.geometry.values)
ious, merged_cnt, frag = [], np.zeros(len(fl), int), []
for k, g in enumerate(gt.geometry.values):
    idx = tree.query(g, predicate='intersects')
    best, nfr = 0, 0
    for j in idx:
        inter = g.intersection(fl.geometry.values[j]).area
        if inter >= .15 * g.area: merged_cnt[j] += 1
        if inter >= .05 * g.area: nfr += 1
        best = max(best, inter / g.union(fl.geometry.values[j]).area)
    ious.append(best); frag.append(nfr)
a = fl.area.values / 1e4
print(f'полей {len(fl)} | эталон {len(gt)} | IoU {np.mean(ious):.3f} | IoU≥0.7: {np.mean(np.array(ious) >= .7):.2f} | merged {int((merged_cnt >= 2).sum())} | frag {np.mean(frag):.2f} | max {a.max():.0f} га | >600 га: {(a > 600).sum()}')
