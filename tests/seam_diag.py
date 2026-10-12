"""Диагностика склейки швов: python3 tests/seam_diag.py <tif> <папка tiles5_...> [preset]"""
import sys, re, json, glob, numpy as np, geopandas as gpd, pandas as pd, shapely, rasterio
from pathlib import Path
from rasterio.windows import Window
ROOT = Path('/home/user/agri_boundaries')
txt = (ROOT / 'core/segmentation_core.py').read_text(encoding='utf-8')
parts = re.split(r'^# %% (\S+)\n', txt, flags=re.M); g = globals()
for i in range(1, len(parts), 2):
    if parts[i] in ('3_imports', '4_segmentation_funcs', '5_vector_funcs'): exec(parts[i + 1], g)
cfg = json.loads((ROOT / 'parts/cfg.json').read_text(encoding='utf-8')); g.update(eval(cfg['BASE']))
src = rasterio.open(sys.argv[1]); H, W, T = src.height, src.width, src.transform
px_area_ha = abs(T.a) * 111320 * np.cos(np.radians((src.bounds.top + src.bounds.bottom) / 2)) * abs(T.e) * 110574 / 1e4
def read_window(win):
    a = src.read([1, 2, 3], window=win).astype(np.float32); return np.nan_to_num(np.moveaxis(a, 0, -1)), None
src_code = (ROOT / 'parts/d_stages123.py').read_text(encoding='utf-8')
i = src_code.index('def seam_jump_fn'); j = src_code.index('_JOB = {}'); exec(src_code[i:j], g)
raw = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f) for f in sorted(Path(sys.argv[2]).glob('tile_*.parquet'))], ignore_index=True), geometry='geometry', crs=None)
gdf = raw.reset_index(drop=True); idx = np.flatnonzero(gdf.seam.values); geoms = gdf.geometry.values
qi, ti = shapely.STRtree(geoms[idx]).query(geoms[idx], predicate='intersects'); m = qi < ti
A, B = idx[qi[m]], idx[ti[m]]; tid = gdf.tile_id.values; d = tid[A] != tid[B]; A, B = A[d], B[d]
shared = shapely.length(shapely.intersection(geoms[A], geoms[B])); mean = gdf[['s0', 's1', 's2']].values / gdf.n_px.values[:, None]
dist = np.linalg.norm(mean[A] - mean[B], axis=1); jmp = seam_jump_fn(geoms[A], geoms[B])
big = shared >= 3
print(f'seam-фрагментов {len(idx)}, пар через шов {len(A)}, с контактом ≥3 px: {big.sum()}')
print('отказ по расстоянию средних (≥48):', int((big & (dist >= 48)).sum()), '| по скачку на шве (≥14):', int((big & (dist < 48) & (jmp >= 14)).sum()),
      '| проходят оба:', int((big & (dist < 48) & ~(jmp >= 14)).sum()))
ok = big & (dist < 48) & ~(jmp >= 14)
n = gdf.n_px.values[A] + gdf.n_px.values[B]
print('из прошедших: сумма > 800 га:', int((ok & (n * px_area_ha > 800)).sum()))
print('скачок на шве, квантили 10/50/90 (контакт≥3):', np.nanquantile(jmp[big], [.1, .5, .9]).round(1))
