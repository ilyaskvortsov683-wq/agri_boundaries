"""Подготовка теста «Донбасс» (растр 2329×1781, 3 канала + размеченные поля с типом использования):
python3 tests/prep_donbas.py <композит.tif> <поля.geojson> <папка_вывода>
Скачивает WorldCover 2021 (открытый S3 ESA) на сетку растра → wc.npy; строит gt_pieces.gpkg (куски одного типа: Засеянное/Пар/Залежь)
и gt_fields.gpkg (куски, объединённые по id поля). Дальше:
  python3 -I tests/run_pipeline.py <tif> run1 '{"wc_npy":"<out>/wc.npy","over":{"RUN_MODE":"test","TEST_SIZE":2400,"USE_WORLDCOVER":true,"COMPARE_PRESETS":true}}'
  LAYER=fields python3 -I tests/eval_gt.py run1/out/fields_final_test_L_base.gpkg <out>/gt_pieces.gpkg
  python3 -I tests/eval_gt.py run1/out/fields_final_test_L_base.gpkg <out>/gt_pieces.gpkg <out>/gt_pieces.gpkg fields_split   # + точность классов"""
import sys, math, numpy as np, rasterio, geopandas as gpd, warnings
from pathlib import Path
from rasterio.warp import reproject, Resampling
warnings.filterwarnings('ignore')
tif, gj, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
s = rasterio.open(tif); b = s.bounds
lat0, lon0 = math.floor((b.bottom + b.top) / 2 / 3) * 3, math.floor((b.left + b.right) / 2 / 3) * 3
tile = f'N{lat0:02d}E{lon0:03d}'
url = f'/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif'
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
    w = rasterio.open(url); dst = np.zeros((s.height, s.width), np.uint8)
    reproject(rasterio.band(w, 1), dst, dst_transform=s.transform, dst_crs=s.crs, resampling=Resampling.nearest, src_nodata=0)
np.save(out / 'wc.npy', dst); print('WorldCover', tile, 'класс 40: %.0f %%' % (100 * (dst == 40).mean()))
g = gpd.read_file(gj); g = g.drop_duplicates(subset=['id', 'type', g.geometry.to_wkb().name if False else 'type']) if False else g
g['wkb'] = g.geometry.to_wkb(); g = g.drop_duplicates(['id', 'type', 'wkb']).drop(columns='wkb')
g['ha'] = g.to_crs(6933).area / 1e4; g.to_file(out / 'gt_pieces.gpkg', driver='GPKG')
f = g.dissolve(by='id', as_index=False); f['ha'] = f.to_crs(6933).area / 1e4; f[['id', 'geometry', 'ha']].to_file(out / 'gt_fields.gpkg', driver='GPKG')
print(len(g), 'кусков,', len(f), 'полей;', g.groupby('type').ha.sum().round(0).to_dict())
