"""Стенд: помесячные медианы NDVI и сезонные окна RGBN Sentinel-2 (2026) на сетке донбасского композита.
python3 fetch_don_2026.py <don.tif> <out_dir>
Фаза 1 — каждая сцена (red, green, blue, nir; 0 = нет данных/облако) пишется на диск uint16;
фаза 2 — медианы считаются полосами строк (без больших массивов в памяти)."""
import sys, os, time, json, numpy as np, rasterio
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_cogs as C

src = rasterio.open(sys.argv[1]); out = sys.argv[2]; os.makedirs(f'{out}/scenes', exist_ok=True)
T, shp, crs = src.transform, src.shape, src.crs
YEAR = 2026
scenes = [s for t in ('37TDN', '37UDP') for s in C.list_scenes(t, YEAR, range(3, 11)) if s['cloud'] < 50 and 'scl' in s['assets']]
print(len(scenes), 'сцен', flush=True)


def load(sc):
    p = f"{out}/scenes/{sc['name']}.npy"
    if os.path.exists(p):
        return sc['date'], p
    for k in range(3):
        try:
            nd, ch = C.scene_ndvi(sc, T, shp, crs, bands=('red', 'green', 'blue', 'nir'))
            a = np.stack([np.nan_to_num(ch[b], nan=0) for b in ('red', 'green', 'blue', 'nir')]).clip(0, 65535).astype(np.uint16)
            if (a[3] > 0).mean() < 0.02:
                return None
            np.save(p, a); return sc['date'], p
        except Exception as e:
            err = e; time.sleep(3)
    print('сбой', sc['name'], err, flush=True); return None


t0 = time.time(); got = []
with ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(load, scenes)):
        if r: got.append(r)
        if i % 20 == 0: print(i, round(time.time() - t0), flush=True)
got.sort(); json.dump(got, open(f'{out}/scenes.json', 'w'))
print('сцен с данными', len(got), flush=True)

WIN = {'A': ('2026-04-10', '2026-05-31'), 'B': ('2026-07-15', '2026-08-31'), 'MID': ('2026-06-01', '2026-07-15')}
months = sorted({int(d[5:7]) for d, _ in got})
H, W = shp; R = 128
mons = np.zeros((len(months), H, W), np.float16); wins = {k: np.zeros((4, H, W), np.uint16) for k in WIN}
nstd = np.zeros((H, W), np.float16); nmax = np.zeros((H, W), np.float16); dmax = np.zeros((H, W), np.int16)
mm = {p: np.load(p, mmap_mode='r') for _, p in got}
doy = np.array([int(time.strftime('%j', time.strptime(d, '%Y-%m-%d'))) for d, _ in got])
for r0 in range(0, H, R):
    r1 = min(H, r0 + R)
    blk = np.stack([mm[p][:, r0:r1] for _, p in got]).astype(np.float32)        # (n, 4, h, W)
    blk[blk == 0] = np.nan
    nd = (blk[:, 3] - blk[:, 0]) / (blk[:, 3] + blk[:, 0] + 1e-6)
    for i, m in enumerate(months):
        sel = [j for j, (d, _) in enumerate(got) if int(d[5:7]) == m]
        mons[i, r0:r1] = np.nanmedian(nd[sel], 0)
    for k, (a, b) in WIN.items():
        sel = [j for j, (d, _) in enumerate(got) if a <= d <= b]
        if sel: wins[k][:, r0:r1] = np.nan_to_num(np.nanmedian(blk[sel], 0)).clip(0, 65535).astype(np.uint16)
    nstd[r0:r1] = np.nanstd(nd, 0); nmax[r0:r1] = np.nanmax(nd, 0)
    ndz = np.where(np.isnan(nd), -9, nd); dmax[r0:r1] = doy[ndz.argmax(0)]
    print('\rстроки', r1, '/', H, end='', flush=True)
np.save(f'{out}/ndvi_monthly.npy', mons); np.save(f'{out}/months.npy', np.array(months))
for k in WIN: np.save(f'{out}/win_{k}.npy', wins[k])
np.save(f'{out}/ndvi_std.npy', nstd); np.save(f'{out}/ndvi_max.npy', nmax); np.save(f'{out}/ndvi_doymax.npy', dmax)
print('\nготово', round(time.time() - t0), 'с; месяцы', months, flush=True)
