# %% [markdown]
# ### Google Earth Engine: подключение и функции сборки многодатных композитов (общие для ноутбуков)
# При первом запуске всплывёт окно авторизации Google — разрешите доступ для вашего проекта GEE.
#
# Что умеют функции ниже:
# - `s2_collection(aoi, start, end)` — Sentinel-2 L2A **HARMONIZED** (значения уже без смещения +1000, как ждут модели FTW),
#   облака/тени маскируются по **Cloud Score+** (`cs_cdf ≥ CS_THR`);
# - `window_median(aoi, start, end, bands)` — медиана за окно дат; пропуски (облака) заполняются медианой расширенного окна;
# - `monthly_ndvi(aoi, year, months)` — помесячные медианы NDVI (многодатный стек для классических методов);
# - `export_blocks(img, aoi, prefix, …)` — экспорт на Google Диск блоками (большой участок режется на квадраты с перекрытием);
# - `wait_tasks(tasks)` — ждать окончания экспортов и показывать прогресс.

# %%
import ee, math
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
print('GEE OK, проект', EE_PROJECT)

S2 = 'COPERNICUS/S2_SR_HARMONIZED'
CSP = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
CS_THR = 0.60          # порог Cloud Score+: 0.5 — мягче (больше данных), 0.65 — строже (меньше облаков)


def s2_collection(aoi, start, end, max_cloud=80):
    col = (ee.ImageCollection(S2).filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
           .linkCollection(ee.ImageCollection(CSP), ['cs_cdf']))
    return col.map(lambda im: im.updateMask(im.select('cs_cdf').gte(CS_THR)))


def _shift(date, days):
    return ee.Date(date).advance(days, 'day')


def window_median(aoi, start, end, bands, pad_days=30):
    """Медиана за окно [start, end); дыры от облаков заполняются медианой окна, расширенного на ±pad_days."""
    core = s2_collection(aoi, start, end).select(bands).median()
    wide = s2_collection(aoi, _shift(start, -pad_days), _shift(end, pad_days)).select(bands).median()
    return core.unmask(wide)


def monthly_ndvi(aoi, year, months):
    """Многоканальный образ: NDVI-медиана каждого месяца (каналы ndvi_MM), 0–1 → ×10000 int16."""
    out = []
    for m in months:
        start = ee.Date.fromYMD(year, m, 1); end = start.advance(1, 'month')
        nd = s2_collection(aoi, start, end).map(lambda im: im.normalizedDifference(['B8', 'B4']).rename('ndvi')).median()
        out.append(nd.rename(f'ndvi_{m:02d}'))
    return ee.Image.cat(out).multiply(10000).round().toInt16()


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def block_grid(bbox, block_deg=0.5, overlap_deg=0.02):
    """Квадраты (по долготе/широте) с перекрытием: [(id, core_bbox, export_bbox)]."""
    x0, y0, x1, y1 = bbox; out = []
    nx, ny = max(1, math.ceil((x1 - x0) / block_deg)), max(1, math.ceil((y1 - y0) / block_deg))
    for i in range(nx):
        for j in range(ny):
            cx0, cy0 = x0 + i * block_deg, y0 + j * block_deg
            core = (cx0, cy0, min(cx0 + block_deg, x1), min(cy0 + block_deg, y1))
            exp = (core[0] - overlap_deg, core[1] - overlap_deg, core[2] + overlap_deg, core[3] + overlap_deg)
            out.append((f'b{i:02d}_{j:02d}', core, exp))
    return out


def export_blocks(img, bbox, prefix, gee_folder, scale=10, block_deg=0.5, dtype='uint16', crs=None):
    """Экспорт изображения на Диск блоками; возвращает список (id, core_bbox, имя_файла, задача).
    Файлы появятся в «Мой диск/<gee_folder>/<prefix>_<id>.tif». crs по умолчанию — UTM центра участка."""
    crs = crs or f'EPSG:{utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)}'
    img = {'uint16': img.toUint16(), 'uint8': img.toUint8(), 'int16': img.toInt16(), 'float': img.toFloat()}[dtype]
    jobs = []
    for bid, core, exp in block_grid(bbox, block_deg):
        name = f'{prefix}_{bid}'
        task = ee.batch.Export.image.toDrive(image=img.clip(ee.Geometry.Rectangle(list(exp))), description=name[:100],
                                             folder=gee_folder, fileNamePrefix=name, region=ee.Geometry.Rectangle(list(exp)),
                                             scale=scale, crs=crs, maxPixels=1e13, fileFormat='GeoTIFF',
                                             fileDimensions=16384)     # один файл на блок (без нарезки на части)
        task.start(); jobs.append((bid, core, name, task))
    print(f'Запущено экспортов: {len(jobs)} (CRS {crs}); следить можно и на https://code.earthengine.google.com/tasks')
    return jobs


def wait_tasks(jobs, poll=30):
    """Ждёт завершения всех задач экспорта. Можно прервать ячейку и вернуться позже — задачи идут на серверах GEE."""
    t0 = time.time()
    while True:
        st = [j[3].status()['state'] for j in jobs]
        done = sum(s == 'COMPLETED' for s in st); bad = [j[2] for j, s in zip(jobs, st) if s in ('FAILED', 'CANCELLED')]
        print(f'\r{time.time() - t0:6.0f} с: готово {done}/{len(jobs)}' + (f', ошибки: {bad}' if bad else ''), end='')
        if done + len(bad) == len(jobs):
            print(); break
        time.sleep(poll)
    for j, s in zip(jobs, st):
        if s == 'FAILED':
            print(j[2], j[3].status().get('error_message'))


def collect_exports(jobs, gee_folder, dst_dir, timeout=900):
    """Переносит экспортированные файлы из «Мой диск/<gee_folder>» в папку проекта (Диск синхронизируется с задержкой)."""
    src_dir = DRIVE_ROOT / gee_folder; dst_dir = Path(dst_dir); dst_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); left = {j[2] for j in jobs}
    while left and time.time() - t0 < timeout:
        for name in list(left):
            for f in src_dir.glob(f'{name}*.tif'):                 # большой блок GEE может разрезать на части …-0000000000-0000000000.tif
                shutil.move(str(f), dst_dir / f.name)
            if list(dst_dir.glob(f'{name}*.tif')):
                left.discard(name)
        if left:
            time.sleep(15)
    print('Файлы в', dst_dir, ':', len(list(dst_dir.glob('*.tif'))), '| не найдено:', sorted(left))
