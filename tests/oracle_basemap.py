# синтетическая «подложка» из эталонной цифровки: каждое поле — свой цвет + текстура, фон — серо-зелёный
import geopandas as _gpd, glob as _glob
_GT = _gpd.read_file(_glob.glob('/root/.claude/uploads/8f56ea04-574c-540f-8aea-f537cd96ba81/1effc679*.geojson')[0]).to_crs(4326)
_rng = np.random.default_rng(5)
_cols = _rng.integers(60, 200, size=(len(_GT), 3))
def basemap_rgb(Tb, shape):
    h, w = shape
    gt = _GT.to_crs(WORK_CRS)
    bounds = (Tb.c, Tb.f + h * Tb.e, Tb.c + w * Tb.a, Tb.f)
    idx = gt.sindex.query(shp_box(*bounds), predicate='intersects')
    L = rasterize([(gt.geometry.values[i], int(i) + 1) for i in idx], out_shape=(h, w), transform=Tb, fill=0, dtype='int32')
    img = np.where(L[..., None] > 0, _cols[np.maximum(L - 1, 0)], np.array([110, 120, 90])).astype(np.float32)
    img += _rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
