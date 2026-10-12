exec(open('ev_lib.py').read())
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
GTF = glob.glob('/root/.claude/uploads/8f56ea04-574c-540f-8aea-f537cd96ba81/1effc679*.geojson')[0]
_gt = gpd.read_file(GTF).to_crs(CRS)
def _fill(gm):
    ps = list(gm.geoms) if gm.geom_type == 'MultiPolygon' else [gm]
    return MultiPolygon([Polygon(p.exterior) for p in ps])
_gt['geometry'] = [_fill(x).buffer(0) for x in _gt.geometry]            # дыры в ручной цифровке заливаем
GT_PX = to_px(_gt)
GT_ID = _rz([(gm, i + 1) for i, gm in enumerate(GT_PX)], out_shape=(H, W), fill=0, dtype='int32')
_npx = np.bincount(GT_ID.ravel(), minlength=len(_gt) + 1)[1:]
_mean = np.stack([np.bincount(GT_ID.ravel(), weights=FEATS[..., k].ravel(), minlength=len(_gt) + 1)[1:] for k in range(3)], 1) / np.maximum(_npx[:, None], 1)
_gt['ha'] = _npx * px_area_ha; _gt['mmin'] = _mean[:, 1]; _gt['mmax'] = _mean[:, 0]
_gt['cls'] = np.where(_gt.mmin <= 85, 'pd', 'y')                         # pd = розовые/тёмные, y = жёлтые
GT_PD = np.isin(GT_ID, np.flatnonzero(_gt.cls == 'pd') + 1)
GT_Y = np.isin(GT_ID, np.flatnonzero(_gt.cls == 'y') + 1)
GT_ANY = GT_ID > 0
BIG = np.flatnonzero((_gt.cls == 'pd') & (_gt.ha >= 15.5)) + 1               # поля для оценки основной стадии
print(f'GT: {len(_gt)} полей; розовых/тёмных {int((_gt.cls=="pd").sum())} (>=15.5 га: {len(BIG)}), жёлтых {int((_gt.cls=="y").sum())}')

def _bnd(idm):
    b = np.zeros(idm.shape, bool)
    b[:, :-1] |= idm[:, :-1] != idm[:, 1:]; b[:-1] |= idm[:-1] != idm[1:]
    return b
GT_B = _bnd(np.where(GT_PD, GT_ID, 0)) 
_gtb_d = ndi.distance_transform_edt(~GT_B)

def evaluate(fl):
    pid = idmap(fl); cov = pid > 0
    npp = np.bincount(pid.ravel(), minlength=len(fl) + 1)
    rec, prec, iou, frag = [], [], [], []
    ov = np.zeros((len(fl) + 1, len(_gt) + 1), np.int64)
    m = (pid > 0) & (GT_ID > 0)
    np.add.at(ov, (pid[m], GT_ID[m]), 1)
    for k in BIG:
        v = ov[1:, k]; j = int(v.argmax()) + 1 if v.max() > 0 else 0
        a = _npx[k - 1]
        if j == 0: rec.append(0); prec.append(0); iou.append(0); frag.append(0); continue
        inter = ov[j, k]; rec.append(inter / a)
        prec.append(inter / npp[j]); iou.append(inter / (a + npp[j] - inter))
        frag.append(int((v >= 0.05 * a).sum()))
    # слияния: полигон, перекрывающий >=2 эталонных поля (pd, >=15.5 га) каждое >=15%
    mer = 0
    for j in range(1, len(fl) + 1):
        c = sum(1 for k in BIG if ov[j, k] >= 0.15 * _npx[k - 1])
        mer += c >= 2
    # пиксельная точность: предсказание vs эталон pd; предсказание поверх жёлтого эталона не штрафуем
    P = cov
    tp = (P & GT_PD).sum(); fp = (P & ~GT_PD & ~GT_Y).sum(); fn = (~P & GT_PD).sum()
    # граничный F (допуск 2 px) по эталонным границам pd-полей
    pb = _bnd(np.where(cov, pid, 0))
    pbd = ndi.distance_transform_edt(~pb)
    bp = ((_gtb_d[pb] <= 2).mean()); br = ((pbd[GT_B] <= 2).mean())
    return dict(n=len(fl), IoU=round(float(np.mean(iou)), 3), recall=round(float(np.mean(rec)), 3), prec=round(float(np.mean(prec)), 3),
                IoU70=round(float((np.array(iou) >= 0.7).mean()), 3), frag=round(float(np.mean(frag)), 2), merged=int(mer),
                pxF1=round(2 * tp / (2 * tp + fp + fn), 3), pxP=round(tp / (tp + fp), 3), pxR=round(tp / (tp + fn), 3),
                bF=round(2 * bp * br / (bp + br), 3), bP=round(bp, 3), bR=round(br, 3))
