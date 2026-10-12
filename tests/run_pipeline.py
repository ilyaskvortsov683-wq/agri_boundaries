"""Стенд: исполняет ячейки собранного dist/field_pipeline_v6.py в одном процессе вне Colab.
python3 -I tests/run_pipeline.py <композит.tif> <рабочая_папка> ['{"over":{"STAGES":{"5":0},"RUN_MODE":"full"}}']"""
import re, sys, json, shutil, os
from pathlib import Path

tif, run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OPT = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
text = (Path(__file__).resolve().parent.parent / 'dist' / 'field_pipeline_v6.py').read_text(encoding='utf-8')
text = re.sub(r'^!pip.*$', '', text, flags=re.M)
parts = re.split(r'^# %% (\S+)(?: \[markdown\])?\n', text, flags=re.M)
cells = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
import matplotlib; matplotlib.use('Agg')
g = globals()
if OPT.get('wc_npy'):                                   # WorldCover из .npy на сетке растра вместо GEE (для проверок вне Colab)
    from unittest.mock import MagicMock
    sys.modules['ee'] = MagicMock()
run.mkdir(parents=True, exist_ok=True)
shutil.copy(tif, run / 'SAM.tif')


def ov():
    g.update(DRIVE_DIR=str(run), COMPOSITE_NAME='SAM.tif', BOUNDARY_NAME=None, OUT_DIR=str(run / 'out'), LOCAL_DIR=str(run),
             COPY_LOCAL=False, RUN_MODE='full', USE_WORLDCOVER=False)
    for k, v in OPT.get('over', {}).items():
        if k in ('STAGES', 'IN_LAYERS'):
            g[k].update({(int(a) if a.isdigit() else a): b for a, b in v.items()})
        else:
            g[k] = v


for name, body in cells:
    if name == '0_intro':
        continue
    if OPT.get('aux') and name in ('9_segmentation_stage', '10b_stage1b_split'):
        body = body.replace('feats, valid = read_window(win)', 'feats, valid = read_window_aux(win)', 1)
    exec(compile(body, name, 'exec'), g)
    if name == '2_params':
        ov()
    if name == '9_segmentation_stage' and OPT.get('wc_npy'):
        _WC = np.load(OPT['wc_npy'])
        def _wc_from_npy(t, h, w, tag):
            c0, r0 = int(round((t.c - g['T'].c) / g['T'].a)), int(round((t.f - g['T'].f) / g['T'].e))
            out = np.zeros((h, w), np.uint8)
            ra, ca = max(r0, 0), max(c0, 0)
            sub = _WC[ra:r0 + h, ca:c0 + w]
            out[ra - r0:ra - r0 + sub.shape[0], ca - c0:ca - c0 + sub.shape[1]] = sub
            return out
        g['get_wc_map'] = _wc_from_npy
    if name == '5_vector_funcs' and OPT.get('barrier_npy'):            # подмена детектора линий готовой картой границ (прототип обученной модели)
        _BAR = np.load(OPT['barrier_npy']); _orig_det = g['detect_line_barriers']; _LW = {}
        def _det(feats, valid, sens, s_along, min_px, edge_sens=None):
            r0, c0 = _LW['r'], _LW['c']; h, w = valid.shape
            out = np.zeros((h, w), bool); sub = _BAR[max(r0, 0):r0 + h, max(c0, 0):c0 + w]
            out[max(r0, 0) - r0:max(r0, 0) - r0 + sub.shape[0], max(c0, 0) - c0:max(c0, 0) - c0 + sub.shape[1]] = sub
            if OPT.get('barrier_union'): out |= _orig_det(feats, valid, sens, s_along, min_px, edge_sens)
            return out
        g['detect_line_barriers'] = _det
    if name == '9_segmentation_stage' and OPT.get('barrier_npy'):
        _orig_rw = g['read_window']
        def _rw(win):
            _LW['r'], _LW['c'] = int(win.row_off), int(win.col_off); return _orig_rw(win)
        g['read_window'] = _rw
    if name == '9_segmentation_stage' and OPT.get('aux'):           # признаки из подложек: барьер-канал №3 и/или каналы признаков регионов
        _A = OPT['aux']; _Z = np.load(_A['npz'])
        _bar = np.max([_Z[k] for k in _A['bar']], 0) * float(_A.get('bar_scale', 1.0)) if _A.get('bar') else None
        _reg = [_Z[k] * float(_A.get('reg_w', 1.0)) for k in _A.get('reg', [])]
        _orig_rw2 = g['read_window']
        def _sl(arr, win):                                           # окно за пределами растра → нули
            r0, c0, h, w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
            out = np.zeros((h, w), np.float32); ra, ca = max(r0, 0), max(c0, 0)
            sub = arr[ra:r0 + h, ca:c0 + w]; out[ra - r0:ra - r0 + sub.shape[0], ca - c0:ca - c0 + sub.shape[1]] = sub
            return out
        def _rw_aux(win):
            feats, valid = _orig_rw2(win)
            ch = [feats, (_sl(_bar, win) if _bar is not None else np.zeros(valid.shape, np.float32))[..., None]]
            ch += [_sl(a, win)[..., None] for a in _reg]
            return np.concatenate(ch, -1).astype(np.float32), valid
        g['read_window_aux'] = _rw_aux
        _orig_sm, _orig_lr = g['smooth_feats'], g['_line_resp']
        def _sm(feats, sigma):                                       # канал барьера в сегментацию/слияние не идёт
            return _orig_sm(np.delete(feats, 3, -1) if feats.shape[-1] > 3 else feats, sigma)
        def _lr(feats, sens, s_along, edge_sens=None):
            r = _orig_lr(feats[..., :3], sens, s_along, edge_sens) if (sens or edge_sens) else 0
            return np.maximum(r, feats[..., 3]) if (feats.shape[-1] > 3 and _bar is not None) else r
        g['smooth_feats'], g['_line_resp'] = _sm, _lr
    if name == '14b_refine_funcs' and OPT.get('fake_basemap'):      # проверка обвязки этапа 5 без сети: шумовая «подложка»
        g['basemap_rgb'] = lambda Tb, shape: np.random.default_rng(0).integers(0, 255, (*shape, 3)).astype(np.uint8)
