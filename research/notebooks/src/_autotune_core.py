# %% [markdown]
# ### Ядро автоподбора (выполните без изменений)
# - `load_pipeline()` — исполняет ячейки скрипта конвейера v7 в этом же
#   ноутбуке: функции этапов становятся доступны, сами этапы не запускаются (включён режим сравнения пресетов).
# - `run_trial(params)` — прогоняет один набор параметров (как пресет поверх `BASE_PRESET`) по всем этапам на тестовом
#   участке и возвращает финальный слой полей.
# - `score(fields)` — метрики на двух половинах разметки: **A** (по ней подбираем) и **B** (контроль — по ней видно,
#   не «подогнались» ли параметры под конкретные поля).

# %%
import re, copy, numpy as np, pandas as pd, geopandas as gpd, shapely


def load_pipeline(py_path, overrides, ns):
    """ns — глобальное пространство имён ноутбука (globals()): так параллельные процессы тайлов видят функции конвейера."""
    text = Path(py_path).read_text(encoding='utf-8')
    text = re.sub(r'^!pip.*$', '', text, flags=re.M)
    parts = re.split(r'^# %% (\S+)(?: \[markdown\])?\n', text, flags=re.M)
    for i in range(1, len(parts), 2):
        name, body = parts[i], parts[i + 1]
        if name == '0_intro':
            continue
        if name == '18_compare':                    # нужны только функции (run_all); сам прогон пресетов не запускаем
            c = ns['COMPARE']; ns['COMPARE'] = False
            exec(compile(body, name, 'exec'), ns); ns['COMPARE'] = c
            continue
        exec(compile(body, name, 'exec'), ns)
        if name == '2_params':
            ns.update(overrides); ns.update(COMPARE_PRESETS=True, COMPARE_NAMES=[])
        if name == '9_segmentation_stage' and WC_NPY:                       # только для проверок вне Colab
            _wc = np.load(WC_NPY)
            def _wc_from_npy(t, h, w, tag, _ns=ns):
                c0, r0 = int(round((t.c - _ns['T'].c) / _ns['T'].a)), int(round((t.f - _ns['T'].f) / _ns['T'].e))
                out = np.zeros((h, w), np.uint8); ra, ca = max(r0, 0), max(c0, 0)
                sub = _wc[ra:r0 + h, ca:c0 + w]
                out[ra - r0:ra - r0 + sub.shape[0], ca - c0:ca - c0 + sub.shape[1]] = sub
                return out
            ns['get_wc_map'] = _wc_from_npy
    return ns


def run_trial(ns, params, name='T'):
    p = dict(ns['PRESETS'][BASE_PRESET]); p.update(params)
    ns['PRESETS'][name] = p
    ns['run_all'](name)
    return ns['FIELDS'].copy()


def gt_halves(gt):
    """Делит эталонные поля на западную (A) и восточную (B) половины по медиане долготы центров."""
    c = gt.to_crs(4326).representative_point().x
    m = float(np.median(c))
    return gt[c < m].reset_index(drop=True), gt[c >= m].reset_index(drop=True)


def score(fields, gtA, gtB, grid):
    kw = dict(transform=grid['transform'], shape=grid['shape'], crs=grid['crs'], type_col=GT_TYPE_COL)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        a = evaluate(fields, gtA, label='A', **kw); b = evaluate(fields, gtB, label='B', **kw)
    obj = lambda r: r['IoU'] + 0.5 * r['IoU07'] + 0.5 * r['bF'] - 0.05 * max(r['frag'] - 1.3, 0) - 0.003 * r['merged']
    return obj(a), obj(b), a, b
