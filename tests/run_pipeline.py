"""Стенд: исполняет ячейки собранного dist/field_pipeline_v5.py в одном процессе вне Colab.
python3 -I tests/run_pipeline.py <композит.tif> <рабочая_папка> ['{"over":{"STAGES":{"5":0},"RUN_MODE":"full"}}']"""
import re, sys, json, shutil, os
from pathlib import Path

tif, run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OPT = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
text = (Path(__file__).resolve().parent.parent / 'dist' / 'field_pipeline_v5.py').read_text(encoding='utf-8')
text = re.sub(r'^!pip.*$', '', text, flags=re.M)
parts = re.split(r'^# %% (\S+)(?: \[markdown\])?\n', text, flags=re.M)
cells = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
import matplotlib; matplotlib.use('Agg')
g = globals()
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
    exec(compile(body, name, 'exec'), g)
    if name == '2_params':
        ov()
    if name == '14b_refine_funcs' and OPT.get('fake_basemap'):      # проверка обвязки этапа 5 без сети: шумовая «подложка»
        g['basemap_rgb'] = lambda Tb, shape: np.random.default_rng(0).integers(0, 255, (*shape, 3)).astype(np.uint8)
