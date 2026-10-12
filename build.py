"""Сборка ноутбука Colab из частей: python3 build.py  →  dist/field_pipeline_v5.py и dist/field_pipeline_TAT_v5.ipynb.
Ячейки в исходниках размечены строками `# %% имя` (и `# %% имя [markdown]`)."""
import re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'dist'
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')

parts = re.split(r'^# %% (\S+)(?: \[markdown\])?\n', rd('core/segmentation_core.py'), flags=re.M)
core = {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}
cfg = json.loads(rd('parts/cfg.json'))


def blk(name, body):
    return f'# %% {name}\n{body.rstrip()}\n\n'


A = rd('parts/a_intro_params.py')
for k in ('BASE', 'PRESETS', 'ACTIVE', 'S2FINE'):
    A = A.replace(f'@@{k}@@', cfg[k])
i0 = A.index('# %% 2_params')                       # импорты — ВЫШЕ ячейки с параметрами (params использует os.cpu_count и т.п.)
out = A[:i0].rstrip() + '\n\n'
out += blk('3_imports', core['3_imports'] + 'from scipy.ndimage import uniform_filter\n')
out += A[i0:].rstrip() + '\n\n'
out += blk('4_segmentation_funcs', core['4_segmentation_funcs'])
out += blk('5_vector_funcs', core['5_vector_funcs'])
out += rd('parts/c_io.py').rstrip() + '\n'
d = rd('parts/d_stages123.py').rstrip() + '\n'
i = d.index('# %% 11_stage2_small_fields')
out += d[:i] + rd('parts/e_split.py').rstrip() + '\n\n' + d[i:]
out += rd('parts/e_field_pass.py').rstrip() + '\n\n'
out += blk('14b_refine_funcs', core['10_refine_funcs'])
out += rd('parts/f_stage5_run.py').rstrip() + '\n'
out += rd('parts/g_compare.py').rstrip() + '\n'

DIST.mkdir(exist_ok=True)
(DIST / 'field_pipeline_v5.py').write_text(out, encoding='utf-8')
print('field_pipeline_v5.py:', len(out.splitlines()), 'строк')

cells = []
sp = re.split(r'^# %% (\S+)( \[markdown\])?\n', out, flags=re.M)
for i in range(1, len(sp), 3):
    md, body = sp[i + 1], sp[i + 2].strip('\n')
    if md:
        s = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)})
    else:
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': body.splitlines(keepends=True)})
nb = {'cells': cells, 'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
(DIST / 'field_pipeline_TAT_v5.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('field_pipeline_TAT_v5.ipynb:', len(cells), 'ячеек')
