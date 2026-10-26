"""Сборка Colab-ноутбуков из исходников в формате «percent» (ячейки размечены строками '# %%' и '# %% [markdown]').
python3 research/notebooks/make_notebooks.py  →  research/notebooks/*.ipynb
Исходник можно прогнать и как обычный скрипт (для проверок вне Colab: переменная окружения LOCAL_TEST=1)."""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def expand_includes(text):
    """Строка '# %% include _common_x.py' заменяется содержимым файла src/_common_x.py (общие ячейки для всех ноутбуков)."""
    return re.sub(r'^# %% include (\S+)\s*$', lambda m: (HERE / 'src' / m.group(1)).read_text(encoding='utf-8'),
                  text, flags=re.M)


def cells_from(src):
    text = expand_includes(src.read_text(encoding='utf-8'))
    parts = re.split(r'^# %%(.*)$', text, flags=re.M)
    cells = []
    for i in range(1, len(parts), 2):
        hdr, body = parts[i], parts[i + 1].strip('\n')
        if not body.strip():
            continue
        if '[markdown]' in hdr:
            lines = [re.sub(r'^# ?', '', l) for l in body.split('\n')]
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines)})
        else:
            body = re.sub(r'^#!(.*)$', r'!\1', body, flags=re.M)        # '#!pip …' в исходнике → '!pip …' в ноутбуке
            body = re.sub(r'^#%(.*)$', r'%\1', body, flags=re.M)        # '#%cd …' → '%cd …'
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': body})
    for c in cells:
        src_lines = c['source'].split('\n')
        c['source'] = [l + '\n' for l in src_lines[:-1]] + [src_lines[-1]]
    return cells


def build(src, out, gpu=False):
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'colab': {'provenance': [], 'toc_visible': True},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                       'language_info': {'name': 'python'}},
          'cells': cells_from(src)}
    if gpu:
        nb['metadata']['accelerator'] = 'GPU'
        nb['metadata']['colab']['gpuType'] = 'T4'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(out.name, len(nb['cells']), 'ячеек')


NOTEBOOKS = [('00_update_from_github.py', '00_Update_from_GitHub.ipynb', False),
             ('01_ftw_global.py', '01_FTW_Global.ipynb', False),
             ('02_ftw_prue.py', '02_FTW_PRUE_ftw_tools.ipynb', True),
             ('03_delineate_anything_v2.py', '03_Delineate_Anything_v2.ipynb', True),
             ('04_ftw_unet_finetune.py', '04_FTW_UNet_finetune.ipynb', True),
             ('05_multidate_edges_gee.py', '05_Multidate_composites_GEE.ipynb', False),
             ('06_autotune_optuna.py', '06_Autotune_pipeline_Optuna.ipynb', False)]

if __name__ == '__main__':
    only = sys.argv[1:]
    for s, o, gpu in NOTEBOOKS:
        if (HERE / 'src' / s).exists() and (not only or s in only):
            build(HERE / 'src' / s, HERE / o, gpu)
