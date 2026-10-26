"""Прогон исходника ноутбука вне Colab (проверка): ячейки исполняются по очереди в __main__, как в Colab.
[MOCK_EE=1] [SKIP_CELLS=4,6] LOCAL_DRIVE=<папка-«Диск»> python3 research/notebooks/run_local.py src/01_ftw_global.py '{"YEARS":[2025]}' [номер_последней_ячейки]
Переопределения из JSON применяются сразу после ячейки, где переменная впервые присваивается."""
import json as _json, re as _re, sys as _sys, os as _os
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from make_notebooks import expand_includes as _expand

_os.environ.setdefault('LOCAL_TEST', '1')
if _os.environ.get('MOCK_EE'):                       # без GEE: ee подменяется заглушкой, ячейки экспорта пропускают готовые файлы
    from unittest.mock import MagicMock as _MM; _sys.modules['ee'] = _MM()
import matplotlib as _mpl; _mpl.use('Agg')
_RL_SRC = _Path(_sys.argv[1]); _RL_OVER = _json.loads(_sys.argv[2]) if len(_sys.argv) > 2 else {}
_RL_STOP = int(_sys.argv[3]) if len(_sys.argv) > 3 else 10 ** 9
_RL_PARTS = _re.split(r'^# %%(.*)$', _expand(_RL_SRC.read_text(encoding='utf-8')), flags=_re.M)
_RL_G = globals(); _RL_G['display'] = print
_RL_DONE = set(); _RL_N = 0
for _RL_I in range(1, len(_RL_PARTS), 2):
    _RL_HDR, _RL_BODY = _RL_PARTS[_RL_I], _RL_PARTS[_RL_I + 1]
    if '[markdown]' in _RL_HDR or not _RL_BODY.strip():
        continue
    _RL_N += 1
    if _RL_N > _RL_STOP:
        break
    if _os.environ.get('SKIP_CELLS') and str(_RL_N) in _os.environ['SKIP_CELLS'].split(','):
        print(f'--- ячейка {_RL_N} пропущена'); continue
    _RL_CODE = '\n'.join(l for l in _RL_BODY.split('\n') if not l.startswith('#!') and not l.startswith('#%'))
    print(f'--- ячейка {_RL_N}', flush=True)
    exec(compile(_RL_CODE, f'{_RL_SRC.name}:cell{_RL_N}', 'exec'), _RL_G)
    for _RL_K, _RL_V in _RL_OVER.items():
        if _RL_K not in _RL_DONE and _re.search(rf'^{_re.escape(_RL_K)}\s*=', _RL_CODE, flags=_re.M):
            _RL_G[_RL_K] = _RL_V; _RL_DONE.add(_RL_K); print(f'    (переопределено {_RL_K} = {_RL_V!r})')
