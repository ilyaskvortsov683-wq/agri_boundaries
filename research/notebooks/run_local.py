"""Прогон исходника ноутбука вне Colab (проверка): ячейки исполняются по очереди в одном процессе.
LOCAL_TEST=1 LOCAL_DRIVE=<папка-«Диск»> python3 research/notebooks/run_local.py src/01_ftw_global.py '{"YEARS":[2025]}' [номер_последней_ячейки]
Переопределения из JSON применяются сразу после ячейки, где переменная впервые присваивается."""
import json, re, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_notebooks import expand_includes

os.environ.setdefault('LOCAL_TEST', '1')
import matplotlib; matplotlib.use('Agg')
src = Path(sys.argv[1]); over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
stop = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
text = expand_includes(src.read_text(encoding='utf-8'))
parts = re.split(r'^# %%(.*)$', text, flags=re.M)
g = {'__name__': '__main__', 'display': print}
done = set(); n = 0
for i in range(1, len(parts), 2):
    hdr, body = parts[i], parts[i + 1]
    if '[markdown]' in hdr or not body.strip():
        continue
    n += 1
    if n > stop:
        break
    if os.environ.get('SKIP_CELLS') and str(n) in os.environ['SKIP_CELLS'].split(','):
        print(f'--- ячейка {n} пропущена'); continue
    code = '\n'.join(l for l in body.split('\n') if not l.startswith('#!') and not l.startswith('#%'))
    print(f'--- ячейка {n}')
    exec(compile(code, f'{src.name}:cell{n}', 'exec'), g)
    for k, v in over.items():
        if k not in done and re.search(rf'^{re.escape(k)}\s*=', code, flags=re.M):
            g[k] = v; done.add(k); print(f'    (переопределено {k} = {v!r})')
