"""Перебор вариантов подачи подложек (tests/aux_features.py → aux.npz) на тесте «Донбасс»: прогон конвейера + eval_gt.py для каждого.
python3 tests/aux_sweep.py <композит.tif> <prep_dir (wc.npy, gt_pieces.gpkg)> <aux.npz> <рабочая_папка> [имя_варианта ...]
Результат — строки таблицы в <рабочая_папка>/sweep.csv (повторный запуск пропускает уже посчитанные варианты)."""
import sys, json, re, subprocess, csv, os
from pathlib import Path

tif, prep, npz, work = [Path(a).resolve() for a in sys.argv[1:5]]
only = sys.argv[5:]
HERE = Path(__file__).resolve().parent
PRE = ["L_base"]                                                             # пресет ядра; подложка добавляется поверх
V = {
    'base': None,
    # барьеры (линии и ступеньки по подложке — в обнаружение линий до слияния и в этап 1b)
    'bar_A': dict(bar=['R_0'], bar_scale=1.0), 'bar_B': dict(bar=['R_1'], bar_scale=1.0), 'bar_AB': dict(bar=['R_0', 'R_1'], bar_scale=1.0),
    'bar_A_x07': dict(bar=['R_0'], bar_scale=0.7), 'bar_B_x07': dict(bar=['R_1'], bar_scale=0.7), 'bar_AB_x07': dict(bar=['R_0', 'R_1'], bar_scale=0.7),
    'bar_A_x14': dict(bar=['R_0'], bar_scale=1.4), 'bar_B_x14': dict(bar=['R_1'], bar_scale=1.4), 'bar_AB_x14': dict(bar=['R_0', 'R_1'], bar_scale=1.4),
    # признаки регионов (доп. каналы в SLIC/слиянии/релаксации)
    'reg_gray_A': dict(reg=['L_0'], reg_w=0.5), 'reg_gray_B': dict(reg=['L_1'], reg_w=0.5),
    'reg_gray_AB': dict(reg=['L_0', 'L_1'], reg_w=0.5), 'reg_pc1': dict(reg=['BOTH_G'], reg_w=0.5), 'reg_pc12': dict(reg=['P1', 'P2'], reg_w=0.5),
    'reg_pc12_w1': dict(reg=['P1', 'P2'], reg_w=1.0), 'reg_gray_B_w1': dict(reg=['L_1'], reg_w=1.0), 'reg_gray_A_w1': dict(reg=['L_0'], reg_w=1.0),
    # барьеры + признаки
    'bar_AB_pc12': dict(bar=['R_0', 'R_1'], bar_scale=1.0, reg=['P1', 'P2'], reg_w=0.5),
    'bar_AB_pc1': dict(bar=['R_0', 'R_1'], bar_scale=1.0, reg=['BOTH_G'], reg_w=0.5),
}
rows, done = [], set()
csvp = work / 'sweep.csv'
work.mkdir(parents=True, exist_ok=True)
if csvp.exists():
    rows = list(csv.DictReader(open(csvp))); done = {r['variant'] for r in rows}
for name, aux in V.items():
    if (only and name not in only) or name in done:
        continue
    run = work / f'run_{name}'
    over = {"RUN_MODE": "test", "TEST_SIZE": 2400, "USE_WORLDCOVER": True, "COMPARE_PRESETS": True, "COMPARE_NAMES": PRE}
    opt = {"wc_npy": str(prep / 'wc.npy'), "over": over}
    if aux:
        opt['aux'] = dict(aux, npz=str(npz))
    subprocess.run([sys.executable, '-I', str(HERE / 'run_pipeline.py'), str(tif), str(run), json.dumps(opt)],
                   stdout=open(work / f'{name}.log', 'w'), stderr=subprocess.STDOUT, check=False)
    f = run / 'out' / f'fields_final_test_{PRE[0]}.gpkg'
    if not f.exists():
        print(name, 'ОШИБКА, см.', work / f'{name}.log'); continue
    env = dict(os.environ, LAYER='fields')
    o = subprocess.run([sys.executable, '-I', str(HERE / 'eval_gt.py'), str(f), str(prep / 'gt_pieces.gpkg'), str(prep / 'gt_pieces.gpkg'), 'fields_split', str(tif)],
                       capture_output=True, text=True, env=env).stdout
    m = re.search(r'полигонов (\d+).*IoU ([\d.]+) \| IoU≥0.7 ([\d.]+) \| frag ([\d.]+) \| merged (\d+) \| bF ([\d.]+).*охват ([\d.]+)', o)
    a = re.search(r'точность ([\d.]+)', o)
    r = dict(variant=name, n=m[1], IoU=m[2], IoU07=m[3], frag=m[4], merged=m[5], bF=m[6], cover=m[7], cls_acc=a[1] if a else '')
    rows.append(r); print(r, flush=True)
    with open(csvp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(r)); w.writeheader(); w.writerows(rows)
