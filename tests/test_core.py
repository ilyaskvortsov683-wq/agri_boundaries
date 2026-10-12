"""Быстрые проверки ядра сегментации (без растра): python3 -I tests/test_core.py"""
import re, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
txt = (ROOT / 'core' / 'segmentation_core.py').read_text(encoding='utf-8')
parts = re.split(r'^# %% (\S+)\n', txt, flags=re.M)
g = globals()
for i in range(1, len(parts), 2):
    if parts[i] in ('3_imports', '4_segmentation_funcs', '5_vector_funcs'):
        exec(parts[i + 1], g)
cfg = json.loads((ROOT / 'parts' / 'cfg.json').read_text(encoding='utf-8'))
g.update(eval(cfg['BASE']))
px_area_ha = 0.0056

# 1. компактность: прямоугольник ≈ 12 (любой поворот/растяжение), «цепочка» заметно ниже
def mom_of(mask):
    yy, xx = np.nonzero(mask); xx = xx.astype(float); yy = yy.astype(float)
    return len(xx), np.array([xx.sum(), yy.sum(), (xx * xx).sum(), (yy * yy).sum(), (xx * yy).sum()])
rect = np.zeros((300, 600), bool); rect[50:250, 100:500] = True
n, m = mom_of(rect); c_rect = _compact_ratio(n, m)
assert 11 < c_rect < 13, c_rect
stretched = np.kron(rect[::2, ::2], np.ones((1, 3), bool))                      # растяжение осей
n2, m2 = mom_of(stretched); assert abs(_compact_ratio(n2, m2) - c_rect) < 1.0
net = np.zeros((600, 600), bool)
for k in range(5): net[k * 120:k * 120 + 60, :] = True                          # «гребёнка» из полос, соединённых краем
net[:, :60] = True
n3, m3 = mom_of(net); assert _compact_ratio(n3, m3) < 9, _compact_ratio(n3, m3)
print('compactness ok', round(c_rect, 2), round(_compact_ratio(n3, m3), 2))

# 2. ограничение размера в HAC: две одинаковые половины не сливаются, если сумма > max_px
lab = np.zeros((100, 200), np.int32); lab[:, :100] = 1; lab[:, 100:] = 2
feats = np.full((100, 200, 3), 100, np.float32)
out = merge_regions(lab, feats, 48, 3, 10, 0, max_px=15000)
assert len(np.unique(out[out > 0])) == 2
out = merge_regions(lab, feats, 48, 3, 10, 0, max_px=25000)
assert len(np.unique(out[out > 0])) == 1
print('size cap ok')

# 3. WC-барьер: крупный блок «застройки» вырезается из регионов, одиночные пиксели-шум игнорируются
BARRIER_THR, BARRIER_WC_CLASSES, WC_BARRIER_MIN_PX = 0, [50], 30
valid = np.ones((80, 80), bool); wc = np.full((80, 80), 40, np.uint8); wc[20:40, 20:40] = 50; wc[60, 60] = 50
b = detect_barriers(np.zeros((80, 80, 3), np.float32), valid, wc)
assert b[25, 25] and not b[60, 60]
print('wc barrier ok')
print('ВСЕ ТЕСТЫ ПРОЙДЕНЫ')
