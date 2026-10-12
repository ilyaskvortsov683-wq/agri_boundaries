import sys, json, random
exec(open('gt_eval.py').read())
g['BASE'].update(MIN_FIELD_HA=15.5, CROP_FRAC_MIN=0.5)
import multiprocessing as mp
SP = dict(MERGE_THR=[35, 40, 45, 50, 55], JUMP_THR=[0, 8, 12, 15, 20], BARRIER_THR=[18, 20, 22, 25, 28],
          BARRIER_MIN_PX=[30, 50, 80, 100], BARRIER_WIDTH_PX=[5, 6, 7], SP_SIZE_PX=[100, 120, 140],
          COMPACTNESS=[0.15, 0.2, 0.3], SMOOTH_SIGMA=[0.9, 1.0, 1.2], RELAX_ITERS=[0, 4, 6, 8],
          RELAX_BETA=[300, 400, 600], RELAX_SIGMA=[0.8, 1.0], SIMPLIFY_PX=[0.5, 0.8, 1.0], MIN_REGION_PX=[100, 150])
FIX = dict(BARRIER_DILATE=0, FILL_HOLES_HA=10, CROP_FRAC_MIN=0.4)
def work(item):
    nm, p = item
    g['PRESETS'] = {nm: p}
    t = time.time()
    fl = fields(nm, yellow=85, tag='rs')
    m = evaluate(fl); m['name'] = nm; m['params'] = json.dumps(p); return m
if __name__ == '__main__':
    n0, n = int(sys.argv[1]), int(sys.argv[2]); rnd = random.Random(n0)
    items = []
    for i in range(n0, n0 + n):
        p = {k: rnd.choice(v) for k, v in SP.items()}; p.update(FIX); items.append((f'R{i:03d}', p))
    out = open(f'ev/search_{n0}.jsonl', 'a')
    with mp.get_context('fork').Pool(2) as pool:
        for m in pool.imap_unordered(work, items):
            out.write(json.dumps(m) + '\n'); out.flush()
