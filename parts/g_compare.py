
# %% 18_compare
# Режим сравнения пресетов (COMPARE_PRESETS=True и RUN_MODE='test'): КАЖДЫЙ пресет проходит все включённые этапы; на диск пишутся
# только слой после этапа 1 (fields_after_stage1_test_<пресет>.gpkg) и финал (fields_final_test_<пресет>.gpkg: fields, fields_split, spots).
# Сегментация кэшируется по параметрам, поэтому пресеты, отличающиеся только этапами (SPLIT, NO_STAGES), считаются почти мгновенно.
COMPARE_ROWS, COMPARE_RES = [], {}


def run_all(nm):
    """Один пресет: этапы 1…6 подряд (по STAGES и перекрытиям пресета), финал → fields_final_<режим>_<пресет>.gpkg."""
    global STAGE1_PRESET, RAW1, FIELDS, TAG, SAVE_STAGES, SPLIT_OVER, NO_STAGES_CUR, CLASS_SPLIT, SPOTS_LAYER
    t0 = time.time()
    P = PRESETS[nm]
    STAGE1_PRESET, RAW1, FIELDS, TAG = nm, None, None, f'_{nm}'
    SAVE_STAGES = {1}
    SPLIT_OVER, NO_STAGES_CUR = P.get('SPLIT', {}), set(P.get('NO_STAGES', []))
    CLASS_SPLIT = SPOTS_LAYER = None
    row, s1 = dict(preset=nm), None
    for k in ORDER:
        if stage_on(k) and k in STAGE_FUNCS:
            STAGE_FUNCS[k]()
            if k == 1:
                s1 = FIELDS.copy()
                d = describe(s1)
                row.update(n_s1=d['n'], max_ha_s1=d['max_ha'], giants_s1=d['n_giant'])
    final_outputs(overview=False)
    d = describe(FIELDS)
    row.update(n=d['n'], median_ha=d['median_ha'], p90_ha=d['p90_ha'], max_ha=d['max_ha'], giants=d['n_giant'], giant_share=d['giant_share'])
    if 'cls_main_name' in FIELDS.columns:
        sh = FIELDS.groupby('cls_main_name').area_ha.sum() / FIELDS.area_ha.sum()
        row.update({f'%{c}': round(100 * float(sh.get(c, 0)), 1) for c in CLASS_NAMES.values()})
    row['min'] = round((time.time() - t0) / 60, 1)
    COMPARE_ROWS.append(row)
    COMPARE_RES[nm] = (s1, FIELDS.copy())
    return row


if COMPARE:
    names = COMPARE_NAMES or [n_ for n_ in PRESETS if not n_.startswith('OLD')]
    for nm in names:
        print(f'\n══════ пресет {nm} ══════')
        try:
            run_all(nm)
        except Exception as e_:
            import traceback; traceback.print_exc()
            COMPARE_ROWS.append(dict(preset=nm, error=str(e_)[:80]))
    table = pd.DataFrame(COMPARE_ROWS)
    print('\n', table.to_string(index=False))
    table.to_csv(Path(OUT_DIR) / f'compare_summary_{RUN_MODE}.csv', index=False)
    ok_ = {n_: v for n_, v in COMPARE_RES.items()}
    for ci, ctr in enumerate(COMPARE_CENTERS):
        win = compare_window(COMPARE_WIN_PX, ctr)
        plot_compare({n_: v[0] for n_, v in ok_.items()}, win=win, path=f'{OUT_DIR}/compare_stage1_{ci}.png')
        plot_compare({n_: v[1] for n_, v in ok_.items()}, win=win, path=f'{OUT_DIR}/compare_final_{ci}.png', color_by_class=True)
