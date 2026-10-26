# %% [markdown]
# # 06 · Автоподбор параметров конвейера v7 по вашей разметке (Optuna)
#
# **Зачем.** У конвейера десятки параметров (размер суперпикселей, пороги слияния, чувствительность к линиям,
# ограничения размера/компактности, внешние границы…). Вместо ручного перебора пресетов Optuna (байесовский поиск TPE)
# сама пробует наборы параметров на тестовом участке и сравнивает результат с вашей разметкой.
#
# **Как устроена честная проверка.** Разметка делится пополам по долготе: по западной половине (**A**) параметры
# подбираются, восточная (**B**) — контроль. Если на B результат тоже вырос — параметры переносятся, а не «подогнаны».
#
# **Что нужно:** тестовый растр (фрагмент вашего композита), разметка полей на него (как для Донбасса),
# скрипт конвейера `field_pipeline_v7.py` (лежит в `research/classical_v7`).
# По желанию — внешняя карта границ (`*_ftw_prob.tif` из ноутбука 01, `*_boundary.tif` из 02/03 или
# `*_multidate_edges.tif` из 05) в `EXT_BOUNDARY_TIF`: тогда подбирается и её порог.
#
# **Время:** один пробный запуск — 1–3 мин на участке ~20×20 км (CPU). 40 проб ≈ 1–2 ч. Ячейку поиска можно
# прерывать и запускать снова — пробы копятся в файле базы Optuna на Диске и не теряются.

# %% include _common_drive.py

# %% [markdown]
# ## Шаг 1. Параметры

# %%
PIPELINE_PY = 'Colab_Notebooks/Agro_terra/research/classical_v7/field_pipeline_v7.py'
TEST_TIF = 'Colab_Notebooks/Agro_terra/test/don_composite.tif'       # тестовый композит (3 канала)
GT_PATH = 'Colab_Notebooks/Agro_terra/test/gt_pieces.gpkg'           # разметка полей на этот участок
GT_TYPE_COL = 'type'
EXT_BOUNDARY_TIF = None              # внешняя карта границ на этот участок (необязательно), напр. '…/don_ftw_prob.tif'
EE_PROJECT = 'my-project-20260614-499410'
BASE_PRESET = 'L_base'               # от какого пресета отталкиваемся
N_TRIALS = 40
OUT_DIR = 'Colab_Notebooks/Agro_terra/research/classical_v7/autotune'
STUDY = 'don_v7'                     # имя «исследования»: продолжить поиск — запустить с тем же именем
WC_NPY = None                        # только для проверок вне Colab

# %% include _common_eval.py

# %% include _autotune_core.py

# %% [markdown]
# ## Шаг 2. Загружаем конвейер и разметку
# Конвейер настраивается на тестовый растр в режиме `test` (весь растр — один тестовый участок), WorldCover — через GEE.

# %%
#!pip -q install optuna higra > /dev/null
import optuna, rasterio
OUT = DRIVE_ROOT / OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
test_tif = DRIVE_ROOT / TEST_TIF
with rasterio.open(test_tif) as s:
    GRID = dict(transform=s.transform, shape=(s.height, s.width), crs=s.crs); TEST_SIZE = max(s.width, s.height) + 10
over = dict(DRIVE_DIR=str(test_tif.parent), COMPOSITE_NAME=test_tif.name, BOUNDARY_NAME=None, OUT_DIR=str(OUT / 'runs'),
            LOCAL_DIR=str(OUT / 'local'), COPY_LOCAL=False, RUN_MODE='test', TEST_SIZE=TEST_SIZE, TEST_CENTER=(0.5, 0.5),
            USE_WORLDCOVER=True, EE_PROJECT=EE_PROJECT)
if WC_NPY:
    from unittest.mock import MagicMock; sys.modules['ee'] = MagicMock()
NS = globals()
load_pipeline(DRIVE_ROOT / PIPELINE_PY, over, NS)
gt_all = gpd.read_file(DRIVE_ROOT / GT_PATH)
GT_A, GT_B = gt_halves(gt_all)
print('конвейер загружен; эталон: A', len(GT_A), '| B', len(GT_B))

# %% [markdown]
# ## Шаг 3. Исходная точка: пресет `BASE_PRESET` как есть (и с внешними границами, если заданы)

# %%
rows = []
base_params = {} if not EXT_BOUNDARY_TIF else dict(EXT_TIF=str(DRIVE_ROOT / EXT_BOUNDARY_TIF), EXT_THR=0.3)
for nm, prm in [('base', {})] + ([('base+ext', base_params)] if EXT_BOUNDARY_TIF else []):
    sa, sb, a, b = score(run_trial(NS, prm, 'B0'), GT_A, GT_B, GRID)
    rows.append(dict(name=nm, score_A=round(sa, 3), score_B=round(sb, 3), **{f'A_{k}': v for k, v in a.items() if k != 'label'},
                     **{f'B_{k}': v for k, v in b.items() if k != 'label'}))
BASELINE = pd.DataFrame(rows); display(BASELINE[['name', 'score_A', 'score_B', 'A_IoU', 'B_IoU', 'A_bF', 'B_bF']])

# %% [markdown]
# ## Шаг 4. Поиск (Optuna TPE)
# Подбираются: размер и компактность суперпикселей, способ нарезки (SLIC или Higra), пороги слияния, чувствительность
# к линиям и ступенькам, ограничения компактности, сглаживание границ, порог внешней карты (если есть).
# Цель = IoU + 0,5·доля(IoU≥0,7) + 0,5·bF − штраф за дробление и склейки (на половине **A**).

# %%
def objective(trial):
    p = dict(SP_SIZE_PX=trial.suggest_int('SP_SIZE_PX', 50, 220, step=10),
             COMPACTNESS=trial.suggest_float('COMPACTNESS', 0.05, 0.8, log=True),
             SEG_METHOD=trial.suggest_categorical('SEG_METHOD', ['slic', 'higra']),
             MERGE_THR=trial.suggest_int('MERGE_THR', 24, 80, step=4),
             JUMP_THR=trial.suggest_int('JUMP_THR', 6, 30, step=2),
             LINE_SENS=trial.suggest_float('LINE_SENS', 0.2, 0.9),
             LINE_EDGE_SENS=trial.suggest_float('LINE_EDGE_SENS', 0.0, 0.6),
             MIN_COMPACT=trial.suggest_float('MIN_COMPACT', 4.0, 9.0),
             RELAX_BETA=trial.suggest_int('RELAX_BETA', 100, 600, step=50))
    if p['SEG_METHOD'] == 'higra':
        p['HG_HIERARCHY'] = trial.suggest_categorical('HG_HIERARCHY', ['area', 'volume', 'dynamics'])
    if EXT_BOUNDARY_TIF:
        p.update(EXT_TIF=str(DRIVE_ROOT / EXT_BOUNDARY_TIF), EXT_THR=trial.suggest_float('EXT_THR', 0.15, 0.8))
    t0 = time.time()
    sa, sb, a, b = score(run_trial(NS, p, 'T'), GT_A, GT_B, GRID)
    for k, v in {**{f'A_{k}': v for k, v in a.items()}, **{f'B_{k}': v for k, v in b.items()}}.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr('score_B', sb)
    print(f'проба {trial.number}: A {sa:.3f} | B {sb:.3f} | IoU A/B {a["IoU"]}/{b["IoU"]} | {time.time() - t0:.0f} с')
    return sa


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(study_name=STUDY, direction='maximize', storage=f'sqlite:///{OUT}/optuna.db',
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=1))
left = N_TRIALS - len(study.trials)
if left > 0:
    study.optimize(objective, n_trials=left)
print('лучшая проба:', study.best_trial.number, '| A', round(study.best_value, 3),
      '| B', round(study.best_trial.user_attrs['score_B'], 3))

# %% [markdown]
# ## Шаг 5. Итоги: лучшие наборы и проверка на половине B
# Таблица отсортирована по A; смотрите, чтобы и **B** был выше исходной точки. Лучший набор сохраняется как пресет
# (`best_preset.json`) — его можно вставить в `PRESETS` конвейера v7.

# %%
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs'))
cols = ['number', 'value', 'user_attrs_score_B', 'user_attrs_A_IoU', 'user_attrs_B_IoU', 'user_attrs_A_bF', 'user_attrs_B_bF',
        'user_attrs_B_IoU07', 'user_attrs_B_frag'] + [c for c in df.columns if c.startswith('params_')]
top = df.sort_values('value', ascending=False)[cols].head(10); display(top); top.to_csv(OUT / 'top_trials.csv', index=False)
display(BASELINE[['name', 'score_A', 'score_B', 'A_IoU', 'B_IoU', 'A_bF', 'B_bF']])
best = dict(study.best_params)
if EXT_BOUNDARY_TIF:
    best['EXT_TIF'] = EXT_BOUNDARY_TIF
json.dump({f'TUNED_{STUDY}': {**dict(NS['PRESETS'][BASE_PRESET]), **best}}, open(OUT / 'best_preset.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
print('Сохранено:', OUT / 'best_preset.json')
print(json.dumps(best, ensure_ascii=False, indent=1, default=str))

# %% [markdown]
# ## Шаг 6. Как применить
# 1. Откройте `best_preset.json`, скопируйте словарь параметров.
# 2. В ноутбуке конвейера v7 добавьте его в `PRESETS` под своим именем и сделайте `ACTIVE = '<имя>'`
#    (или включите в `COMPARE_NAMES`, чтобы сравнить с `L_base` на других участках).
# 3. Проверьте на **другом** участке (не тестовом) глазами: параметры подобраны под один участок и одну разметку.
