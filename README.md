# agri_boundaries

Конвейер для Google Colab: выделение и классификация сельхозполей Самарской области по сезонному NDVI-композиту.
Полное описание этапов, решений и открытых вопросов — в [`HANDOFF_field_pipeline.md`](HANDOFF_field_pipeline.md).

## Запуск
Открыть в Colab `dist/field_pipeline_TAT_v5.ipynb` (копия лежит на Диске: `Colab_Notebooks/Agro_terra/TAT`). Это единственная точка запуска.

## Структура
| путь | что |
|---|---|
| `core/segmentation_core.py` | ядро (из v2, v5: ограничения размера/компактности, склейка швов): импорты, сегментация, векторизация, уточнение по подложке |
| `parts/` | части ноутбука (параметры, ввод-вывод, этапы 1–6, режим сравнения `g_compare.py`); `cfg.json` — BASE/пресеты |
| `build.py` | `python3 build.py` → `dist/field_pipeline_v5.py` и `dist/field_pipeline_TAT_v5.ipynb` |
| `dist/` | собранные скрипт и ноутбук (руками не править — правьте `parts/` и пересобирайте) |
| `tests/` | `test_core.py` (быстрые проверки ядра), `make_synthetic_big.py` + `eval_synth.py` (многотайловая сцена с эталоном), `run_pipeline.py` (прогон ноутбука вне Colab); `gt_eval.py` и др. — старая оценка по ручной цифровке SAM (пути внутри под старую песочницу) |
| `docs/ASSESSMENT.md` | оценка метода, причины гигантов, что изменено, что делать дальше |
| `legacy/` | устаревшее (`e_stage4.py`, пресеты) |

## Проверка вне Colab
```
python3 -I tests/test_core.py
python3 tests/make_synthetic_big.py big.tif big_gt.geojson 6000 4000 0
python3 -I tests/run_pipeline.py big.tif run1 '{"over":{"RUN_MODE":"test","COMPARE_PRESETS":true,"COMPARE_NAMES":["C_base","C_nosplit"]}}'
python3 -I tests/eval_synth.py run1/out/fields_final_test_C_base.gpkg big_gt.geojson fields
```
