# agri_boundaries

Конвейер для Google Colab: выделение и классификация сельхозполей Самарской области по сезонному NDVI-композиту.
Полное описание этапов, решений и открытых вопросов — в [`HANDOFF_field_pipeline.md`](HANDOFF_field_pipeline.md).

## Запуск
Открыть в Colab `dist/field_pipeline_SAM_v3.ipynb` (копия лежит на Диске: `Colab_Notebooks/Agro_terra/TAT`). Это единственная точка запуска.

## Структура
| путь | что |
|---|---|
| `core/field_boundaries_v2_core.py` | ядро из v2: импорты, сегментация, векторизация, уточнение по подложке |
| `parts/` | части v3 (параметры, ввод-вывод, этапы 1–6); `cfg.json` — BASE/пресеты |
| `build.py` | `python3 build.py` → `dist/field_pipeline_v3.py` и `dist/field_pipeline_SAM_v3.ipynb` |
| `dist/` | собранные скрипт и ноутбук (руками не править — правьте `parts/` и пересобирайте) |
| `tests/` | `make_synthetic.py` + `run_v3.py` — прогон вне Colab; `gt_eval.py` и др. — оценка по ручной цифровке (пути внутри пока под старую песочницу) |
| `legacy/` | устаревшее (`e_stage4.py`, пресеты) |

## Проверка вне Colab
```
python3 tests/make_synthetic.py syn.tif
python3 -I tests/run_v3.py syn.tif run1
```
