# research — обзор готовых алгоритмов выделения полей

| Файл | Что внутри |
|---|---|
| `reports/Алгоритмы выделения полей Sentinel 2.md` / `.pdf` | итоговый отчёт: готовые методы, данные обучения, сходство с Россией, оценка трудоёмкости, рецепты для Colab, дорожная карта |
| `research_notes/Алгоритмы выделения полей Sentinel 2/dl_models_open.md` | нейросети с открытым кодом/весами (FTW/PRUE, Delineate Anything, SAM, DECODE и др.) |
| `research_notes/…/datasets.md` | размеченные датасеты (FTW, AI4Boundaries, GloCAB, NASA Harvest Ukraine, LPIS) и их сходство с РФ |
| `research_notes/…/classical_methods.md` | классические методы (CEWS, SACFEI, North, Higra/UCM, SNIC+Canny в GEE) |
| `research_notes/…/russia_cis.md` | работы по России и СНГ, ЕФИС ЗСН, коммерческие сервисы |
| `research_notes/…/colab_practicality.md` | как запускать инструменты в Colab, вход, экспорт из GEE |
| `md2pdf.py` | сборка компактного PDF из отчёта: `python3 research/md2pdf.py report.md report.pdf` |

Значок † в отчёте — факт взят из поисковой выдержки (первоисточник был недоступен).

## Ноутбуки Colab (`notebooks/`) и инструкция

**С чего начать:** `INSTRUCTION.md` (и `INSTRUCTION.pdf`) — что запускать, в каком порядке, что получится, результаты на Донбассе.

| Ноутбук | Что делает | Среда |
|---|---|---|
| `01_FTW_Global.ipynb` | вырезка готовой карты FTW (2017–2025) по участку + карта вероятности границ на сетке композита | CPU |
| `02_FTW_PRUE_ftw_tools.ipynb` | GEE: окна A/B → 8 каналов → нейросеть FTW/PRUE → полигоны и карта границ | GPU |
| `03_Delineate_Anything_v2.ipynb` | GEE RGB (или композит) → Delineate Anything v2 → полигоны и карта границ | GPU |
| `04_FTW_UNet_finetune.ipynb` | дообучение FTW/PRUE на GloCAB (Россия) + своей разметке; честный тест на отложенном участке | GPU |
| `05_Multidate_composites_GEE.ipynb` | помесячные NDVI → многодатные границы, СКО NDVI, дата максимума (на сетке композита) | CPU |
| `06_Autotune_pipeline_Optuna.ipynb` | автоподбор параметров конвейера v7 по разметке (подбор на западной половине, контроль на восточной) | CPU |

Исходники — `notebooks/src/*.py` (формат «percent», общие ячейки `_common_*.py` подставляются строкой `# %% include`).
Сборка: `python3 research/notebooks/make_notebooks.py`. Проверка вне Colab:
`MOCK_EE=1 LOCAL_DRIVE=<папка> python3 research/notebooks/run_local.py research/notebooks/src/02_ftw_prue.py '{"NAME": "…"}'`.

`methods/` — скрипты, которыми получены цифры на Донбассе: `ftw_global/ftw_global_lib.py` (чтение FTW Global с source.coop),
`multidate/` (сборка многодатных слоёв из архива Sentinel-2 на AWS и тест границ).
