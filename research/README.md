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

Значок † в отчёте — факт взят из поисковой выдержки (первоисточник был недоступен). Ничего из рецептов в Colab ещё не запускалось.
