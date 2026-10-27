# %% [markdown]
# # 00 · Обновить ноутбуки research из GitHub
#
# Скачивает **свежие версии** всех ноутбуков (01–06, конвейер v7 и композит без GEE) из репозитория
# [agri_boundaries](https://github.com/ilyaskvortsov683-wq/agri_boundaries) (ветка `claude/great-thompson-znuqoj`)
# и раскладывает их по папкам `Colab_Notebooks/Agro_terra/research/…` и `…/composite` на вашем Диске.
#
# Когда запускать: **сразу, один раз** (он положит конвейер v7 в `classical_v7`), и потом — если я написал, что ноутбуки
# обновлены. Просто «Среда выполнения → Выполнить все» (≈30 с, GPU не нужен).
# Если вы меняли ячейки ноутбука (например, параметры), старая версия сохраняется рядом как `*_old.ipynb`.
# Ваши результаты в папках `out` не затрагиваются.

# %% include _common_drive.py

# %%
import urllib.request
BRANCH = 'claude/great-thompson-znuqoj'
RAW = f'https://raw.githubusercontent.com/ilyaskvortsov683-wq/agri_boundaries/{BRANCH}/'
R = DRIVE_ROOT / 'Colab_Notebooks/Agro_terra'
FILES = {'research/notebooks/01_FTW_Global.ipynb': 'research/ftw_global',
         'research/notebooks/02_FTW_PRUE_ftw_tools.ipynb': 'research/ftw_prue',
         'research/notebooks/03_Delineate_Anything_v2.ipynb': 'research/delineate_anything',
         'research/notebooks/04_FTW_UNet_finetune.ipynb': 'research/ftw_finetune',
         'research/notebooks/05_Multidate_composites_GEE.ipynb': 'research/multidate',
         'research/notebooks/06_Autotune_pipeline_Optuna.ipynb': 'research/classical_v7',
         'dist/field_pipeline_TAT_v7.ipynb': 'research/classical_v7',
         'composite/S2_composite.ipynb': 'composite',
         'research/INSTRUCTION.pdf': 'research',
         'research/notebooks/00_Update_from_GitHub.ipynb': 'research'}


def same_code(a, b):
    """Тексты ячеек совпадают (выводы и форматирование файла не важны)."""
    try:
        return [''.join(c['source']) for c in json.loads(a)['cells']] == [''.join(c['source']) for c in json.loads(b)['cells']]
    except Exception:
        return a == b


for src, sub in FILES.items():
    dst = R / sub / Path(src).name; dst.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(RAW + src, timeout=60).read()
    if dst.exists() and dst.suffix == '.ipynb' and not same_code(dst.read_bytes(), data):
        shutil.copy(dst, dst.with_name(dst.stem + '_old.ipynb'))        # вы меняли ноутбук — старая версия сохраняется
    dst.write_bytes(data)
    print(f'{sub}/{dst.name}: {len(data) // 1024} КБ')
print('Готово. Открывайте ноутбуки из папок research/… и composite на Диске.')
