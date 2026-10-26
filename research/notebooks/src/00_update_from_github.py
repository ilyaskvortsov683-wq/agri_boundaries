# %% [markdown]
# # 00 · Обновить ноутбуки research из GitHub
#
# Скачивает **свежие версии** всех ноутбуков (01–06 и конвейер v7) из репозитория
# [agri_boundaries](https://github.com/ilyaskvortsov683-wq/agri_boundaries) (ветка `claude/great-thompson-znuqoj`)
# и раскладывает их по папкам `Colab_Notebooks/Agro_terra/research/…` на вашем Диске.
#
# Когда запускать: если я написал, что ноутбуки обновлены. Просто «Среда выполнения → Выполнить все».
# Старые версии не удаляются, а переименовываются в `*_old.ipynb` (на случай, если вы в них что-то меняли).
# Ваши результаты в папках `out` не затрагиваются.

# %% include _common_drive.py

# %%
import urllib.request
BRANCH = 'claude/great-thompson-znuqoj'
RAW = f'https://raw.githubusercontent.com/ilyaskvortsov683-wq/agri_boundaries/{BRANCH}/'
R = DRIVE_ROOT / 'Colab_Notebooks/Agro_terra/research'
FILES = {'research/notebooks/01_FTW_Global.ipynb': 'ftw_global',
         'research/notebooks/02_FTW_PRUE_ftw_tools.ipynb': 'ftw_prue',
         'research/notebooks/03_Delineate_Anything_v2.ipynb': 'delineate_anything',
         'research/notebooks/04_FTW_UNet_finetune.ipynb': 'ftw_finetune',
         'research/notebooks/05_Multidate_composites_GEE.ipynb': 'multidate',
         'research/notebooks/06_Autotune_pipeline_Optuna.ipynb': 'classical_v7',
         'dist/field_pipeline_TAT_v7.ipynb': 'classical_v7',
         'research/INSTRUCTION.pdf': '.',
         'research/notebooks/00_Update_from_GitHub.ipynb': '.'}
for src, sub in FILES.items():
    dst = R / sub / Path(src).name; dst.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(RAW + src, timeout=60).read()
    if dst.exists() and dst.read_bytes() != data and dst.suffix == '.ipynb':
        shutil.copy(dst, dst.with_name(dst.stem + '_old.ipynb'))
    dst.write_bytes(data)
    print(f'{sub}/{dst.name}: {len(data) // 1024} КБ')
print('Готово. Открывайте ноутбуки из папок research/… на Диске.')
