# %% [markdown]
# ### Подключение Google Диска
# Выполните ячейку и разрешите доступ к Диску (всплывёт окно Google). Все результаты пишутся в папку
# `Colab_Notebooks/Agro_terra/research/…` — её можно поменять в параметрах ниже.

# %%
import os, sys, json, time, glob, shutil, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
LOCAL_TEST = bool(os.environ.get('LOCAL_TEST'))          # True только при проверке вне Colab
if not LOCAL_TEST:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive')
else:
    DRIVE_ROOT = Path(os.environ['LOCAL_DRIVE'])
print('Диск:', DRIVE_ROOT, '| файлов в корне:', len(list(DRIVE_ROOT.iterdir())))
