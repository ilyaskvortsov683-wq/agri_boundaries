"""Сборка ноутбука: python3 composite/build_notebook.py → composite/S2_composite.ipynb
(общая ячейка подключения Диска берётся из research/notebooks/src/_common_drive.py)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'research' / 'notebooks'))
from make_notebooks import build

if __name__ == '__main__':
    build(HERE / 'src' / 's2_composite.py', HERE / 'S2_composite.ipynb')
