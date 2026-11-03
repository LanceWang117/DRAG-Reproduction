"""Build a runnable copy without editing the authors' source directory."""
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / 'src/original_drag'
CHANGES = PROJECT / 'src/reproduction_changes/model'
RUNTIME = PROJECT / '.runtime/drag'


def build_runtime():
    """Copy upstream files, then apply only the two documented Judge files."""
    if not (ORIGINAL / 'main.py').is_file():
        raise FileNotFoundError('Missing original DRAG source')
    # Generated code lives outside src/ so the original remains easy to audit.
    shutil.copytree(ORIGINAL, RUNTIME, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['drag.py', 'judge_decision.py']:
        shutil.copy2(CHANGES / name, RUNTIME / 'model' / name)
    return RUNTIME


if __name__ == '__main__':
    print('Runnable source:', build_runtime())
