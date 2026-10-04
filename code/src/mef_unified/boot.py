from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT = BASE
FROZEN = BASE / 'vendor'
for path in (ROOT, BASE/'src', FROZEN):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
