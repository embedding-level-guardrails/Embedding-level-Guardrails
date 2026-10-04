"""Make the src/ package importable when pytest runs from the repository root.

The repo-wide CI installs only the root requirements.txt; tests are skipped
there when this package's own dependencies are missing."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))

collect_ignore_glob = [] if importlib.util.find_spec("yaml") else ["tests/*"]
