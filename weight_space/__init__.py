"""Inference-only tools for inspecting and editing released CoSTI weights.

The original CoSTI code lives in ``costi/`` (an unmodified copy of the
upstream repository). Its modules import each other as ``src.*``, so the
``costi/`` folder is put on ``sys.path`` here, once, for every tool in this
package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COSTI_ROOT = REPO_ROOT / "costi"

if str(COSTI_ROOT) not in sys.path:
    sys.path.insert(0, str(COSTI_ROOT))
