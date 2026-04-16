"""
Configuration pytest — ajoute la racine du projet au sys.path.

Nécessaire car le .pth de l'install éditable ne se propage pas
avec les chemins Unicode sous macOS (bug connu Python/site).
"""

import sys
from pathlib import Path

# Racine du projet (parent de tests/)
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
