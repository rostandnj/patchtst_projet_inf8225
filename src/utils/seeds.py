"""
src/utils/seeds.py

Fixe les seeds pour la reproductibilité.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42):
    """Fixe la seed globale (Python random, numpy, hash, et torch si dispo)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
        # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     torch.mps.manual_seed(seed)
    except ImportError:
        pass
