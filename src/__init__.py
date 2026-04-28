"""
src package init.

Active automatiquement le fallback MPS pour les opérations PyTorch non
implémentées sur MPS (notamment unfold_backward utilisé par PatchEmbedding).
Cela doit être fait AVANT tout import de torch dans le code applicatif.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
