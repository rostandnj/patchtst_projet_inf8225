"""
src/training/device.py

Gestion du device PyTorch (MPS sur Apple Silicon, fallback CPU).

Utilitaires défensifs vis-à-vis des bugs connus de MPS qu'on a déjà
rencontrés (cf. fix dans patchtst.py:reconstruction_loss).

IMPORTANT — Activation automatique de PYTORCH_ENABLE_MPS_FALLBACK :
    PyTorch MPS n'implémente pas toutes les opérations (notamment
    `unfold_backward` utilisé pendant le backward pass de notre
    PatchEmbedding). La variable d'environnement PYTORCH_ENABLE_MPS_FALLBACK=1
    permet à PyTorch de retomber automatiquement sur CPU pour ces ops
    spécifiques, sans casser le reste qui tourne sur MPS.

    On l'active DÈS L'IMPORT DE CE MODULE, parce qu'elle doit être set
    AVANT le premier appel à torch.backends.mps. Mettre la variable plus
    tard (par ex. dans get_device) est trop tard.
"""

from __future__ import annotations

import os
import warnings


# CRITICAL : doit être set avant tout appel à torch.backends.mps
# pour avoir effet. On le fait au tout début de l'import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (import après setenv)


def get_device(prefer_mps: bool = True, force: str = None) -> torch.device:
    """Retourne le meilleur device disponible.

    Args:
        prefer_mps: si True (défaut), utilise MPS si dispo. Si False, force CPU.
        force: si fourni, force le device ('cpu', 'mps', 'cuda'). Override tout.

    Override possible aussi via la variable d'environnement KNM_DEVICE :
        KNM_DEVICE=cpu python script.py
    """
    # Variable d'environnement a la priorité sur tout le reste
    #env_force = os.environ.get("KNM_DEVICE", "").lower()
    #if env_force in ("cpu", "mps", "cuda"):
        #return torch.device(env_force)
    return torch.device("cpu")
    if force is not None:
        return torch.device(force)

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_int_to_float(t: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convertit un tenseur entier en float de manière défensive sur MPS.

    Le bug MPS sur clamp(int64) qu'on a rencontré nous incite à toujours
    convertir explicitement avant les ops mixtes int/float.
    """
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        return t.to(target_dtype)
    return t


def force_finite(t: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Remplace inf/nan par fill_value. Utile en debug uniquement."""
    return torch.where(torch.isfinite(t), t, torch.full_like(t, fill_value))


def warn_if_mps_known_issues():
    """Avertit si on est sur MPS et qu'il y a des bugs connus à surveiller.

    PYTORCH_ENABLE_MPS_FALLBACK est déjà activé au niveau module.
    """
    if torch.backends.mps.is_available():
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            warnings.warn(
                "MPS fallback should be enabled. Some PatchTST operations "
                "(unfold_backward) are not implemented on MPS and require CPU fallback."
            )
