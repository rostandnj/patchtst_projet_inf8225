"""
src/data/sigma_lognormal.py

Extraction des 14 paramètres sigma-lognormaux de Faci et al. (2021) depuis
l'extracteur SSVn (nombre de lognormales libre, équivalent à l'extracteur
utilisé dans le papier original).

Choix de la version des paramètres :
    SSVn stocke ses paramètres dans DEUX champs :
      - `original` : toutes les lognormales détectées (taille variable, = nbLogs)
      - `correction` : version propre à 2 lognormales agoniste/antagoniste,
        obtenue par fusion/sélection des lognormales d'original.

    On utilise `correction` (2 valeurs) pour extraire (mu, sigma, D, t0)
    et obtenir une représentation physiologique agoniste/antagoniste
    cohérente. Les angles (Qs, Qe) ne sont présents que dans `original` ;
    on prend alors les 2 angles correspondant aux 2 lognormales principales
    (soit les 2 premiers, soit ceux avec les plus grandes amplitudes D
    d'original — on prend les 2 premiers par simplicité, c'est l'ordre
    canonique de l'extracteur).

    `snr` et `nbLogs` sont globaux au trait : on les utilise tels quels.
    `nbLogs` peut valoir 2, 3, 4... selon la complexité détectée -> feature
    discriminante comme dans Faci et al.

14 FEATURES (ordre canonique Faci Table 1) :
    SNR, nbLog, SNR_nbLog,
    t0, D, mu, sigma, theta_s, theta_e,    # physiologiques
    M, m, t_bar, s, Ac                      # dérivés (eqs 1-5)

Formules dérivées (Plamondon et al. 2003, Faci et al. 2021 eqs 1-5) :
    M     = t0 + exp(mu - sigma^2)                          (1) Mode
    m     = t0 + exp(mu)                                    (2) Median
    t_bar = t0 + exp(mu + sigma^2 / 2)                      (3) Time delay
    s     = exp(mu + sigma^2/2) * sqrt(exp(sigma^2) - 1)    (4) Response time
    Ac    = exp(sigma^2) - 1                                (5) Asymmetry
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# 14 noms dans l'ordre canonique Faci Table 1
PARAMETER_NAMES = [
    "SNR",
    "nbLog",
    "SNR_nbLog",
    "t0",
    "D",
    "mu",
    "sigma",
    "theta_s",
    "theta_e",
    "M",
    "m",
    "t_bar",
    "s",
    "Ac",
]
N_PARAMS = len(PARAMETER_NAMES)


@dataclass
class SigmaLognormalStroke:
    """Vecteur de 14 paramètres sigma-lognormaux pour un trait (extraction SSVn)."""
    SNR: float
    nbLog: int
    SNR_nbLog: float
    t0: float
    D: float
    mu: float
    sigma: float
    theta_s: float
    theta_e: float
    M: float
    m: float
    t_bar: float
    s: float
    Ac: float

    def to_vector(self) -> np.ndarray:
        """Retourne le vecteur de 14 features dans l'ordre canonique."""
        return np.array([getattr(self, k) for k in PARAMETER_NAMES],
                        dtype=np.float64)


def _circular_mean(angles: np.ndarray) -> float:
    """Moyenne circulaire d'angles en radians (évite la discontinuité 0/2pi)."""
    if len(angles) == 0:
        return float("nan")
    if len(angles) == 1:
        return float(angles[0])
    return math.atan2(np.sin(angles).mean(), np.cos(angles).mean())


def extract_sigma_lognormal_parameters(
    trial: dict,
) -> Optional[SigmaLognormalStroke]:
    """Extrait les 14 paramètres Faci depuis l'extraction SSVn d'un trial.

    - (mu, sigma, D, t0) : depuis SSVn.parameters.correction (2 valeurs).
    - (theta_s, theta_e) : depuis SSVn.parameters.original (Qs, Qe), moyenne
      circulaire sur les 2 premières lognormales (ordre canonique SSVn).
    - SNR, nbLog : SSVn.snr et SSVn.nbLogs (globaux au trait).

    Retourne None si l'extraction SSVn est absente/incomplète.
    """
    extracted = trial.get("extracted", {})
    ssvn_block = extracted.get("SSVn", {}).get("extractedData")
    if not ssvn_block:
        return None

    snr = ssvn_block.get("snr")
    n_logs = ssvn_block.get("nbLogs")
    if snr is None or n_logs is None:
        return None
    if not math.isfinite(float(snr)) or n_logs <= 0:
        return None

    params = ssvn_block.get("parameters", {})
    correction = params.get("correction", {})
    original = params.get("original", {})
    if not correction:
        return None

    # Paramètres physiologiques depuis correction (2 valeurs ago/anta)
    mu_vec = np.asarray(correction.get("mu", []), dtype=np.float64)
    sigma_vec = np.asarray(correction.get("sigma", []), dtype=np.float64)
    D_vec = np.asarray(correction.get("D", []), dtype=np.float64)
    t0_vec = np.asarray(correction.get("t0", []), dtype=np.float64)

    if len(mu_vec) == 0 or len(sigma_vec) == 0:
        return None

    # Angles depuis original (non présents dans correction).
    # On prend les 2 premiers angles (ordre canonique de l'extracteur).
    Qs_raw = np.asarray(original.get("Qs", []), dtype=np.float64)
    Qe_raw = np.asarray(original.get("Qe", []), dtype=np.float64)
    Qs_vec = Qs_raw[:2] if len(Qs_raw) >= 2 else Qs_raw
    Qe_vec = Qe_raw[:2] if len(Qe_raw) >= 2 else Qe_raw

    # Agrégation au niveau trait
    mu_mean = float(mu_vec.mean())
    sigma_mean = float(sigma_vec.mean())
    D_mean = float(D_vec.mean()) if len(D_vec) > 0 else float("nan")
    t0_mean = float(t0_vec.mean()) if len(t0_vec) > 0 else float("nan")
    theta_s = _circular_mean(Qs_vec) if len(Qs_vec) > 0 else float("nan")
    theta_e = _circular_mean(Qe_vec) if len(Qe_vec) > 0 else float("nan")

    # Paramètres dérivés : calculés par lognormale puis moyennés
    n_common = min(len(mu_vec), len(sigma_vec), len(t0_vec))
    if n_common == 0:
        return None

    mu_v = mu_vec[:n_common]
    sigma_v = sigma_vec[:n_common]
    t0_v = t0_vec[:n_common] if len(t0_vec) >= n_common else np.zeros(n_common)

    M_vec = t0_v + np.exp(mu_v - sigma_v**2)                                # Mode (1)
    m_vec = t0_v + np.exp(mu_v)                                             # Median (2)
    t_bar_vec = t0_v + np.exp(mu_v + sigma_v**2 / 2.0)                      # Time delay (3)
    s_vec = np.exp(mu_v + sigma_v**2 / 2.0) * np.sqrt(np.exp(sigma_v**2) - 1.0)  # (4)
    Ac_vec = np.exp(sigma_v**2) - 1.0                                       # Asymmetry (5)

    M_mean = float(M_vec.mean())
    m_mean = float(m_vec.mean())
    t_bar_mean = float(t_bar_vec.mean())
    s_mean = float(s_vec.mean())
    Ac_mean = float(Ac_vec.mean())

    snr_nblog = float(snr) / float(n_logs) if n_logs > 0 else float("nan")

    return SigmaLognormalStroke(
        SNR=float(snr),
        nbLog=int(n_logs),
        SNR_nbLog=snr_nblog,
        t0=t0_mean,
        D=D_mean,
        mu=mu_mean,
        sigma=sigma_mean,
        theta_s=theta_s,
        theta_e=theta_e,
        M=M_mean,
        m=m_mean,
        t_bar=t_bar_mean,
        s=s_mean,
        Ac=Ac_mean,
    )


def is_valid_faci(
    stroke: SigmaLognormalStroke,
    snr_min: float = 15.0,
    d_max_mm: float = 500.0,
    duration_s: Optional[float] = None,
    duration_max_s: Optional[float] = None,
) -> tuple[bool, str]:
    """Applique les critères de rejet logiciel de Faci et al. (2021, Sec. 2.3).

    Règles :
        R1. SNR < snr_min -> rejet
        R2. t0 < 0 ET mu > 0 -> rejet (erreur software)
        R3. D > d_max_mm -> rejet (amplitude non plausible)
        R4. Tout paramètre NaN -> rejet (sécurité numérique)
        R5 (optionnel). duration_s > duration_max_s -> rejet (trait aberrant)

    Returns:
        (is_valid, reason). reason == '' si valide.
    """
    if stroke.SNR < snr_min:
        return False, f"R1_SNR<{snr_min}"
    if stroke.t0 < 0.0 and stroke.mu > 0.0:
        return False, "R2_t0<0_and_mu>0"
    if not math.isfinite(stroke.D) or stroke.D > d_max_mm:
        return False, f"R3_D>{d_max_mm}mm"
    for name in PARAMETER_NAMES:
        v = getattr(stroke, name)
        if not math.isfinite(v):
            return False, f"R4_NaN_{name}"
    if duration_s is not None and duration_max_s is not None:
        if duration_s > duration_max_s:
            return False, f"R5_duration>{duration_max_s}s"
    return True, ""
