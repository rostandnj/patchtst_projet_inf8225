"""
src/models/patchtst_official_wrapper.py

WRAPPER pour utiliser le PatchTST officiel (yuqinie98/PatchTST) en
classification, plutôt qu'en forecasting (sa tâche native).

PROBLÈME :
    Le PatchTST_backbone officiel produit en sortie un tenseur
    (B, n_vars, target_window) destiné au forecasting. Pour faire de la
    classification, on doit récupérer les FEATURES intermédiaires (avant
    la tête forecasting) et y attacher notre propre tête classification.

APPROCHE :
    1. Instancier PatchTST_backbone avec target_window=1 (peu importe la
       valeur, on n'utilisera pas la sortie forecasting)
    2. Hooker la sortie de leur self.backbone (TSTiEncoder) qui produit
       les features (B, n_vars, d_model, patch_num)
    3. Y attacher notre propre tête classification

PARAMÉTRAGE PAR DÉFAUT (pour correspondre à notre setup Exp A) :
    n_channels=14, seq_len=20, patch_len=4, stride=2, n_layers=2,
    d_model=64, n_heads=4, d_ff=128, dropout=0.2, attn_dropout=0.1
    revin=True, affine=True, subtract_last=False, individual=False

INIT FIX :
    Comme dans notre PatchTSTClassifier maison, on applique init_head_zero
    par défaut pour éviter le mode collapse sur petit dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Imports du repo officiel (à copier manuellement dans patchtst_official/)
from src.models.patchtst_official.PatchTST_backbone import PatchTST_backbone


@dataclass
class OfficialPatchTSTConfig:
    """Hyperparamètres du PatchTST officiel pour notre cas Exp A."""
    # Données
    n_channels: int = 14            # c_in
    seq_len: int = 20               # context_window
    patch_len: int = 4
    stride: int = 2

    # Modèle
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.2
    attn_dropout: float = 0.1
    fc_dropout: float = 0.05
    head_dropout: float = 0.0

    # RevIN
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False

    # Architecture
    individual: bool = False        # tête shared vs per-channel
    padding_patch: str = "end"      # padding strategy
    pre_norm: bool = False
    res_attention: bool = True
    pe: str = "zeros"
    learn_pe: bool = True


def _patch_num(seq_len: int, patch_len: int, stride: int, padding_patch: str = "end") -> int:
    """Calcule le nombre de patches selon la formule officielle."""
    n = (seq_len - patch_len) // stride + 1
    if padding_patch == "end":
        n += 1
    return n


class OfficialPatchTSTClassifier(nn.Module):
    """PatchTST officiel + tête classification.

    Forward:
        x : (B, n_channels, seq_len)
        return : (B, n_classes)

    Le backbone officiel est instancié comme s'il faisait du forecasting
    (target_window=1) mais on remplace sa tête par la nôtre.
    """

    def __init__(
        self,
        cfg: OfficialPatchTSTConfig,
        n_classes: int = 2,
        init_head_zero: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        # Instancie le backbone officiel
        # On met target_window=1 (forecasting horizon arbitraire, on ignore la sortie)
        self.official_backbone = PatchTST_backbone(
            c_in=cfg.n_channels,
            context_window=cfg.seq_len,
            target_window=1,           # ignoré, on remplace la tête
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            fc_dropout=cfg.fc_dropout,
            head_dropout=cfg.head_dropout,
            individual=cfg.individual,
            revin=cfg.revin,
            affine=cfg.affine,
            subtract_last=cfg.subtract_last,
            padding_patch=cfg.padding_patch,
            pre_norm=cfg.pre_norm,
            res_attention=cfg.res_attention,
            pe=cfg.pe,
            learn_pe=cfg.learn_pe,
        )

        # Remplace leur tête par une tête identité pour qu'on récupère les features
        # Leur tête prend (B, n_vars, d_model, patch_num) et produit (B, n_vars, target_window)
        # Nous, on veut récupérer (B, n_vars, d_model, patch_num) directement
        self.official_backbone.head = nn.Identity()

        # Notre tête classification :
        # 1. global pool sur patch_num : (B, n_vars, d_model, patch_num) -> (B, n_vars, d_model)
        # 2. flatten n_vars × d_model : -> (B, n_vars * d_model)
        # 3. dropout + linear : -> (B, n_classes)
        head_in_dim = cfg.n_channels * cfg.d_model
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # pool sur la dernière dim (patch_num)
            nn.Flatten(start_dim=1),  # (B, n_vars * d_model)
            nn.Dropout(cfg.dropout),
            nn.Linear(head_in_dim, n_classes),
        )

        # Init fix : éviter le mode collapse comme observé dans notre code maison
        if init_head_zero:
            with torch.no_grad():
                final_linear = self.classification_head[-1]
                nn.init.zeros_(final_linear.bias)
                final_linear.weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, n_channels, seq_len)
        Returns:
            logits : (B, n_classes)
        """
        # On bypass le forward() complet du PatchTST_backbone (qui fait
        # des transformations destinées à la tête forecasting et qui
        # crashent sans elle). On appelle directement les sous-modules
        # nécessaires : RevIN -> patching -> TSTiEncoder.

        # 1. RevIN normalize
        z = x.permute(0, 2, 1)                              # (B, seq_len, n_vars)
        if self.cfg.revin:
            z = self.official_backbone.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)                              # (B, n_vars, seq_len)

        # 2. Padding (si padding_patch='end')
        if self.cfg.padding_patch == "end":
            z = self.official_backbone.padding_patch_layer(z)

        # 3. Patching : (B, n_vars, seq_len) -> (B, n_vars, patch_num, patch_len)
        z = z.unfold(
            dimension=-1,
            size=self.cfg.patch_len,
            step=self.cfg.stride,
        )
        # 4. Reshape pour leur TSTiEncoder qui attend (B, n_vars, patch_len, patch_num)
        z = z.permute(0, 1, 3, 2)                          # (B, n_vars, patch_len, patch_num)

        # 5. Le TSTiEncoder sort (B, n_vars, d_model, patch_num)
        features = self.official_backbone.backbone(z)

        # 6. Notre tête classification
        # Pool sur patch_num puis flatten n_vars × d_model
        B, n_vars, d_model, patch_num = features.shape
        features_reshaped = features.reshape(B * n_vars, d_model, patch_num)
        pooled = nn.functional.adaptive_avg_pool1d(features_reshaped, 1)
        pooled = pooled.squeeze(-1)
        pooled = pooled.reshape(B, n_vars * d_model)

        # Dropout + Linear (skip Pool et Flatten du Sequential car déjà fait)
        logits = self.classification_head[2](pooled)       # Dropout
        logits = self.classification_head[3](logits)       # Linear

        return logits

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
