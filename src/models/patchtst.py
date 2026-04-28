"""
src/models/patchtst.py

PatchTST principal pour Exp A : classification binaire au niveau séquence
(ou fenêtre glissante) à partir d'une série multivariée de paramètres
sigma-lognormaux par enfant.

ARCHITECTURE :
    Input: (B, M, L)
        B = batch size
        M = nombre de canaux (typiquement 14 paramètres sigma-lognormaux)
        L = longueur de séquence (typiquement 20 traits, ou 10 pour fenêtres)

    1. RevIN normalize       (B, M, L)
    2. Channel-Independence : reshape -> (B*M, L)
    3. PatchEmbedding        -> (B*M, N, D) avec PE
    4. TransformerEncoder    -> (B*M, N, D)
    5. Reshape               -> (B, M, N, D)
    6. Tête (classification ou reconstruction)

DEUX MODES DE TÊTE :

    ClassificationHead :
        Prend (B, M, N, D), produit (B, n_classes).
        Stratégie : flatten temporel (mean pool sur N) -> concat sur M ->
        linear vers n_classes. Capture toutes les contributions canal x
        position avant classification finale.

    ReconstructionHead (pour SSL pretraining) :
        Prend (B*M, N, D), produit (B*M, N, P).
        Linear projection D -> P sur chaque token. À utiliser avec un masque
        binaire (B*M, N) indiquant quels patches étaient masqués pour
        calculer la loss MSE uniquement sur eux.

USAGE TYPE :

    # Classification supervisée
    model = PatchTSTClassifier(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2,
        n_classes=2,
    )
    logits = model(x)  # x: (B, 14, 20) -> logits: (B, 2)

    # SSL pretraining
    pretrainer = PatchTSTReconstructor(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2,
    )
    recon, mask, target = pretrainer(x, mask_ratio=0.4)
    loss = ((recon - target) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .patching import PatchEmbedding, compute_n_patches, make_patches
from .revin import RevIN
from .transformer import TransformerEncoder


@dataclass
class PatchTSTConfig:
    """Hyperparamètres d'un modèle PatchTST."""
    n_channels: int          # M (ex: 14 paramètres)
    seq_len: int             # L (ex: 20 traits)
    patch_len: int           # P (ex: 4)
    stride: int              # S (ex: 2)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.2
    attn_dropout: float = 0.1
    use_revin: bool = True
    revin_affine: bool = True

    @property
    def n_patches(self) -> int:
        return compute_n_patches(self.seq_len, self.patch_len, self.stride)


class PatchTSTBackbone(nn.Module):
    """Encodeur PatchTST avec channel-independence.

    Forward:
        x : (B, M, L)
        return : (B, M, N, D)
    """

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_patches = cfg.n_patches

        if cfg.use_revin:
            self.revin = RevIN(num_features=cfg.n_channels, affine=cfg.revin_affine)
        else:
            self.revin = None

        self.patch_embed = PatchEmbedding(
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            max_n_patches=self.n_patches + 4,  # marge de sécurité
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.encoder = TransformerEncoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, M, L) -> (B, M, N, D)."""
        B, M, L = x.shape
        if M != self.cfg.n_channels:
            raise RuntimeError(f"Expected M={self.cfg.n_channels} channels, got {M}")
        if L != self.cfg.seq_len:
            raise RuntimeError(f"Expected L={self.cfg.seq_len}, got {L}")

        # 1. RevIN
        if self.revin is not None:
            x = self.revin(x, mode="norm")

        # 2. Channel-independence : (B, M, L) -> (B*M, L)
        x_ci = x.reshape(B * M, L)

        # 3. Patch embedding -> (B*M, N, D)
        z = self.patch_embed(x_ci)

        # 4. Transformer encoder -> (B*M, N, D)
        z = self.encoder(z)

        # 5. Reshape -> (B, M, N, D)
        N, D = z.shape[1], z.shape[2]
        z = z.reshape(B, M, N, D)
        return z


class ClassificationHead(nn.Module):
    """Tête de classification depuis (B, M, N, D) vers (B, n_classes).

    Stratégie : mean pool sur N, flatten sur M, linear vers n_classes.
    Régularisation : LayerNorm + dropout avant la projection finale.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = n_channels * d_model
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z : (B, M, N, D)
        # Mean pool sur N : (B, M, D)
        z_pooled = z.mean(dim=2)
        # Flatten sur M : (B, M*D)
        z_flat = z_pooled.reshape(z_pooled.size(0), -1)
        z_flat = self.dropout(self.norm(z_flat))
        return self.fc(z_flat)


class ReconstructionHead(nn.Module):
    """Tête de reconstruction pour SSL : projette (D) -> (P) par token.

    Forward:
        z : (B, M, N, D)  output du backbone
        return : (B, M, N, P) reconstruction
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.fc = nn.Linear(d_model, patch_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z : (B, M, N, D) -> (B, M, N, P)
        return self.fc(z)


class PatchTSTClassifier(nn.Module):
    """PatchTST + tête classification.

    Forward:
        x : (B, M, L)
        return : (B, n_classes)

    Args:
        cfg : PatchTSTConfig
        n_classes : nombre de classes (2 pour binaire CTRL/ADHD)
        init_head_zero : si True, force le bias de la tête de classification
            à 0 et scale ses poids par 0.01. Recommandé sur petit n
            pour éviter le biais d'initialisation qui peut conduire à un
            mode collapse vers une classe pendant l'entraînement.
    """

    def __init__(
        self,
        cfg: PatchTSTConfig,
        n_classes: int = 2,
        init_head_zero: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.backbone = PatchTSTBackbone(cfg)
        self.head = ClassificationHead(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            n_classes=n_classes,
            dropout=cfg.dropout,
        )
        if init_head_zero:
            with torch.no_grad():
                nn.init.zeros_(self.head.fc.bias)
                self.head.fc.weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchTSTReconstructor(nn.Module):
    """PatchTST + tête de reconstruction pour SSL pretraining.

    Forward retourne directement (recon, mask, target) où :
        recon  : (B, M, N, P) reconstruction prédite
        mask   : (B, M, N) booléen, True = patch masqué
        target : (B, M, N, P) patches originaux (avant masquage)
    Tu peux ensuite calculer la loss MSE sur les patches masqués.
    """

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = PatchTSTBackbone(cfg)
        self.head = ReconstructionHead(d_model=cfg.d_model, patch_len=cfg.patch_len)
        # Token d'embedding pour les patches masqués (appris)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self, x: torch.Tensor, mask_ratio: float = 0.4, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward SSL.

        Args:
            x : (B, M, L)
            mask_ratio : fraction de patches à masquer (par échantillon × canal)
            mask : optionnellement fournir un masque déterministe (B, M, N)
                bool. Si None, on échantillonne aléatoirement.

        Returns:
            recon : (B, M, N, P)
            mask : (B, M, N) bool, True = masqué
            target : (B, M, N, P) — patches dans l'espace normalisé
        """
        B, M, L = x.shape

        # 1. Normalisation RevIN UNE SEULE FOIS (les stats sont mémorisées)
        if self.backbone.revin is not None:
            x_norm = self.backbone.revin(x, mode="norm")
        else:
            x_norm = x

        # 2. Patches cibles : extraits des données normalisées
        x_ci = x_norm.reshape(B * M, L)
        target_patches = make_patches(x_ci, self.cfg.patch_len, self.cfg.stride)
        N = target_patches.shape[1]
        target = target_patches.reshape(B, M, N, self.cfg.patch_len)

        # 3. Embedding des patches (mêmes patches que target, mais projetés)
        z = self.backbone.patch_embed(x_ci)         # (B*M, N, D)

        # 4. Génération du masque si pas fourni
        if mask is None:
            mask = self._sample_mask(B, M, N, mask_ratio, x.device)

        # 5. Remplacement des positions masquées par mask_token
        mask_flat = mask.reshape(B * M, N)
        z = torch.where(
            mask_flat.unsqueeze(-1),
            self.mask_token.expand(B * M, N, -1),
            z,
        )

        # 6. Encoder Transformer
        z = self.backbone.encoder(z)                # (B*M, N, D)
        z = z.reshape(B, M, N, -1)

        # 7. Tête de reconstruction
        recon = self.head(z)                        # (B, M, N, P)

        return recon, mask, target

    @staticmethod
    def _sample_mask(
        B: int, M: int, N: int, mask_ratio: float, device: torch.device,
    ) -> torch.Tensor:
        """Échantillonne un masque uniformément, fraction mask_ratio par séquence."""
        n_mask = max(1, int(round(N * mask_ratio)))
        # Pour chaque (B, M) on tire n_mask indices sans remise
        # Implémentation simple : noise + topk
        noise = torch.rand(B, M, N, device=device)
        # Indices des n_mask plus grandes valeurs = positions masquées
        idx = noise.topk(n_mask, dim=-1).indices       # (B, M, n_mask)
        mask = torch.zeros(B, M, N, dtype=torch.bool, device=device)
        mask.scatter_(-1, idx, True)
        return mask

    def reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE sur les patches masqués uniquement.

        Note: on convertit explicitement mask.sum() en float AVANT le clamp,
        sinon on hit un bug MPS sur la conversion int64->float dans clamp.
        """
        # recon, target : (B, M, N, P) ; mask : (B, M, N)
        per_patch_mse = (recon - target).pow(2).mean(dim=-1)  # (B, M, N)
        masked_count = mask.sum().to(per_patch_mse.dtype).clamp(min=1.0)
        return (per_patch_mse * mask).sum() / masked_count

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def transfer_pretrained_backbone(
    pretrained: PatchTSTReconstructor,
    classifier: PatchTSTClassifier,
) -> None:
    """Transfère le backbone pré-entraîné vers le classifier (in-place).

    Utilisé après SSL pretraining : on jette la tête de reconstruction et on
    initialise le classifier avec le backbone (RevIN + patch_embed + encoder)
    appris pendant le pretraining.
    """
    if pretrained.cfg.d_model != classifier.cfg.d_model:
        raise ValueError("d_model mismatch between pretrained and classifier")
    if pretrained.cfg.n_channels != classifier.cfg.n_channels:
        raise ValueError("n_channels mismatch")
    classifier.backbone.load_state_dict(pretrained.backbone.state_dict())
