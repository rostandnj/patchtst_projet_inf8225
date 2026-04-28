"""
test_phase_b2_diag_nan.py

DIAGNOSTIC : pourquoi NaN dans P(ADHD) sur Exp B ?

Hypothèses possibles :
    1. RevIN explose à cause du padding zéro (std proche de 0 sur certains canaux)
    2. Amplitudes des coordonnées trop élevées (jusqu'à ±177 mm), Transformer overflow
    3. Bug spécifique à MPS sur certaines opérations

PROCÉDURE :
    1. Forward sur 5 batches en mode eval, sans entraînement
    2. Vérifier les NaN à CHAQUE étape : input -> RevIN -> patches -> backbone -> head
    3. Si NaN, identifier l'étape responsable
    4. Refaire avec normalisation pré-RevIN pour vérifier que ça résout

Durée : ~30 secondes.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_b_datasets import RawTraceStrokeDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.device import get_device
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed
from src.utils.traces_io import load_traces


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def check_tensor(t, name):
    """Affiche stats + flag si NaN/Inf."""
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    flag = ""
    if n_nan > 0: flag += f" ⚠ NAN={n_nan}"
    if n_inf > 0: flag += f" ⚠ INF={n_inf}"
    if n_nan == 0 and n_inf == 0:
        print(f"  {name}: shape={tuple(t.shape)}, "
              f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
              f"mean={t.mean().item():.4f}, std={t.std().item():.4f}")
    else:
        # On ne peut pas calculer min/max sur un tenseur avec NaN
        finite_mask = torch.isfinite(t)
        n_finite = int(finite_mask.sum().item())
        if n_finite > 0:
            t_finite = t[finite_mask]
            print(f"  {name}: shape={tuple(t.shape)}{flag}, "
                  f"finite_min={t_finite.min().item():.4f}, "
                  f"finite_max={t_finite.max().item():.4f}, "
                  f"finite_mean={t_finite.mean().item():.4f}")
        else:
            print(f"  {name}: shape={tuple(t.shape)}{flag} (ALL NON-FINITE)")
    return n_nan + n_inf


def main():
    set_global_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Charge les données
    traces = load_traces(Path("data/processed"))
    ds_params = load_processed_dataset(Path("data/processed"))
    labels = ds_params.labels_per_child

    # =====================================================================
    # 1. INPUT : statistiques détaillées d'un batch
    # =====================================================================
    section("1. Input batch (raw traces, x/y centered, T=200)")
    ds = RawTraceStrokeDataset(
        traces=traces, labels_per_child=labels,
        child_ids=traces.child_ids,
        target_len=200, channels=("x", "y"), center_xy_flag=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    print(f"  Batch shape: {tuple(x_batch.shape)}, labels: {y_batch.tolist()[:8]}...")
    check_tensor(x_batch, "x_batch")

    # Stats par sample
    print(f"\n  Per-sample stats (first 8 samples):")
    for i in range(min(8, x_batch.size(0))):
        sample = x_batch[i]
        # Détection padding (zéros consécutifs à la fin)
        x_chan = sample[0]
        nonzero_mask = x_chan != 0
        n_nonzero = int(nonzero_mask.sum().item())
        # Le padding est en fin, donc on cherche le dernier index non-zéro
        if nonzero_mask.any():
            last_nonzero = int(nonzero_mask.nonzero()[-1].item())
        else:
            last_nonzero = -1
        print(f"    sample {i}: n_actual~{last_nonzero+1}, "
              f"x[std={sample[0].std().item():.3f}], "
              f"y[std={sample[1].std().item():.3f}]")

    # =====================================================================
    # 2. RevIN seul : sortie sur ce batch
    # =====================================================================
    section("2. After RevIN normalization")
    cfg = PatchTSTConfig(
        n_channels=2, seq_len=200, patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2, init_head_zero=True).to(device)
    model.eval()

    x = x_batch.to(device)
    with torch.no_grad():
        # Forward étape par étape via le backbone
        # Le backbone applique : RevIN -> patches -> embed -> encoder
        # On hooke chaque étape

        # Étape 1 : RevIN
        x_norm = model.backbone.revin(x, mode="norm")
    n_bad = check_tensor(x_norm, "x_norm (after RevIN)")
    if n_bad > 0:
        print(f"  ⚠ NaN/Inf after RevIN. Probable cause: a sample has "
              f"std≈0 on some channel.")
        # Localiser quel sample/canal
        per_sample_std = x.std(dim=-1)  # (B, n_channels)
        print(f"  Per-sample std (avant RevIN):")
        print(f"    min={per_sample_std.min().item():.6f}")
        print(f"    samples with std < 1e-3:")
        bad_mask = per_sample_std < 1e-3
        if bad_mask.any():
            indices = bad_mask.nonzero()
            for idx in indices[:5]:
                b, c = idx.tolist()
                print(f"      sample {b}, channel {c}: "
                      f"std={per_sample_std[b, c].item():.8f}")

    # =====================================================================
    # 3. Patching
    # =====================================================================
    section("3. After patching")
    with torch.no_grad():
        # Apply patching to x_norm
        from src.models.patching import make_patches
        patches = make_patches(x_norm, patch_len=cfg.patch_len, stride=cfg.stride)
    check_tensor(patches, "patches")
    print(f"  patches shape (should be (B, n_channels, n_patches, patch_len))")

    # =====================================================================
    # 4. Forward complet, vérifier NaN à chaque étape
    # =====================================================================
    section("4. Full forward, check NaN at each layer")
    model.eval()

    # Hook pour tracer les NaN
    nan_log = []
    def hook(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                n_nan = int(torch.isnan(output).sum().item())
                n_inf = int(torch.isinf(output).sum().item())
                nan_log.append((name, output.shape, n_nan, n_inf))
        return fn

    # Enregistre des hooks sur les modules clés
    handles = []
    handles.append(model.backbone.register_forward_hook(hook("backbone")))
    handles.append(model.head.register_forward_hook(hook("head")))

    with torch.no_grad():
        try:
            logits = model(x)
            check_tensor(logits, "logits (final)")
            probs = F.softmax(logits, dim=-1)
            check_tensor(probs, "probs (after softmax)")
        except Exception as e:
            print(f"  ⚠ ERROR during forward: {e}")

    print(f"\n  NaN/Inf log per module:")
    for name, shape, n_nan, n_inf in nan_log:
        flag = ""
        if n_nan > 0: flag += f" ⚠ NAN={n_nan}"
        if n_inf > 0: flag += f" ⚠ INF={n_inf}"
        print(f"    {name}: shape={tuple(shape)}{flag}")

    for h in handles: h.remove()

    # =====================================================================
    # 5. Test avec NORMALISATION PRÉ-DATASET
    # =====================================================================
    section("5. Test with pre-normalization (divide by 100 mm)")
    print(f"  Pre-normalize x and y by dividing by 100 (typical amplitude)")

    # Crée un batch normalisé manuellement
    x_normalized = x_batch / 100.0
    print(f"  x_normalized stats:")
    check_tensor(x_normalized, "x_normalized")

    x_norm_dev = x_normalized.to(device)
    with torch.no_grad():
        logits_norm = model(x_norm_dev)
    check_tensor(logits_norm, "logits with pre-normalization")
    probs_norm = F.softmax(logits_norm, dim=-1)
    check_tensor(probs_norm, "probs with pre-normalization")

    # =====================================================================
    # 6. Test SANS RevIN
    # =====================================================================
    section("6. Test WITHOUT RevIN")
    cfg_no_revin = PatchTSTConfig(
        n_channels=2, seq_len=200, patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    # Désactiver RevIN dans le modèle
    model_no_revin = PatchTSTClassifier(cfg_no_revin, n_classes=2,
                                          init_head_zero=True).to(device)
    # On désactive en remplaçant par identité
    model_no_revin.backbone.revin = torch.nn.Identity()
    # Mais Identity ne supporte pas le mode='norm', on doit hacker autrement
    # En fait, regardons le backbone pour savoir comment retirer RevIN proprement

    print(f"  (test skipped — would need refactor of backbone)")
    print(f"  Use fix from section 5 instead.")

    # =====================================================================
    # CONCLUSION
    # =====================================================================
    section("CONCLUSION")
    print(f"  Diagnose hypothesis based on results above:")
    print(f"")
    print(f"  - If RevIN produces NaN -> per-sample std too small (padding)")
    print(f"  - If patches produce NaN -> issue in unfold/permute")
    print(f"  - If only logits NaN -> issue in encoder or head")
    print(f"")
    print(f"  If section 5 (pre-normalize /100) gives finite logits,")
    print(f"  the fix is to scale x,y before passing to the model.")


if __name__ == "__main__":
    main()
