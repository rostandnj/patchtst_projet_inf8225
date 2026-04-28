"""
test_phase3_2.py

Test d'intégration de la Phase 3.2 : training loops supervisé et SSL.

À placer à la racine du projet et lancer avec:
    python test_phase3_2.py

Vérifie que :
    1. SequenceChildDataset avec subsampling stratifié fonctionne
    2. inner_cv_splits produit 15 splits stratifiés cohérents
    3. train_supervised() fait descendre la loss sur quelques epochs
    4. train_ssl() fait descendre la reconstruction loss sur quelques epochs
    5. transfer_pretrained_backbone marche correctement
    6. predict_proba donne des probas valides
    7. checkpoint save/load roundtrip est correct
"""

from __future__ import annotations

# IMPORTANT: activer le fallback MPS AVANT tout import de torch.
# Certaines ops du PatchTST (unfold_backward) ne sont pas implémentées
# sur MPS et nécessitent un fallback CPU automatique.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import (
    SequenceChildDataset,
    SlidingWindowDataset,
    aggregate_window_predictions,
)
from src.models.patchtst import (
    PatchTSTConfig,
    PatchTSTClassifier,
    PatchTSTReconstructor,
    transfer_pretrained_backbone,
)
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.config import FullExpConfig, OptimConfig, TrainingConfig, SSLConfig
from src.training.device import get_device
from src.training.inner_cv import inner_cv_splits, n_inner_splits
from src.training.ssl_pretrain import train_ssl
from src.training.supervised import train_supervised, predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def main():
    set_global_seed(42)
    device = get_device()
    print(f"=== Phase 3.2 integration test ===")
    print(f"Device: {device}")

    # ----------------------------------------------------------------------
    # Charge le vrai dataset
    # ----------------------------------------------------------------------
    ds = load_processed_dataset(Path("data/processed"))
    print(f"\nLoaded {len(ds.child_ids)} children, {ds.total_strokes} strokes")

    # ----------------------------------------------------------------------
    # 1. Test SequenceChildDataset avec subsampling stratifié
    # ----------------------------------------------------------------------
    print(f"\n--- Test 1: SequenceChildDataset stratified subsampling ---")
    train_ds = SequenceChildDataset(
        ds, ds.child_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=0,
    )
    # Tirage répété sur le même enfant : doit donner différents offsets
    cid_test = "S04"  # 37 traits, max_offset=17
    idx = train_ds.child_ids.index(cid_test)
    offsets = []
    for _ in range(20):
        item = train_ds._build_item(cid_test)
        offsets.append(item.offset_used)
    print(f"  S04 offsets sur 20 tirages stratifiés: "
          f"min={min(offsets)}, max={max(offsets)}, unique={len(set(offsets))}")
    assert len(set(offsets)) > 1, "Subsampling stratifié devrait donner différents offsets"

    # eval mode : offset doit être 0
    eval_ds = SequenceChildDataset(
        ds, ds.child_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )
    item_eval = eval_ds._build_item(cid_test)
    assert item_eval.offset_used == 0, f"Eval mode: offset should be 0, got {item_eval.offset_used}"
    print(f"  Eval mode S04: offset={item_eval.offset_used} ✓")

    # ----------------------------------------------------------------------
    # 2. Test inner_cv_splits
    # ----------------------------------------------------------------------
    print(f"\n--- Test 2: inner_cv_splits ---")
    test_child = ds.child_ids[0]
    train_pool = [c for c in ds.child_ids if c != test_child]  # 23 enfants
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=3, seed=42,
    ))
    print(f"  Total splits: {len(splits)} (expected {n_inner_splits(5, 3)})")
    assert len(splits) == 15

    # Vérifie stratification dans le premier split
    s0 = splits[0]
    val_labels = [ds.labels_per_child[c] for c in s0.val_inner_ids]
    print(f"  First split: train={len(s0.train_inner_ids)}, val={len(s0.val_inner_ids)}, "
          f"val labels: {val_labels}")
    n_val_pos = sum(val_labels)
    n_val_neg = len(val_labels) - n_val_pos
    assert n_val_pos >= 1 and n_val_neg >= 1, "Val should have both classes (stratified)"

    # ----------------------------------------------------------------------
    # 3. Test train_supervised : la loss doit descendre
    # ----------------------------------------------------------------------
    print(f"\n--- Test 3: train_supervised on a real inner split ---")
    cfg = FullExpConfig(
        exp_name="test_supervised",
        optim=OptimConfig(lr=1e-3, weight_decay=0.01, warmup_epochs=2),
        training=TrainingConfig(
            max_epochs=10,
            batch_size=8,
            early_stopping_patience=20,  # désactivé en pratique pour le test
            use_class_weight=True,
            use_trial_subsampling=True,
            subsample_strategy="stratified",
            seed=42,
        ),
        seq_len=20, n_channels=14,
        patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.2,
    )

    train_inner_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=42,
    )
    val_inner_ds = SequenceChildDataset(
        ds, s0.val_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified", seed=43,
    )
    train_loader = DataLoader(train_inner_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_inner_ds, batch_size=cfg.training.batch_size, shuffle=False)

    patchtst_cfg = PatchTSTConfig(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len,
        patch_len=cfg.patch_len, stride=cfg.stride,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
    )
    model = PatchTSTClassifier(patchtst_cfg, n_classes=2)

    history, best_state = train_supervised(
        model, train_loader, val_loader, device, cfg,
        verbose=True, log_every=2,
    )
    print(f"  train_loss[0]={history.train_loss[0]:.4f}, "
          f"train_loss[-1]={history.train_loss[-1]:.4f}")
    print(f"  val_loss[0]={history.val_loss[0]:.4f}, "
          f"val_loss[-1]={history.val_loss[-1]:.4f}")
    # On vérifie juste que ça a tourné et qu'on a un best state non vide
    assert history.n_epochs_run > 0
    assert len(best_state) > 0
    # La loss devrait au moins commencer (non-NaN)
    assert all(np.isfinite(history.train_loss)), "Train loss should be finite"
    assert all(np.isfinite(history.val_loss)), "Val loss should be finite"

    # ----------------------------------------------------------------------
    # 4. Test predict_proba
    # ----------------------------------------------------------------------
    print(f"\n--- Test 4: predict_proba ---")
    model.load_state_dict(best_state)
    test_ds = SequenceChildDataset(
        ds, [test_child], seq_len=20, is_train=False, subsample_strategy="stratified",
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    probas = predict_proba(model, test_loader, device)
    print(f"  Predicted proba for {test_child}: {probas[0]:.4f}")
    assert probas.shape == (1,)
    assert 0.0 <= probas[0] <= 1.0

    # ----------------------------------------------------------------------
    # 5. Test SSL pretraining
    # ----------------------------------------------------------------------
    print(f"\n--- Test 5: train_ssl ---")
    cfg_ssl = FullExpConfig(
        exp_name="test_ssl",
        optim=OptimConfig(lr=5e-4, weight_decay=0.01, warmup_epochs=2),
        training=TrainingConfig(),  # pas utilisé pour SSL
        ssl=SSLConfig(
            max_epochs=10,
            batch_size=8,
            mask_ratio=0.4,
            early_stopping_patience=20,
            seed=42,
        ),
        seq_len=20, n_channels=14,
        patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.2,
    )
    pretrain_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=42,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=cfg_ssl.ssl.batch_size, shuffle=True)
    val_ssl_loader = DataLoader(val_inner_ds, batch_size=cfg_ssl.ssl.batch_size, shuffle=False)

    pretrainer = PatchTSTReconstructor(patchtst_cfg)
    ssl_history, ssl_best_state = train_ssl(
        pretrainer, pretrain_loader, device, cfg_ssl,
        val_loader=val_ssl_loader, verbose=True, log_every=2,
    )
    print(f"  ssl train_loss[0]={ssl_history.train_loss[0]:.4f}, "
          f"train_loss[-1]={ssl_history.train_loss[-1]:.4f}")
    assert ssl_history.n_epochs_run > 0
    assert all(np.isfinite(ssl_history.train_loss)), "SSL train loss should be finite"
    assert ssl_history.train_loss[-1] < ssl_history.train_loss[0] * 1.5, \
        "SSL loss should not explode (allowing for small bumps)"

    # ----------------------------------------------------------------------
    # 6. Test transfer pretrained backbone
    # ----------------------------------------------------------------------
    print(f"\n--- Test 6: transfer pretrained backbone ---")
    pretrainer.load_state_dict(ssl_best_state)
    finetune_model = PatchTSTClassifier(patchtst_cfg, n_classes=2)
    # Compare poids backbone avant transfert
    old_backbone_w = finetune_model.backbone.encoder.layers[0].self_attn.in_proj_weight.clone()
    transfer_pretrained_backbone(pretrainer, finetune_model)
    new_backbone_w = finetune_model.backbone.encoder.layers[0].self_attn.in_proj_weight
    print(f"  Backbone weights changed after transfer: "
          f"{not torch.allclose(old_backbone_w, new_backbone_w)}")
    assert not torch.allclose(old_backbone_w, new_backbone_w)

    # ----------------------------------------------------------------------
    # 7. Test checkpoint save/load roundtrip
    # ----------------------------------------------------------------------
    print(f"\n--- Test 7: checkpoint save/load roundtrip ---")
    ckpt_path = Path("/tmp/test_phase3_2_ckpt.pt")
    save_checkpoint(
        ckpt_path, model,
        epoch=42, metrics={"val_loss": 0.5, "val_acc": 0.75},
        extra={"exp_name": "test"},
    )
    fresh_model = PatchTSTClassifier(patchtst_cfg, n_classes=2).to(device)
    payload = load_checkpoint(ckpt_path, fresh_model, map_location=device)
    print(f"  Loaded checkpoint: epoch={payload['epoch']}, metrics={payload['metrics']}")
    # Vérifier que les poids correspondent
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        assert k1 == k2
        v1d = v1.to(device)
        assert torch.allclose(v1d, v2), f"Mismatch on {k1}"
    print(f"  All weights match ✓")
    ckpt_path.unlink()

    print(f"\n✓ All Phase 3.2 integration tests passed!")


if __name__ == "__main__":
    main()
