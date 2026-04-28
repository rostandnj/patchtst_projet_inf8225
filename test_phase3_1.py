# test_phase3_1.py — à mettre à la racine du projet
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.patchtst import (
    PatchTSTConfig, PatchTSTClassifier, PatchTSTReconstructor,
    transfer_pretrained_backbone,
)
from src.utils.data_io import load_processed_dataset
from src.data.exp_a_datasets import (
    SequenceChildDataset, SlidingWindowDataset, aggregate_window_predictions,
)

# Vérification device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Charge le vrai dataset
ds = load_processed_dataset(Path("data/processed"))
print(f"Loaded {len(ds.child_ids)} children, {ds.total_strokes} strokes")

# Test 1 : SequenceChildDataset
seq_ds = SequenceChildDataset(ds, ds.child_ids, seq_len=20, return_dict=True)
print(f"\n--- SequenceChildDataset (L=20) ---")
for i in range(3):
    item = seq_ds[i]
    print(f"  {item['child_id']}: x.shape={tuple(item['x'].shape)}, "
          f"y={item['y'].item()}, n_padded={item['pad_mask'].sum().item()}")

# Test 2 : SlidingWindowDataset
win_ds = SlidingWindowDataset(ds, ds.child_ids, window_len=10, stride=1)
print(f"\n--- SlidingWindowDataset (window=10, stride=1) ---")
print(f"  total windows: {len(win_ds)}")
print(f"  children skipped (< 10 strokes): {win_ds._skipped_children}")
x, y = win_ds[0]
print(f"  first window: x.shape={tuple(x.shape)}, y={y.item()}")

# Test 3 : Forward d'un classifier sur un batch
print(f"\n--- PatchTSTClassifier forward test ---")
cfg = PatchTSTConfig(
    n_channels=14, seq_len=20, patch_len=4, stride=2,
    d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2,
)
print(f"  Config n_patches: {cfg.n_patches}")

model = PatchTSTClassifier(cfg, n_classes=2).to(device)
print(f"  Model n_parameters: {model.n_parameters():,}")

# Batch synthétique de 4 enfants
batch_x = torch.randn(4, 14, 20).to(device)
logits = model(batch_x)
print(f"  Input: {tuple(batch_x.shape)} -> Output: {tuple(logits.shape)}")
assert logits.shape == (4, 2), "Bad output shape"

# Test 4 : Forward d'un reconstructor (SSL)
print(f"\n--- PatchTSTReconstructor forward test ---")
recon_model = PatchTSTReconstructor(cfg).to(device)
print(f"  Reconstructor n_parameters: {recon_model.n_parameters():,}")

recon, mask, target = recon_model(batch_x, mask_ratio=0.4)
print(f"  recon shape: {tuple(recon.shape)}")
print(f"  mask shape: {tuple(mask.shape)}, mask_fraction: {mask.float().mean():.3f}")
print(f"  target shape: {tuple(target.shape)}")
loss = recon_model.reconstruction_loss(recon, target, mask)
print(f"  loss: {loss.item():.4f}")

# Test 5 : Transfer pretrained -> classifier
print(f"\n--- Transfer pretrained backbone ---")
classifier_finetune = PatchTSTClassifier(cfg, n_classes=2).to(device)
transfer_pretrained_backbone(recon_model, classifier_finetune)
# Vérifie qu'au moins certains poids ont changé
old_w = model.backbone.encoder.layers[0].self_attn.in_proj_weight.clone()
new_w = classifier_finetune.backbone.encoder.layers[0].self_attn.in_proj_weight
print(f"  weights different: {not torch.allclose(old_w, new_w)}")

# Test 6 : Forward sur fenêtres
cfg_win = PatchTSTConfig(
    n_channels=14, seq_len=10, patch_len=2, stride=1,
    d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2,
)
print(f"\n--- PatchTSTClassifier sur fenêtres (L=10) ---")
print(f"  Config n_patches: {cfg_win.n_patches}")
model_win = PatchTSTClassifier(cfg_win, n_classes=2).to(device)
batch_x_win = torch.randn(8, 14, 10).to(device)
logits_win = model_win(batch_x_win)
print(f"  Input: {tuple(batch_x_win.shape)} -> Output: {tuple(logits_win.shape)}")

print("\n✓ All Phase 3.1 tests passed!")