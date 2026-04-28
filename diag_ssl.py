# diag_ssl.py — à lancer à la racine du projet
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.patchtst import PatchTSTConfig, PatchTSTReconstructor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(42)

cfg = PatchTSTConfig(
    n_channels=14, seq_len=20, patch_len=4, stride=2,
    d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2,
)
model = PatchTSTReconstructor(cfg).to(device)

x = torch.randn(4, 14, 20).to(device)
print(f"\nInput x: shape={tuple(x.shape)}, "
      f"min={x.min().item():.3f}, max={x.max().item():.3f}, "
      f"any_inf={torch.isinf(x).any().item()}, any_nan={torch.isnan(x).any().item()}")

# --- Étape 1 : RevIN normalize ---
x_norm = model.backbone.revin(x, mode="norm")
print(f"\nAfter RevIN: shape={tuple(x_norm.shape)}, "
      f"min={x_norm.min().item():.3f}, max={x_norm.max().item():.3f}, "
      f"any_inf={torch.isinf(x_norm).any().item()}, "
      f"any_nan={torch.isnan(x_norm).any().item()}")

print(f"  RevIN mean.shape={tuple(model.backbone.revin.mean.shape)}, "
      f"stdev.shape={tuple(model.backbone.revin.stdev.shape)}")
print(f"  RevIN stdev: min={model.backbone.revin.stdev.min().item():.6f}, "
      f"max={model.backbone.revin.stdev.max().item():.6f}")

# --- Étape 2 : forward complet ---
with torch.no_grad():
    recon, mask, target = model(x, mask_ratio=0.4)

print(f"\nrecon: min={recon.min().item():.3f}, max={recon.max().item():.3f}, "
      f"any_inf={torch.isinf(recon).any().item()}, "
      f"any_nan={torch.isnan(recon).any().item()}")
print(f"target: min={target.min().item():.3f}, max={target.max().item():.3f}, "
      f"any_inf={torch.isinf(target).any().item()}, "
      f"any_nan={torch.isnan(target).any().item()}")
print(f"mask: dtype={mask.dtype}, n_True={mask.sum().item()}, "
      f"total={mask.numel()}")

# --- Étape 3 : calcul détaillé de la loss ---
diff = recon - target
print(f"\ndiff: min={diff.min().item():.3f}, max={diff.max().item():.3f}, "
      f"any_inf={torch.isinf(diff).any().item()}, "
      f"any_nan={torch.isnan(diff).any().item()}")

sq = diff.pow(2)
print(f"diff**2: min={sq.min().item():.3f}, max={sq.max().item():.3f}, "
      f"any_inf={torch.isinf(sq).any().item()}, "
      f"any_nan={torch.isnan(sq).any().item()}")

per_patch = sq.mean(dim=-1)
print(f"per_patch_mse: shape={tuple(per_patch.shape)}, "
      f"min={per_patch.min().item():.3f}, max={per_patch.max().item():.3f}, "
      f"any_inf={torch.isinf(per_patch).any().item()}")

masked = per_patch * mask
print(f"masked: any_inf={torch.isinf(masked).any().item()}, "
      f"sum={masked.sum().item():.3f}")
print(f"mask.sum: {mask.sum().item()}")

loss_manual = masked.sum() / mask.sum().clamp(min=1.0)
print(f"\nmanual loss: {loss_manual.item()}")
loss_method = model.reconstruction_loss(recon, target, mask)
print(f"method loss: {loss_method.item()}")

# --- Étape 4 : test sur CPU pour exclure MPS ---
print("\n--- Same test on CPU ---")
model_cpu = PatchTSTReconstructor(cfg).to("cpu")
x_cpu = torch.randn(4, 14, 20)
torch.manual_seed(42)
with torch.no_grad():
    recon_c, mask_c, target_c = model_cpu(x_cpu, mask_ratio=0.4)
loss_cpu = model_cpu.reconstruction_loss(recon_c, target_c, mask_c)
print(f"CPU loss: {loss_cpu.item()}")
print(f"CPU recon: min={recon_c.min().item():.3f}, max={recon_c.max().item():.3f}")
print(f"CPU target: min={target_c.min().item():.3f}, max={target_c.max().item():.3f}")