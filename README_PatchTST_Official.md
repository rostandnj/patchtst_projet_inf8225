# Test PatchTST officiel sur Exp A

## Objectif

Valider notre conclusion (PatchTST AUC ~0.45 sur Exp A) en utilisant
le **backbone officiel** du repo `yuqinie98/PatchTST` au lieu de notre
implémentation maison.

Si l'official donne **les mêmes résultats** → notre conclusion est bétonnée.
Si l'official donne **de bien meilleurs résultats** → on a un bug à corriger.

## Procédure (à exécuter dans l'ordre)

### Étape 1 — Cloner le repo officiel et copier les fichiers

```bash
# Dans un dossier temporaire
cd /tmp
git clone https://github.com/yuqinie98/PatchTST.git PatchTST_official

# Copier les 3 fichiers nécessaires dans ton projet
PROJECT_DIR=/Users/rostandnj/SCHOOLS/INF8225/PATCHTST_VF
cp /tmp/PatchTST_official/PatchTST_supervised/layers/RevIN.py             $PROJECT_DIR/src/models/patchtst_official/
cp /tmp/PatchTST_official/PatchTST_supervised/layers/PatchTST_layers.py   $PROJECT_DIR/src/models/patchtst_official/
cp /tmp/PatchTST_official/PatchTST_supervised/layers/PatchTST_backbone.py $PROJECT_DIR/src/models/patchtst_official/
```

### Étape 2 — Adapter les imports dans les fichiers officiels

Le code officiel utilise des imports relatifs comme :
```python
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
```

Mais notre structure est différente. Tu dois éditer manuellement les imports
dans `PatchTST_backbone.py` :

```python
# AVANT (dans PatchTST_backbone.py, ligne ~10):
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# APRÈS:
from src.models.patchtst_official.PatchTST_layers import *
from src.models.patchtst_official.RevIN import RevIN
```

### Étape 3 — Lancer le test

```bash
# Run rapide single-seed (~1 min)
python test_phase3_official_loso.py

# Si OK, run multi-seed pour comparaison robuste (~5 min)
python test_phase3_official_loso.py --multi-seed 5 --quiet
```

## Résultats attendus

À la fin du run multi-seed, on verra :

```
=== Comparison ===
  Custom PatchTST (5 seeds):   acc=0.400 ± 0.077, AUC=0.450 ± 0.085
  Official PatchTST (5 seeds): acc=X.XXX ± X.XXX, AUC=X.XXX ± X.XXX
  Baseline RICH_STATS/GB:      acc=0.792, AUC=0.806
```

### Interprétation

- **Si AUC officiel ≈ 0.45 ± 0.10** : notre code est OK, le problème est
  intrinsèque au dataset Exp A. **Conclusion bétonnée.**

- **Si AUC officiel > 0.55** : on a un bug à investiguer dans notre code.

- **Si AUC officiel = 0.50** (mode collapse total) : leur init_head_zero
  ne fonctionne pas pareil, on doit regarder.
