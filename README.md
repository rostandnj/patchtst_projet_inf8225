# PatchTST pour la détection du TDAH à partir d'écriture manuscrite

# Déclaration d'utilisation d'outils d'IA

Ce projet a bénéficié de l'assistance de Claude (Anthropic).

## Tâches assistées par Claude

- **Architecture du code** : structure des modules `models/patchtst.py`,
  `data/preprocessing.py` discutée et raffinée avec Claude
- **Débogage** : diagnostic du bug MPS `aten::unfold_backward`
- **Rédaction** : structure et reformulation du rapport IJCAI
- **Visualisations** : scripts matplotlib pour figures d'attention LOSO
- **Documentation** : structure de ce README

## Tâches strictement humaines

- Conception scientifique du projet et hypothèses de recherche
- Pré-traitement des données réelles 
- Exécution de toutes les expériences sur GPU/CPU
- Validation et interprétation des résultats
- Décisions méthodologiques finales

## Responsabilité scientifique

L'auteur (Rostand Marlone Njomo Njampou) reste pleinement
responsable du contenu scientifique de ce projet. Toute
question sur les résultats, méthodologies ou décisions
peut être posée à : rostand.njomo@etud.polymtl.ca

> Projet de cours **INF8225 — Techniques probabilistes et d'apprentissage** (Polytechnique Montréal).
> Application de l'architecture **PatchTST** (Patch Time Series Transformer) à la classification
> binaire **TDAH vs contrôle** sur des données de tracés d'écriture chez l'enfant, comparée à
> des baselines classiques et à une analyse d'attention pour l'interprétabilité.
> Utilisation de claude code pour 

---

## 1. Contexte et motivation

Le diagnostic du **TDAH** (Trouble du Déficit de l'Attention avec ou sans Hyperactivité) repose
aujourd'hui sur des évaluations cliniques longues et subjectives. Plusieurs travaux récents
(Faci *et al.*, 2021) ont montré que des **paramètres sigma-lognormaux** extraits de tracés
manuscrits portent un signal moteur lié au TDAH.

Ce projet pose la question :

> Une architecture **transformer pour séries temporelles** (PatchTST) peut-elle capturer ce
> signal mieux qu'une baseline classique opérant sur des features agrégées ? Et si oui, sur
> **quelle représentation** des données — paramètres sigma-lognormaux, traces brutes, ou les deux ?

Trois représentations de l'entrée sont comparées à structure expérimentale identique
(LOSO, multi-seed, classes équilibrées) :

| Exp | Entrée | Granularité | Shape | n train |
|-----|--------|-------------|-------|---------|
| **A** | 14 paramètres sigma-lognormaux | par enfant (séquence de traits) | `(14, 20)` | 22 enfants |
| **B** | Traces brutes `x(t), y(t)` | par trait | `(2, 200)` | ~660 traits |
| **C** | Traces + paramètres concaténés | par trait | `(16, 200)` | ~660 traits |

---

## 2. Données

Dataset privé **Laboratoire Scribens** : 24 enfants (12 TDAH + 12 contrôles) effectuant
des tâches d'écriture standardisées. ~700 traits exploitables après filtrage qualité.

Pour chaque trait, on dispose de :

- la trace brute `x(t), y(t), t` échantillonnée à ~200 Hz ;
- l'extraction sigma-lognormale (SSVn) fournissant 14 paramètres physiologiques par trait
  (D, t0, μ, σ, θs, θe, SNR, ...).

Les données brutes ne sont **pas redistribuées** dans ce dépôt (cf. `.gitignore`). Le
pipeline `scripts/01_preprocess.py` produit `data/processed/{params.npz, traces.npz, metadata.parquet}`.

---

## 3. Architecture

### 3.1 Backbone PatchTST

Implémentation custom suivant Nie *et al.* (2023) avec :

- **RevIN** (Reversible Instance Normalization) par canal ;
- **Channel-Independence** : chaque canal est traité indépendamment, poids partagés ;
- **Patch embedding** : `patch_len` consécutifs → 1 token de dimension `d_model`, avec PE appris ;
- **Transformer encoder** standard (`n_layers`, `n_heads`, `d_ff`, dropout, attn_dropout) ;
- Tête de **classification** : mean-pool temporel → flatten canaux → linéaire vers `n_classes`.

```
(B, M, L)
  ├── RevIN(norm)
  ├── reshape (B*M, L)               # channel independence
  ├── PatchEmbed → (B*M, N, D)
  ├── TransformerEncoder → (B*M, N, D)
  ├── reshape (B, M, N, D)
  └── ClassificationHead → (B, n_classes)
```

Une seconde tête `ReconstructionHead` permet le **pré-entraînement SSL** par masquage
de patches (~40 %) avec loss MSE — utilisée pour un protocole de pretrain optionnel
(`src/training/ssl_pretrain.py`).

### 3.2 Wrapper officiel

Pour valider notre implémentation, le backbone officiel `yuqinie98/PatchTST` est wrappé
dans `src/models/patchtst_official_wrapper.py`. Procédure de mise en place :
voir [`README_PatchTST_Official.md`](./README_PatchTST_Official.md).

### 3.3 Astuce d'entraînement : `init_head_zero`

Sur un dataset de cette taille (n=24), l'initialisation aléatoire de la tête de
classification provoque fréquemment un **mode collapse** vers une classe. Le flag
`init_head_zero=True` force `head.fc.bias = 0` et scale `weight ×= 0.01` à
l'initialisation, ce qui stabilise nettement l'optimisation et a été validé
empiriquement (cf. `test_phase3_2_init_bias.py`).

---

## 4. Protocole expérimental

- **LOSO** (Leave-One-Subject-Out) strict sur les 24 enfants → 24 folds.
- Train **équilibré 1:1** : retrait aléatoire d'un enfant de la classe majoritaire (11 + 11).
- **Multi-seed** : 5 seeds (`42, 1042, 2042, 3042, 4042`) pour estimer la variance d'init.
- 30 epochs fixes, pas de validation interne, AdamW + warmup + cosine, grad clip = 1.0.
- Agrégation **trait → enfant** par moyenne des probas (Exp B/C).
- Métriques : accuracy, F1, sensitivity, specificity, **AUC** (cible principale).

---

## 5. Résultats

### 5.1 Comparaison des trois expériences (5 seeds, LOSO 24 folds)

| Modèle / Entrée | Accuracy | AUC | F1 |
|---|---|---|---|
| Baseline `RICH_STATS / GB` (params agrégés) | **0.792** | 0.806 | — |
| **Exp A** — PatchTST sur params (séquence enfant) | 0.40 ± 0.08 | 0.45 ± 0.08 | 0.36 |
| **Exp B** — PatchTST sur traces brutes (`x, y`) | **0.78 ± 0.02** | **0.85 ± 0.04** | 0.81 |
| **Exp C** — PatchTST traces + params (16 canaux) | **0.80 ± 0.05** | **0.84 ± 0.01** | 0.82 |

Les fichiers `summary.csv` correspondants sont dans `results/exp_{a,b,c}_simple_loso/`.

### 5.2 Conclusions principales

1. **Exp A échoue** (AUC ≈ 0.45 ≈ chance) — le signal au niveau séquence-d'agrégats par enfant
   est trop pauvre (n=24 séquences seulement). Le PatchTST officiel donne le même résultat
   (cf. `results/exp_a_official_loso/`), confirmant que le problème est **intrinsèque au format
   d'entrée**, pas un bug d'implémentation.
2. **Exp B fonctionne** (AUC ≈ 0.85) — passer au niveau trait fournit ~660 échantillons
   d'entraînement, ce qui rend l'optimisation possible. PatchTST sur traces brutes **dépasse
   la baseline classique** sur cette métrique.
3. **Exp C ≈ Exp B** — concaténer les paramètres sigma-lognormaux comme canaux
   supplémentaires n'améliore pas significativement le modèle ; les traces brutes contiennent
   déjà l'information utile.

### 5.3 Analyse d'attention

`test_attention_analysis_loso.py` capture les poids d'attention du dernier head sur l'enfant
test (rigueur LOSO : aucune fuite). Les profils sont agrégés par classe sur 24 folds × N seeds.

Sorties (`results/attention_analysis_loso/`) :

- `loso_temporal_profile_ExpB.png` / `loso_temporal_profile_ExpC.png` — profil d'attention
  temporel moyen ± 1σ inter-seed, par classe.
- `loso_channel_importance_C.png` — importance relative des 16 canaux en Exp C.
- `loso_comparison_B_vs_C.png` — overlay des deux profils.
- `loso_attention_profiles.csv` — données numériques.

---

## 6. Structure du dépôt

```
PATCHTST_VF/
├── data/                             # (gitignoré)
│   ├── raw/                          # SXX.json (Neuroscribens)
│   └── processed/                    # params.npz, traces.npz, metadata.parquet
│
├── scripts/
│   ├── 01_preprocess.py              # JSON bruts → dataset processed
│   ├── 02_qc_visualize.py            # Visualisations QC
│   ├── 03_run_baselines.py           # LOSO toutes baselines (LDA/SVM/KNN/RF/GB)
│   └── 04_diagnose_svm_bug.py
│
├── src/
│   ├── data/
│   │   ├── exp_a_datasets.py         # SequenceChildDataset, SlidingWindowDataset
│   │   ├── exp_b_datasets.py         # RawTraceStrokeDataset
│   │   ├── exp_c_datasets.py         # RawTraceWithParamsDataset (16 canaux)
│   │   ├── loader.py, raw_traces.py, sigma_lognormal.py
│   ├── models/
│   │   ├── patchtst.py               # Implémentation custom (cls + recon)
│   │   ├── patching.py, revin.py, transformer.py
│   │   ├── patchtst_official/        # Backbone officiel (à copier, voir README dédié)
│   │   └── patchtst_official_wrapper.py
│   ├── baselines/
│   │   ├── classifiers.py            # LDA, SVM-RBF, KNN, RF, GB, LogReg
│   │   ├── features.py               # FACI4_MEAN, ALL14_MEAN_STD, RICH_STATS, STROKE_14
│   │   ├── run_child_level.py, run_stroke_level.py
│   ├── training/
│   │   ├── supervised.py, ssl_pretrain.py
│   │   ├── optimizer.py, early_stopping.py, inner_cv.py, checkpoint.py
│   │   ├── config.py, device.py
│   ├── evaluation/
│   │   ├── loso.py, metrics.py, stats.py    # Tests de permutation, Bonferroni
│   └── utils/
│       ├── data_io.py, traces_io.py, seeds.py
│
├── test_phase3_2_simple_loso_v2.py            # Exp A — PatchTST custom
├── test_phase3_official_loso.py               # Exp A — PatchTST officiel
├── test_phase_b2_simple_loso.py               # Exp B
├── test_phase_c1_simple_loso.py               # Exp C
├── test_attention_analysis_loso.py            # Analyse attention LOSO multi-seed
├── test_attention_analysis_multiseed.py       # Variante single-fold multi-seed
│
├── results/                          # CSV de sortie + figures
│   ├── exp_a_simple_loso/, exp_a_official_loso/, exp_a_cpu_validation/
│   ├── exp_b_simple_loso/
│   ├── exp_c_simple_loso/
│   ├── attention_analysis_loso/
│   └── attention_analysis_multiseed_one_child/
│
├── requirements.txt
├── README.md
└── README_PatchTST_Official.md       # Procédure d'intégration du backbone officiel
```

---

## 7. Installation

Testé sur **macOS (Apple Silicon, M5 Pro)** avec accélération **MPS**, et CPU fallback.
Python 3.10+ recommandé.

```bash
git clone https://github.com/<your-user>/PATCHTST_VF.git
cd PATCHTST_VF

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Dépendances clés (épinglées sur NumPy 1.x pour compatibilité large) :

```
numpy 1.26.x · scipy · pandas 2.x · scikit-learn 1.5
matplotlib · seaborn
torch 2.2–2.4 (MPS Apple Silicon)
pyarrow · tqdm · pyyaml
```

---

## 8. Reproduire les résultats

### 8.1 Pré-traitement (une fois)

```bash
python scripts/01_preprocess.py \
    --raw-dir data/raw \
    --out-dir data/processed \
    --snr-min 15.0 --d-max 500.0 --duration-max 3.0
```

### 8.2 Baselines classiques (LOSO)

```bash
python scripts/03_run_baselines.py --processed-dir data/processed
```
→ écrit `results/baselines/{summary.csv, detailed.parquet, pairwise_tests.csv}`.

### 8.3 PatchTST — les trois expériences

```bash
# Exp A : params sigma-lognormaux par enfant
python test_phase3_2_simple_loso_v2.py --multi-seed 5

# Exp B : traces brutes (x, y) par trait
python test_phase_b2_simple_loso.py --multi-seed 5

# Exp C : traces + params (16 canaux)
python test_phase_c1_simple_loso.py --multi-seed 5
```

Variable d'environnement utile : `KNM_DEVICE=cpu` pour forcer le CPU
(certaines opérations exotiques sont plus stables en CPU sur MPS).

### 8.4 Analyse d'attention

```bash
python test_attention_analysis_loso.py --n-seeds 3       # ~30 min, recommandé
python test_attention_analysis_loso.py --n-seeds 5       # ~50 min
```

### 8.5 Validation avec PatchTST officiel (optionnel)

Suivre [`README_PatchTST_Official.md`](./README_PatchTST_Official.md) pour copier les
fichiers du repo `yuqinie98/PatchTST`, puis :

```bash
python test_phase3_official_loso.py --multi-seed 5 --quiet
```

---

## 9. Points méthodologiques

- **Pas de fuite LOSO.** L'enfant test n'est jamais vu pendant l'entraînement, ni pour
  l'extraction d'attention.
- **Pas de normalisation hors RevIN.** RevIN par canal est appliqué *à l'intérieur* du
  modèle, sur les statistiques du batch courant uniquement.
- **Padding edge** sur traces courtes ; **truncation centrée** sur traces longues
  (préserve le pic d'accélération typiquement situé au milieu).
- **Tests de permutation** (`src/evaluation/stats.py`) avec correction Bonferroni pour les
  comparaisons par paires entre méthodes.
- **Reproductibilité** : seeding global numpy + torch + cudnn deterministic
  (`src/utils/seeds.py`).

---

## 10. Limites et travaux futurs

- **n=24** reste très faible. La variance fold-à-fold reste l'obstacle dominant ;
  un dataset plus large (KNM2 complet) est la prochaine étape.
- **SSL pretrain** implémenté (`PatchTSTReconstructor`) mais pas exploité dans les runs
  finaux faute de données non-labellisées additionnelles.
- L'analyse d'attention reste **descriptive** ; un protocole d'attribution plus rigoureux
  (Integrated Gradients, attention rollout calibré) reste à ajouter.

---

## 11. Références

- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). **A Time Series is Worth 64
  Words: Long-term Forecasting with Transformers.** *ICLR 2023.*
  Repo officiel : <https://github.com/yuqinie98/PatchTST>
- Faci, N., Plamondon, R., O'Reilly, C. (2021). **Sigma-lognormal modeling of children
  handwriting and its application to ADHD diagnosis.**
- Kim, T. *et al.* (2022). **Reversible Instance Normalization for Accurate Time-Series
  Forecasting against Distribution Shift.** *ICLR 2022.*

---

## 12. Auteur

**Rostand Njomo** — INF8225, Polytechnique Montréal — Hiver 2026.
Encadrement et données : équipe Neuroscribens (laboratoire Scribens).