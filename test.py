# Dans un terminal Python interactif ou un mini-script
from src.utils.data_io import load_processed_dataset
from pathlib import Path
ds = load_processed_dataset(Path("data/processed"))
for cid in ds.child_ids:
    print(f"{cid}: label={ds.labels_per_child[cid]}")