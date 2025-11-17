from pathlib import Path
import os

def get_dir_from_env(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).resolve()

DATASET_DIR = get_dir_from_env("DATASET_DIR", "./dataset")
RUNS_DIR    = get_dir_from_env("RUNS_DIR", "./runs")
ROOT        = Path(__file__).resolve().parents[2]  # repo gyökér (src/printerfail/..)
