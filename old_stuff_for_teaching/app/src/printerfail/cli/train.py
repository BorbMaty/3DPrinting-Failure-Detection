import argparse, yaml, os
from pathlib import Path
from ..io.paths import DATASET_DIR, RUNS_DIR
from ..model.yolo import train as yolo_train
from ..utils.logging import get_logger

log = get_logger(__name__)

def _subst_env(s: str) -> str:
    return (s.replace("${DATASET_DIR}", str(DATASET_DIR))
             .replace("${RUNS_DIR}", str(RUNS_DIR)))

def main():
    ap = argparse.ArgumentParser("printerfail-train")
    ap.add_argument("--config", default="src/printerfail/config/default.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="k=v párok (pl. train.lr0=0.001)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Alap env utak
    cfg.setdefault("paths", {})
    cfg["paths"]["dataset_dir"] = str(DATASET_DIR)
    cfg["paths"]["runs_dir"] = str(RUNS_DIR)

    # Override k=v
    for kv in args.override:
        k, v = kv.split("=", 1)
        sect, key = k.split(".", 1)
        # típusmegőrző cast
        cur = cfg.get(sect, {}).get(key, v)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        cfg.setdefault(sect, {})[key] = v

    train_cfg = dict(cfg.get("train", {}))

    # Útvonalak feloldása
    if "data_yaml" in train_cfg:
        train_cfg["data_yaml"] = Path(_subst_env(str(train_cfg["data_yaml"])))
    if "project" in train_cfg:
        train_cfg["project"] = Path(_subst_env(str(train_cfg["project"])))

    # Futás
    log.info(f"Train params: {train_cfg}")
    yolo_train(**train_cfg)

if __name__ == "__main__":
    main()
