# teacher.py — Ultralytics 8.3+ kompatibilis tréner (cfg-alapú), metrikamentéssel és mintapredikcióval
# by Selene ✨

import argparse, json, os, sys, glob, shutil
from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO


# ---------- helpers ----------

def read_and_validate_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"data.yaml nem található: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = ["train", "val", "nc", "names"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Hiányzó kulcsok a data.yaml-ban: {missing}")

    # abszolút utakra váltás
    for k in ["train", "val"]:
        data[k] = str((path.parent / data[k]).resolve())

    # dict->lista normalizálás
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]
        data["names"] = names

    if len(data["names"]) != int(data["nc"]):
        raise ValueError(f"nc ({data['nc']}) != names hossza ({len(data['names'])})")
    return data


def pick_sample_images(root: str, limit: int = 12):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(files)[:limit]


def write_cfg(model: str, data: str, args) -> Path:
    """
    YOLO 8.3+ train cfg YAML előállítása.
    FIGYELEM: a loss-súly kulcsokat (fl_gamma, box, cls, dfl) jelen buildben nem erőltetjük,
    mert némelyik kiadásban átnevezték/áthelyezték őket. Ezek később hyp.yaml-ból adhatók.
    """
    cfg = {
        "task": "detect",
        "mode": "train",
        "model": model,
        "data": data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch if (isinstance(args.batch, int) or str(args.batch).lower() == "auto") else "auto",
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.run_name,
        "seed": args.seed,
        "patience": args.patience,
        "verbose": True,
        "plots": True,
        "exist_ok": True,
    }

    # valid, támogatott opt/hyp kulcsok
    if args.lr0 is not None:
        cfg["lr0"] = args.lr0
    if args.lrf is not None:
        cfg["lrf"] = args.lrf
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    for k in ["mosaic", "mixup", "degrees", "translate", "scale", "fliplr"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    out = Path(args.project) / "cfg"
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / f"{args.run_name}_train.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path


# ---------- main ----------

def main():
    p = argparse.ArgumentParser("YOLOv8 train – cfg alapú tréner + metrikák + mintapred")
    p.add_argument("--data", required=True, type=str, help="data.yaml abszolút útvonala")
    p.add_argument("--model", default="yolov8n.pt", type=str)
    p.add_argument("--img", dest="imgsz", default=640, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch", default="auto", help="int vagy 'auto'")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--name", default=None)
    p.add_argument("--project", default="runs")
    p.add_argument("--patience", default=50, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--resume", action="store_true")

    # opcionális – támogatott opt/hyp-ek
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--lrf", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)

    # augment
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--fliplr", type=float, default=None)

    # felhasználói kimenetek
    p.add_argument("--export", nargs="*", default=[])
    p.add_argument("--predict-n", default=12, type=int)
    p.add_argument("--conf", default=0.25, type=float)

    # elfogadjuk, de most NEM írjuk be a cfg-be (ne dobjon hibát a legújabb build)
    p.add_argument("--fl-gamma", dest="fl_gamma", type=float, default=None)
    p.add_argument("--cls-w", dest="cls", type=float, default=None)
    p.add_argument("--box-w", dest="box", type=float, default=None)
    p.add_argument("--dfl-w", dest="dfl", type=float, default=None)

    args = p.parse_args()

    data_path = Path(args.data).resolve()
    data_yaml = read_and_validate_yaml(data_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = args.name or f"yolov8_{ts}"

    # info, ha valaki megadta a loss-súlyokat
    for k in ["fl_gamma", "cls", "box", "dfl"]:
        if getattr(args, k, None) is not None:
            print(f"[INFO] Megadott {k} jelen buildben nem lesz közvetlenül alkalmazva. "
                  f"Javaslat: külön hyp.yaml-ból add meg később.")

    # modell betöltése
    model = YOLO(args.model)

    # cfg írás
    cfg_path = write_cfg(args.model, str(data_path), args)
    print(f"=== Tréning indul cfg-vel: {cfg_path} ===")

    # --- argv izoláció, hogy az Ultralytics ne olvassa be a mi CLI-nkat ---
    _saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        if args.resume:
            model.train(cfg=str(cfg_path), resume=True)
        else:
            model.train(cfg=str(cfg_path))
    finally:
        sys.argv = _saved_argv
    # ----------------------------------------------------------------------

    # valid
    print("=== Validáció ===")
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=args.project,
        name=f"{args.run_name}_val",
        save_json=False,
        verbose=True,
    )

    # kimeneti dir (detect/<name>)
    out_dir = Path(args.project) / "detect" / args.run_name
    if not out_dir.exists():
        cand_root = Path(args.project) / "detect"
        if cand_root.exists():
            cands = sorted(cand_root.glob(f"{args.run_name}*"), key=lambda p: p.stat().st_mtime)
            if cands:
                out_dir = cands[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrikák mentése
    metrics_path = out_dir / "metrics_summary.json"
    try:
        mdict = getattr(metrics, "results_dict", None)
        if mdict is None and isinstance(metrics, dict):
            mdict = metrics
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(mdict or {}, f, indent=2, ensure_ascii=False)
        print(f"Metriák mentve: {metrics_path}")
    except Exception as e:
        print(f"[WARN] Metrika mentés hiba: {e}")

    # results.csv becsatolása a futás mappába
    try:
        runs_root = Path(args.project) / "detect"
        csvs = list(runs_root.rglob("results.csv"))
        if csvs:
            shutil.copy2(csvs[-1], out_dir / "results.csv")
            print(f"results.csv bemásolva: {out_dir / 'results.csv'}")
    except Exception as e:
        print(f"[WARN] results.csv másolás hiba: {e}")

    # mintapredikciók a val képekre
    print("=== Mintapredikciók ===")
    samples = pick_sample_images(data_yaml["val"], limit=args.predict_n)
    if samples:
        pred_dir = out_dir / "sample_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        model.predict(
            source=samples, conf=args.conf, imgsz=args.imgsz, device=args.device,
            save=True, project=str(pred_dir), name=".", exist_ok=True,
            max_det=300, verbose=False
        )
        print(f"Predikciók mentve: {pred_dir}")
    else:
        print("Nincs elég valid kép a mintapredikcióhoz.")

    # export(ok)
    if args.export:
        print("=== Export ===")
        for fmt in args.export:
            try:
                exp = model.export(format=fmt, opset=12 if fmt == "onnx" else None)
                print(f"Exportálva ({fmt}): {exp}")
            except Exception as e:
                print(f"[WARN] Export sikertelen ({fmt}): {e}")

    print("=== Kész ===")
    print(f"Futás mappa: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
# teacher.py — Ultralytics 8.3+ compatible trainer (cfg-based) with metrics saving and sample predictions