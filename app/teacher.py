
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import sys
import glob

import yaml
from ultralytics import YOLO


def read_and_validate_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"data.yaml nem található: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = ["train", "val", "nc", "names"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Hiányzó kulcsok a data.yaml-ban: {missing}")

    # Abszolút utakra váltás (YOLO mindkettőt tudja, de legyen robusztus)
    for k in ["train", "val"]:
        data[k] = str((path.parent / data[k]).resolve())

    # Konszisztencia-ellenőrzés
    names = data["names"]
    if isinstance(names, dict):  # egyes exportoknál dict lehet
        # rendezés index szerint, majd lista
        names = [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]
        data["names"] = names

    if len(data["names"]) != int(data["nc"]):
        raise ValueError(
            f"nc ({data['nc']}) != names hossza ({len(data['names'])}). "
            f"Javítsd a data.yaml-t!"
        )

    return data


def pick_sample_images(root: str, limit: int = 12):
    # Összes képfájl rekurzívan; első N-et visszaadjuk
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    files = sorted(files)
    return files[:limit]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 tréning szkript (Ultralytics)")
    parser.add_argument("--data", type=str, required=True, help="data.yaml elérési út")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Kezdő modell (pl. yolov8n.pt) vagy konfig (yolov8n.yaml)")
    parser.add_argument("--img", type=int, default=640, help="Képméret (imgsz)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochok száma")
    parser.add_argument("--batch", type=int, default="auto", help="Batch méret (int vagy 'auto')")
    parser.add_argument("--device", type=str, default="auto", help="GPU index (pl. '0') vagy 'cpu'/'auto'")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--name", type=str, default=None, help="Futás neve")
    parser.add_argument("--project", type=str, default="runs", help="Projekt mappa a YOLO számára")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Véletlenmag a reprodukálhatósághoz")
    parser.add_argument("--resume", action="store_true", help="Tréning folytatása az utolsó checkpointból")
    parser.add_argument("--lr0", type=float, default=None, help="Kezdő tanulási ráta (opcionális)")
    parser.add_argument("--lrf", type=float, default=None, help="Végső tanulási ráta szorzó (opcionális)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Súlycsökkenés (opcionális)")
    parser.add_argument("--export", nargs="*", default=[],
                        choices=["onnx", "torchscript", "openvino", "engine", "xml", "pb"],
                        help="Model export formátum(ok) a tréning végén")
    parser.add_argument("--predict-n", type=int, default=12, help="Ennyi validációs képre csinál mintapredikciót")
    parser.add_argument("--conf", type=float, default=0.25, help="Konfidencia küszöb a predikcióhoz")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    data = read_and_validate_yaml(data_path)

    # Futásnév
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"yolov8_{timestamp}"

    # Modell betöltése
    model = YOLO(args.model)

    # Tréning argumentumok összerakása
    train_kwargs = dict(
        data=str(data_path),
        imgsz=args.img,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        seed=args.seed,
        patience=args.patience,
        plots=True,        # loss/metric ábrák mentése
        exist_ok=True,     # ne dobjon hibát azonos név esetén
        verbose=True,
    )

    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay

    # Tréning
    print("=== Tréning indul ===")
    if args.resume:
        # Ha folytatsz, a .train-ben a resume=True is működik, de akkor a legtöbb paramétert az előző futásból veszi át
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    # Validáció
    print("=== Validáció ===")
    metrics = model.val(
        data=str(data_path),
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,   # confusion_matrix, PR görbe stb. mentése
        project=args.project,
        name=f"{run_name}_val",
        save_json=False,
        verbose=True,
    )

    # Metriák mentése JSON-ba az egyszerű feldolgozáshoz
    out_dir = Path(args.project) / "detect" / run_name  # Ultralytics detect/train alapértelmezett struktúra
    if not out_dir.exists():
        # alternatíva: újabb verzióknál 'runs/detect/train' -> 'runs/detect/<name>'
        # biztos, ami biztos, keressük meg a legutóbbi train mappát a név alapján
        candidates = sorted((Path(args.project) / "detect").glob(f"{run_name}*"), key=os.path.getmtime)
        if candidates:
            out_dir = candidates[-1]

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_summary.json"
    try:
        # metrics általában egy objektum, .results_dict vagy hasonló elérhető
        mdict = getattr(metrics, "results_dict", None)
        if mdict is None and isinstance(metrics, dict):
            mdict = metrics
        if mdict is None:
            mdict = {"note": "Nem találtam results_dict-et; YOLO verziótól függhet."}
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(mdict, f, ensure_ascii=False, indent=2)
        print(f"Metriák mentve: {metrics_path}")
    except Exception as e:
        print(f"Figyelem: a metrikák mentése közben hiba történt: {e}", file=sys.stderr)

    # Mintapredikciók a val képekről
    print("=== Mintapredikciók ===")
    sample_imgs = pick_sample_images(data["val"], limit=args.predict_n)
    if sample_imgs:
        pred_dir = out_dir / "sample_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        _ = model.predict(
            source=sample_imgs,
            conf=args.conf,
            imgsz=args.img,
            device=args.device,
            save=True,
            project=str(pred_dir),
            name=".",
            exist_ok=True,
            max_det=300,
            verbose=False,
        )
        print(f"Mintapredikciók mentve ide: {pred_dir}")
    else:
        print("Nem találtam képeket a val mappában mintapredikcióhoz.")

    # Export(ok)
    if args.export:
        print("=== Export ===")
        for fmt in args.export:
            try:
                exp_path = model.export(format=fmt, opset=12 if fmt == "onnx" else None)
                print(f"Exportálva ({fmt}): {exp_path}")
            except Exception as e:
                print(f"Export sikertelen ({fmt}): {e}", file=sys.stderr)

    print("=== Kész ===")
    print(f"Futás mappa: {out_dir.resolve()}")
    print("Hasznos fájlok: results.csv, results.png, confusion_matrix.png, PR görbe(k), metrics_summary.json, sample_preds/ ...")


if __name__ == "__main__":
    main()