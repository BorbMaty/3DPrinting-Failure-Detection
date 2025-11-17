import argparse
from pathlib import Path
from ultralytics import YOLO

from ..utils.logging import get_logger
from ..model.postprocess import draw_detections
from ..io.paths import DATASET_DIR, RUNS_DIR
import cv2

log = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser("printerfail-predict")
    ap.add_argument("--weights", type=str, required=True,
                    help="best.pt vagy más súlyfájl")
    ap.add_argument("--source", type=str, required=True,
                    help="Kép vagy mappa (jpg/png/webp).")
    ap.add_argument("--out", type=str, default=str((RUNS_DIR / "predict_out").resolve()),
                    help="Kimeneti mappa (annotált képek).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--show", action="store_true", help="OpenCV ablakban is megjeleníti.")
    args = ap.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    if src.is_dir():
        images = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
    else:
        images = [src]

    log.info(f"Infer {len(images)} file(s) | conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        # Ultralytics v8: lista; vesszük az elsőt
        r = results[0]
        im = cv2.imread(str(img_path))
        im_annot = draw_detections(im, r)
        out_file = out_dir / img_path.name
        cv2.imwrite(str(out_file), im_annot)
        if args.show:
            cv2.imshow("predict", im_annot)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()
    log.info(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
