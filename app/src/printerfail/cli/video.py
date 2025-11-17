import argparse
from ultralytics import YOLO
from ..utils.logging import get_logger
from ..model.postprocess import draw_detections
from ..io.video import VideoReader, VideoWriter

log = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser("printerfail-video")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True, help="Video input (mp4/avi) vagy kamera index (pl. 0).")
    ap.add_argument("--out", type=str, required=True, help="Kimeneti videó útvonala (mp4/avi).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # modell
    log.info(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # video I/O
    reader = VideoReader(args.source)
    writer = VideoWriter(args.out, fps=reader.fps, width=reader.width, height=reader.height)

    log.info(f"Video infer: {reader.width}x{reader.height} @ {reader.fps:.2f}fps")
    for frame in reader:
        # Egyetlen képkocka inferencia (numpy frame)
        res = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        out = draw_detections(frame, res)
        writer.write(out)
        if args.show:
            reader.show(out)
            if reader.key_pressed(27):  # ESC
                break

    reader.release()
    writer.release()
    log.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
