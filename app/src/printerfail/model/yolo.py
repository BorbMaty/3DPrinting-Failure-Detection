from ultralytics import YOLO
from pathlib import Path
from ..utils.logging import get_logger

log = get_logger(__name__)

def train(data_yaml: Path, model: str, imgsz: int, epochs: int, batch: int,
          workers: int, project: Path, seed: int | None = None, **kwargs):
    log.info(f"Training: model={model}, data={data_yaml}, imgsz={imgsz}, epochs={epochs}")
    y = YOLO(model)
    results = y.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        project=str(project),
        seed=seed,
        **kwargs
    )
    log.info("Training done.")
    return results

def predict(weights: Path, source: str | Path, conf: float = 0.25, iou: float = 0.45,
            imgsz: int = 896, save_txt: bool = True, save_conf: bool = True, **kwargs):
    log.info(f"Predict: weights={weights}, source={source}")
    y = YOLO(str(weights))
    return y.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save_txt=save_txt,
        save_conf=save_conf,
        **kwargs
    )
