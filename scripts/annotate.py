#!/usr/bin/env python3
"""
cvat_auto_annotate.py - Auto-annotate CVAT tasks using local YOLOv8x weights.

Usage:
    python3 annotate.py --weights /home/lucy/ThesisTest/3DPrinting-Failure-Detection/terraform_v2/services/judge/best.pt \
        --cvat-url http://localhost:8080 \
        --username mara --password alma \
        --task-id 23
"""

import argparse
import sys
import tempfile
from pathlib import Path

import requests
from ultralytics import YOLO

CLASS_NAMES = [
    "stringing",
    "warping",
    "layer_shift",
    "under_extrusion",
    "over_extrusion",
    "nozzle_clog",
    "foreign_object_on_print_area",
    "not_sticking",
    "spagetti",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--cvat-url", default="http://localhost:8080")
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


class CVATClient:
    def __init__(self, base_url, username, password):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self._login(username, password)

    def _login(self, username, password):
        # Step 1: hit the login page to get csrftoken cookie
        r = self.session.get(f"{self.base}/api/auth/login", headers={"Accept": "application/json"})
        # extract csrftoken from cookies if present
        csrf = self.session.cookies.get("csrftoken", "")

        headers = {"Content-Type": "application/json"}
        if csrf:
            headers["X-CSRFToken"] = csrf

        # Step 2: POST credentials
        r = self.session.post(
            f"{self.base}/api/auth/login",
            json={"username": username, "password": password},
            headers=headers,
        )
        if r.status_code not in (200, 201):
            print(f"  Login response {r.status_code}: {r.text[:300]}")
            r.raise_for_status()

        data = r.json()
        # CVAT returns a token key
        token = data.get("key") or data.get("token")
        if token:
            self.session.headers.update({"Authorization": f"Token {token}"})
            print(f"Logged in (token auth)")
        else:
            # Session cookie auth — refresh csrf
            csrf = self.session.cookies.get("csrftoken", "")
            if csrf:
                self.session.headers.update({"X-CSRFToken": csrf})
            print(f"Logged in (session auth)")

    def _get_csrf(self):
        csrf = self.session.cookies.get("csrftoken", "")
        if csrf:
            self.session.headers.update({"X-CSRFToken": csrf})

    def get(self, path, **kwargs):
        r = self.session.get(f"{self.base}{path}", **kwargs)
        r.raise_for_status()
        return r.json()

    def get_bytes(self, path, **kwargs):
        r = self.session.get(f"{self.base}{path}", **kwargs)
        r.raise_for_status()
        return r.content

    def put(self, path, **kwargs):
        self._get_csrf()
        r = self.session.put(f"{self.base}{path}", **kwargs)
        if not r.ok:
            print(f"  PUT {path} → {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r

    def get_tasks(self):
        return self.get("/api/tasks?page_size=500")["results"]

    def get_task(self, task_id):
        return self.get(f"/api/tasks/{task_id}")

    def get_labels(self, task_id):
        data = self.get(f"/api/labels?task_id={task_id}&page_size=200")
        return {l["name"]: l["id"] for l in data["results"]}

    def get_annotations(self, task_id):
        return self.get(f"/api/tasks/{task_id}/annotations")

    def get_frame(self, task_id, frame_idx):
        return self.get_bytes(
            f"/api/tasks/{task_id}/data",
            params={"type": "frame", "number": frame_idx, "quality": "original"},
        )

    def put_annotations(self, task_id, payload):
        return self.put(f"/api/tasks/{task_id}/annotations", json=payload)


def is_unlabeled(client, task_id):
    ann = client.get_annotations(task_id)
    return len(ann.get("shapes", [])) == 0 and len(ann.get("tags", [])) == 0


def run_inference(model, image_path, conf):
    results = model(image_path, conf=conf, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        detections.append((cls, x1, y1, x2, y2, score))
    return detections, results.orig_shape


def annotate_task(client, model, task, conf, dry_run):
    task_id = task["id"]
    print(f"\n── Task {task_id}: '{task['name']}' ({task['size']} frames) ──")

    label_id_map = client.get_labels(task_id)
    print(f"  CVAT labels: {list(label_id_map.keys())}")
    if not label_id_map:
        print("  [skip] No labels on this task.")
        return

    all_shapes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for frame_idx in range(task["size"]):
            frame_path = tmpdir / f"frame_{frame_idx:06d}.jpg"
            frame_bytes = client.get_frame(task_id, frame_idx)
            frame_path.write_bytes(frame_bytes)

            detections, orig_shape = run_inference(model, str(frame_path), conf)

            if detections:
                names = [CLASS_NAMES[d[0]] for d in detections if d[0] < len(CLASS_NAMES)]
                print(f"  frame {frame_idx:4d}: {len(detections)} → {', '.join(names)}")
            else:
                print(f"  frame {frame_idx:4d}: —")

            for cls_idx, x1, y1, x2, y2, score in detections:
                if cls_idx >= len(CLASS_NAMES):
                    continue
                label_name = CLASS_NAMES[cls_idx]
                if label_name not in label_id_map:
                    print(f"  [warn] '{label_name}' not in CVAT labels, skipping")
                    continue
                all_shapes.append({
                    "type": "rectangle",
                    "label_id": label_id_map[label_name],
                    "frame": frame_idx,
                    "points": [x1, y1, x2, y2],
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "group": 0,
                    "attributes": [],
                    "source": "auto",
                })

    if dry_run:
        print(f"  [dry-run] Would upload {len(all_shapes)} shapes.")
        return

    if not all_shapes:
        print("  No detections — nothing uploaded.")
        return

    client.put_annotations(task_id, {
        "version": 0,
        "tags": [],
        "shapes": all_shapes,
        "tracks": [],
    })
    print(f"  ✓ Uploaded {len(all_shapes)} shapes.")


def main():
    args = parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    print(f"Connecting to CVAT at {args.cvat_url} …")
    client = CVATClient(args.cvat_url, args.username, args.password)

    if args.task_id is not None:
        tasks = [client.get_task(args.task_id)]
    else:
        all_tasks = client.get_tasks()
        tasks = [t for t in all_tasks if is_unlabeled(client, t["id"])]
        print(f"Found {len(tasks)} unlabeled task(s) out of {len(all_tasks)} total.")

    if not tasks:
        print("Nothing to annotate.")
        sys.exit(0)

    for task in tasks:
        try:
            annotate_task(client, model, task, args.conf, args.dry_run)
        except Exception as e:
            import traceback
            print(f"  [error] Task {task['id']} failed: {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()

#usage 
