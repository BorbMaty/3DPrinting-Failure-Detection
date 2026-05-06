import cv2, os, time, sys, threading

RTSP_URLS = {
    "cam1": "rtsp://192.168.1.10:8554/cam1",
    "cam2": "rtsp://192.168.1.10:8554/cam2",
    "cam3": "rtsp://192.168.1.10:8554/cam3",
}

INTERVAL = 1
OUTPUT_DIR = "dataset/raw"
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


class CamReader:
    """Background thread that continuously drains the RTSP stream, always keeping the latest frame."""

    def __init__(self, name, url):
        self.name = name
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            print(f"Warning: could not open {name} ({url})")
        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


def open_caps():
    return {name: CamReader(name, url) for name, url in RTSP_URLS.items()}


def release_caps(caps):
    for cam in caps.values():
        cam.release()
    cv2.destroyAllWindows()


def show_previews(caps):
    for cam_name, cam in caps.items():
        ret, frame = cam.read()
        if ret and frame is not None:
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            cv2.imshow(cam_name, preview)
    cv2.waitKey(1)


def get_starting_counter(folder, failure_name, cam_name):
    if not os.path.exists(folder):
        return 0
    existing = [
        f for f in os.listdir(folder)
        if f.startswith(f"{failure_name}-{cam_name}-") and f.endswith(".jpg")
    ]
    if not existing:
        return 0
    indices = [int(f.replace(f"{failure_name}-{cam_name}-", "").replace(".jpg", "")) for f in existing]
    return max(indices) + 1


def wait_for_enter(stop_event):
    input()
    stop_event.set()


def capture_session(caps, failure_name, counters):
    import termios
    termios.tcflush(sys.stdin, termios.TCIFLUSH)

    stop_event = threading.Event()
    t = threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True)
    t.start()

    print(f"Recording '{failure_name}' — press Enter to stop...")
    last_capture = time.time() - INTERVAL

    while not stop_event.is_set():
        now = time.time()

        for cam_name, cam in caps.items():
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

            if counters[cam_name] > 0:
                label = f"REC  {failure_name} | {cam_name} | #{counters[cam_name]}"
                color = (0, 0, 255)
            else:
                label = f"READY | {cam_name}"
                color = (0, 255, 0)

            cv2.putText(preview, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(cam_name, preview)

            if now - last_capture >= INTERVAL:
                folder = os.path.join(OUTPUT_DIR, cam_name, failure_name)
                os.makedirs(folder, exist_ok=True)
                filename = f"{failure_name}-{cam_name}-{counters[cam_name]:04d}.jpg"
                cv2.imwrite(os.path.join(folder, filename), frame)
                counters[cam_name] += 1

        if now - last_capture >= INTERVAL:
            last_capture = now

        cv2.waitKey(1)

    first_cam = next(iter(counters))
    print(f"Stopped. {counters[first_cam]} frames per camera for '{failure_name}'.")


def main():
    caps = open_caps()
    print("Dataset capture tool — Ctrl+C to quit.\n")

    try:
        while True:
            print("Showing previews... (windows may need focus)")
            for _ in range(20):
                show_previews(caps)
                time.sleep(0.05)

            failure_name = input("Failure class name (e.g. stringing, warping): ").strip()
            if not failure_name:
                print("Name cannot be empty.")
                continue

            counters = {}
            for cam_name in caps:
                folder = os.path.join(OUTPUT_DIR, cam_name, failure_name)
                counters[cam_name] = get_starting_counter(folder, failure_name, cam_name)
            print(f"Starting counters: { {k: v for k, v in counters.items()} }")

            input(f"Press Enter to start recording '{failure_name}'...")
            capture_session(caps, failure_name, counters)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        release_caps(caps)


if __name__ == "__main__":
    main()
