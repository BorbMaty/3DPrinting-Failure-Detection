FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt || true && \
    pip install ultralytics opencv-python-headless pandas scipy 

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash"]
