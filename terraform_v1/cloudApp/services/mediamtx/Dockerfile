FROM bluenviron/mediamtx:latest AS mediamtx

FROM python:3.11-slim

COPY --from=mediamtx /mediamtx /mediamtx
COPY mediamtx.yml /mediamtx.yml

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless google-cloud-pubsub

COPY extractor.py /app/extractor.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MTX_CONFIG=/mediamtx.yml
EXPOSE 8889

ENTRYPOINT ["/entrypoint.sh"]
