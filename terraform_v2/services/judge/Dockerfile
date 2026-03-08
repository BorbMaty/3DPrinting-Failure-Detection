FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Model will be downloaded from GCS at startup or baked in
# For Vertex AI we bake it in at build time
COPY best.pt /app/best.pt

CMD ["python", "main.py"]