FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY best.pt .
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/best.pt
ENV CONF_THRESHOLD=0.35
CMD ["python", "main.py"]
