# Dockerfile for Python-based Gene Expression AI Pipeline
FROM python:3.10-slim

# don’t buffer logs, don’t write .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# install build deps if you need any compiled packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# copy your code
COPY . .

# default command: run the full pipeline
ENTRYPOINT ["python", "pipeline.py"]
