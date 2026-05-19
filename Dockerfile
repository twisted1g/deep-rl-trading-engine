FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# CPU-only torch — без CUDA-колёс (~6 GB меньше), inference на CPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

# Submodule с research-кодом (env/encoders/agents). Должен быть
# инициализирован на хосте: `git submodule update --init --recursive`.
COPY vendor ./vendor
COPY src ./src
COPY run.py config.yaml ./

VOLUME ["/app/state"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "run.py"]
