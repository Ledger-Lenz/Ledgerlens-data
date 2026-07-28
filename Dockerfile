FROM python:3.12-slim

WORKDIR /app

COPY requirements.in ./
COPY requirements/ requirements/
COPY scripts/validate_lockfiles.py scripts/validate_lockfiles.py
RUN python scripts/validate_lockfiles.py && \
    pip install --no-cache-dir --require-hashes \
      -r requirements/requirements-linux-py312.txt && \
    pip check

COPY . .

ENTRYPOINT ["python", "run_pipeline.py"]
