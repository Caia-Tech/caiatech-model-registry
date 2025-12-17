FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server.py /app/server.py
COPY ingest.py /app/ingest.py
COPY publish_checkpoint.py /app/publish_checkpoint.py
COPY bench.py /app/bench.py
COPY maintenance.py /app/maintenance.py
COPY smoke_test.py /app/smoke_test.py

ENV CAIA_REGISTRY_DB_PATH=/data/registry.db
EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]

