FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements-inference.txt /app/
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY pcbparse /app/pcbparse
COPY models/detector /app/models/detector
COPY models/pins /app/models/pins
COPY models/topology /app/models/topology
COPY models/semantics /app/models/semantics
COPY models/candidates/official-pins /app/models/candidates/official-pins
COPY models/candidates/association-balanced /app/models/candidates/association-balanced
COPY models/candidates/association-mlp-balanced /app/models/candidates/association-mlp-balanced
COPY models/pipeline.json /app/models/pipeline.json
COPY models/symbols/templates.json /app/models/symbols/templates.json
COPY models/symbols/templates.npz /app/models/symbols/templates.npz
COPY run.py /app/
ENTRYPOINT ["python", "run.py"]
