FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

WORKDIR /opt/program

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY container-requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/program/requirements.txt

COPY app /opt/program/app

EXPOSE 8080

# SageMaker starts custom containers as `docker run <image> serve`. With only
# a CMD, that `serve` argument replaces gunicorn and the container exits with
# "entrypoint executable serve not found". An ENTRYPOINT via sh -c absorbs the
# extra argument (it lands in $0 and is ignored).
ENTRYPOINT ["/bin/sh", "-c", "exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 120 --chdir /opt/program app.server:app"]
