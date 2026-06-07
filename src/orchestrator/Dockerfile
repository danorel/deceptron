# Image for dagster-webserver + dagster-daemon (see docker-compose.yml).
# Bundles deceptron itself so orchestration ops can `import` the pipeline's
# run_query/run_repos functions directly — no subprocess, no duplicated CLI.
FROM python:3.11-slim

WORKDIR /opt/dagster/app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dagster dagster-webserver

COPY src/ ./src/
COPY workspace.yaml dagster.yaml ./
RUN pip install --no-cache-dir -e .

ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p "$DAGSTER_HOME" && cp dagster.yaml "$DAGSTER_HOME/dagster.yaml"
