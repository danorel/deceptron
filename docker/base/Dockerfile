ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -q uv
