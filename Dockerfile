# Workshop 8: federated Cox survival model (fed-survival-flower) in one image.
#
#   docker build -t hereditary-ws8 .
#   docker run --rm --shm-size=1g -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" hereditary-ws8
#
# The iDPP data is NOT in the image: mount it at /data. See DOCKER.md.

# ---- Stage 1: build the virtual environment (some packages, e.g. ecos on arm64, need gcc) --
FROM python:3.11-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /usr/local/bin/uv
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
ENV UV_NO_CACHE=1
# Exact versions and hashes from docker/requirements.lock
COPY docker/requirements.lock /tmp/requirements.lock
RUN uv venv --python /usr/local/bin/python3.11 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --require-hashes -r /tmp/requirements.lock

# ---- Stage 2: the runtime image ------------------------------------------------------------
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /usr/local/bin/uv
COPY --from=builder /opt/venv /opt/venv

# FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION: Flower 1.43 installs the app's dependencies
# into a fresh env for every run (uv sync). The image has them all, so use /opt/venv, offline.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/home/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    RAY_USAGE_STATS_ENABLED=0 \
    RAY_DISABLE_IMPORT_WARNING=1 \
    UV_NO_CACHE=1 \
    FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

# The code
WORKDIR /app
COPY fed-survival-flower/ /app/fed-survival-flower/
COPY SCHEMA.md README.md LICENSE NOTICE /app/
COPY docker/entrypoint.sh docker/set_config.py /usr/local/bin/
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /app/fed-survival-flower \
 && cp /app/fed-survival-flower/pyproject.toml /app/pyproject.orig.toml \
 && chmod +x /usr/local/bin/entrypoint.sh \
 # Smoke test at build time: every module imports
 && cd /app/fed-survival-flower \
 && python -c "import fed_survival.server_app, fed_survival.client_app, sksurv, ray, matplotlib; print('imports OK')" \
 # Non-root user, but writable for ANY uid (docker run --user "$(id -u):$(id -g)")
 && useradd --uid 1000 --create-home --home-dir /home/app app \
 && mkdir -p /app/data /outputs /data \
 && rm -rf /app/fed-survival-flower/outputs /tmp/matplotlib \
 && ln -s /outputs /app/fed-survival-flower/outputs \
 && chmod -R a+rwX /app /home/app /outputs

USER app
WORKDIR /app/fed-survival-flower
VOLUME ["/outputs"]
# 9093: the SuperLink (Deployment Engine: SuperNodes and flwr run both connect here); 8000: inference UI
EXPOSE 9093 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]
