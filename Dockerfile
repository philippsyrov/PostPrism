# PostPrism GPU image. Model weights are not baked in: the Hugging Face models
# download on first start into /data/hf, and the RAM++ checkpoint downloads into
# /data/artifacts (hash-checked) when USE_VISUAL_TAGS=true. Mount /data as a volume
# so this happens once. See README "Run with Docker" and docs/DEPLOYMENT.md.
FROM nvidia/cuda:12.9.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SERVICE_COMMIT=unknown
ARG RAM_SOURCE_COMMIT=7cb804a8609e9f4b1a50b7f31436d2df40bb9481
ARG BERT_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git gnupg \
    && echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN python3.11 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir pip==26.2.1 setuptools==84.0.0 wheel==0.48.0

WORKDIR /app
COPY requirements.txt requirements-visual-tags.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -r requirements-visual-tags.txt

COPY . .

# Build and install the pinned RAM++ package, and fetch its small tokenizer files.
# The build script refuses any source other than the pinned commit.
# hadolint ignore=DL3064
ENV RAM_TOKENIZER_DIR=/opt/ram/bert-base-uncased
# hadolint ignore=DL3013
RUN git clone https://github.com/xinyu1205/recognize-anything /tmp/recognize-anything \
    && git -C /tmp/recognize-anything checkout --detach "$RAM_SOURCE_COMMIT" \
    && RAM_WHEEL="$(python -m scripts.build_ram_package --source-dir /tmp/recognize-anything --output-dir /tmp/ram-build)" \
    && python -m pip install --no-cache-dir --no-deps --no-index "$RAM_WHEEL" \
    && mkdir -p "$RAM_TOKENIZER_DIR" \
    && for f in vocab.txt tokenizer_config.json tokenizer.json; do \
        curl -fsSL -o "$RAM_TOKENIZER_DIR/$f" \
            "https://huggingface.co/google-bert/bert-base-uncased/resolve/$BERT_REVISION/$f"; \
    done \
    && cp packaging/ram/tokenizer-artifacts.json "$RAM_TOKENIZER_DIR/" \
    && rm -rf /tmp/recognize-anything /tmp/ram-build \
    && python -m pip check

ENV SERVICE_COMMIT=$SERVICE_COMMIT \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/data/hf \
    POSTPRISM_ARTIFACTS=/data/artifacts \
    RAM_CHECKPOINT_FILE=/data/artifacts/ram_plus_swin_large_14m.pth \
    USE_ASR=true \
    USE_VISUAL_TAGS=true \
    USE_CAPTIONS=true

VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["/app/packaging/docker/entrypoint.sh"]
CMD ["python", "app.py"]
