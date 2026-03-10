FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ARG UV_VERSION="0.10.9"

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.9/compat/

ENV UV_PROJECT_ENVIRONMENT="/opt/venv" \
    PATH="/opt/venv/bin:${PATH}"

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels (compatible with vLLM 0.15.1)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock /build/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /build && \
    uv sync --frozen --no-dev

# Install vLLM after syncing project dependencies so the image keeps the
# worker lockfile as the source of truth without forcing vLLM's CUDA-specific
# extra-index setup into pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python "vllm[flashinfer]==0.17.0" --extra-index-url https://download.pytorch.org/whl/cu129

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    uv pip install --python /opt/venv/bin/python -U vllm --pre --index-url https://pypi.org/simple  --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    uv pip install --python /opt/venv/bin/python git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
