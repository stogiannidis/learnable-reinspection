FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_HOME=/data/users/stogian/.torch \
    HF_HOME=/data/Huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DNNLIB_CACHE_DIR=/data/users/stogian/.cache/dnnlib \
    TORCH_EXTENSIONS_DIR=/data/users/stogian/.torch/torch_extensions \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    TORCHINDUCTOR_CACHE_DIR=/data/users/stogian/.cache/torch_compile \
    XDG_CACHE_HOME=/data/users/stogian/.cache \
    USER=stogian_ph \
    TORCH_KERNEL_CACHE_PATH=/data/users/stogian/.torch/kernels \
    PYTORCH_KERNEL_CACHE_PATH=/data/users/stogian/.torch/kernels \
    TRITON_CACHE_DIR=/data/users/stogian/.cache/triton \
    TOKENIZERS_PARALLELISM=false \
    NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/data/users/stogian/learnable-reinspection

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        curl \
        git \
        build-essential && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl -Ls https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data/users/stogian/learnable-reinspection

COPY requirements.txt .

RUN uv pip install --system --no-cache --break-system-packages -r requirements.txt

COPY src/ src/

RUN mkdir -p \
        /data/Huggingface \
        /data/users/stogian/.torch \
        /data/users/stogian/.cache/dnnlib \
        /data/users/stogian/.torch/torch_extensions \
        /data/users/stogian/.cache/torch_compile \
        /data/users/stogian/.cache \
        /data/users/stogian/.torch/kernels \
        /data/users/stogian/.cache/triton && \
    groupadd --gid 4451 stogian_ph && \
    useradd -u 30545 -g 4451 -m -s /bin/bash stogian_ph && \
    chown -R stogian_ph /data/users/stogian /data/Huggingface

RUN chown -R stogian_ph /data/users/stogian/learnable-reinspection

USER stogian_ph

CMD ["bash"]
