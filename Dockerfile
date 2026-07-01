# TopoBench TDL Challenge — training/eval container.
# Exact parity with the local topobench env: Python 3.11 + torch 2.3.0+cu121 + pyg 2.8.0
# (topobench requires-python is strictly >=3.11,<3.12, so we match 3.11 deliberately).
#
# topobench itself is NOT installed in the image. The git-deployed fork is bind-mounted at
# /workspace and put on PYTHONPATH by the slurm job, so code changes ship via git (a fast
# `cluster-submit`) and never require a container rebuild. Only dependency changes do.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Python 3.11 (deadsnakes) + git (the two git+https deps) + build tools (fallback if any
# sdist compiles; the compiled deps torch-scatter/sparse/cluster ship prebuilt cu121 wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# requirements.txt carries the pytorch cu121 extra-index + pyg find-links (for torch +
# torch-scatter/sparse/cluster) and exact pins from `uv pip freeze` of the working venv.
COPY requirements.txt /app/
RUN python3.11 -m pip install -r requirements.txt

# Fail the build early if the core training stack can't import (cheap smoke test).
RUN python3.11 -c "import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster, \
lightning, hydra, topomodelx, toponetx, graph_universe; \
print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda-build', torch.version.cuda)"

CMD ["python"]
