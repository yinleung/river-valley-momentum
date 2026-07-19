#!/bin/bash
#PJM -N @NAME@
#PJM -L rscgrp=@RSCGRP@
#PJM -L node=1
#PJM -L elapse=@ELAPSE@
#PJM -g gs26
#PJM -j
#PJM -S
#PJM -o @LOG@
# Template: full-node 8-GPU DDP job (regular-a -> small-a for 1 node, 48 h ceiling;
# token charge = 8 x elapsed h x 1.5). Filled by submit.sh. The payload usually launches
# `torchrun --standalone --nproc_per_node=8 <driver>`.

set -euo pipefail
REPO=@REPO@

module purge
module load gcc/8.3.1 python/3.11.7
export LD_LIBRARY_PATH=/work/opt/local/x86_64/apps/gcc/8.3.1/python/3.11.7/lib:${LD_LIBRARY_PATH:-}
source "$REPO/.venv-a100/bin/activate"

export XDG_CACHE_HOME="$REPO/.cache"
export TORCH_HOME="$REPO/.cache/torch"
export TRITON_CACHE_DIR="$REPO/.cache/triton"
export HF_HOME="$REPO/.cache/hf"
export MPLCONFIGDIR="$REPO/.cache/mpl"
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single-node DDP over NVSwitch; socket family pinned per Leon's working scripts.
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

echo "== node $(hostname) | job ${PJM_JOBID:-?} | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "== python $(which python)"

cd "$REPO/codebases"
@PAYLOAD@
