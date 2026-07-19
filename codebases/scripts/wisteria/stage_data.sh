#!/bin/bash
# Stage all datasets from the LOGIN NODE (compute nodes have no internet). Idempotent.
#
#   - CIFAR-10/100 python tarballs -> codebases/resnet-cifar/data/ (torchvision layout)
#   - FineWeb 10BT GPT-2 shards: verifies the pre-staged copy at
#     /work/gs26/s26001/modded-nanogpt/data/fineweb10B (kjj0/fineweb10B-gpt2, llm.c format,
#     100M tokens/shard) and records the path; set FINEWEB_DIR to override.
#     To extend: N_CHUNKS=48 bash stage_data.sh  (downloads missing train chunks in-place).
#   - shakespeare_char bins for the legacy nanogpt task (laptop-scale E11), if absent.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENVPY="$REPO/.venv-a100/bin/python"
export LD_LIBRARY_PATH=/work/opt/local/x86_64/apps/gcc/8.3.1/python/3.11.7/lib:${LD_LIBRARY_PATH:-}

# --- CIFAR-10/100 ---------------------------------------------------------
CDATA="$REPO/codebases/resnet-cifar/data"
mkdir -p "$CDATA"
cd "$CDATA"
[ -f cifar-10-python.tar.gz ]  || curl -sLO https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[ -f cifar-100-python.tar.gz ] || curl -sLO https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
md5sum -c - <<'EOF'
c58f30108f718f92721af3b95e74349a  cifar-10-python.tar.gz
eb9058c3a382ffc7106e4002c42a8d85  cifar-100-python.tar.gz
EOF
[ -d cifar-10-batches-py ] || tar xzf cifar-10-python.tar.gz
[ -d cifar-100-python ]    || tar xzf cifar-100-python.tar.gz

# --- FineWeb 10BT (GPT-2 tokens, llm.c shards) ----------------------------
FINEWEB_DIR="${FINEWEB_DIR:-/work/gs26/s26001/modded-nanogpt/data/fineweb10B}"
n_train=$(ls "$FINEWEB_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "fineweb10B at $FINEWEB_DIR: $n_train train shards (x100M tokens) $(ls "$FINEWEB_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l) val"
if [ -n "${N_CHUNKS:-}" ] && [ "$N_CHUNKS" -gt "$n_train" ]; then
    "$VENVPY" - "$FINEWEB_DIR" "$N_CHUNKS" <<'EOF'
import sys, os
from huggingface_hub import hf_hub_download
local, n = sys.argv[1], int(sys.argv[2])
def get(fname):
    if not os.path.exists(os.path.join(local, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local)
get("fineweb_val_%06d.bin" % 0)
for i in range(1, n + 1):
    get("fineweb_train_%06d.bin" % i)
EOF
fi

# --- shakespeare_char (legacy nanogpt laptop task) ------------------------
SC="$REPO/codebases/nanogpt/upstream/data/shakespeare_char"
if [ ! -f "$SC/train.bin" ]; then
    "$VENVPY" -m pip install --no-cache-dir -q requests
    (cd "$SC" && "$VENVPY" prepare.py)
fi
echo "staging complete."
