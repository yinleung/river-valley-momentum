#!/bin/bash
# Wisteria login-node environment bootstrap (plan_v5 §3.1 B1). Idempotent.
#
# Builds the campaign venv at <repo>/.venv-a100 from the python/3.11.7 module with
# torch cu12x wheels from PyPI (they bundle CUDA runtime + cuDNN + NCCL; the A100
# node driver 570.148.08 supports CUDA <= 12.8). Compute nodes have no internet:
# run this ON THE LOGIN NODE only. Job scripts then only `module load` + activate.
#
# Usage:  bash codebases/scripts/wisteria/setup_env.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="$REPO/.venv-a100"

source /etc/profile.d/modules.sh
module purge
module load gcc/8.3.1 python/3.11.7
export LD_LIBRARY_PATH=/work/opt/local/x86_64/apps/gcc/8.3.1/python/3.11.7/lib:${LD_LIBRARY_PATH:-}

if [ ! -x "$VENV/bin/python" ]; then
    python3.11 -m venv "$VENV"
fi
# --no-cache-dir: /home quota is nearly full; never cache wheels there.
# torch from the cu128 index: plain PyPI serves cu130 builds, which need a >=580
# driver — the A100 nodes run 570.148.08 (CUDA <= 12.8). cu128 needs >= 570.26.
"$VENV/bin/pip" install --no-cache-dir -U pip
"$VENV/bin/pip" install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install --no-cache-dir numpy scipy pyyaml requests huggingface_hub tiktoken

"$VENV/bin/python" - <<'EOF'
import torch, torchvision, numpy, scipy
print("torch", torch.__version__, "| cuda built", torch.version.cuda,
      "| torchvision", torchvision.__version__, "| numpy", numpy.__version__)
EOF

# Environment lockfile for the public artifact (plan §5): exact wheel versions.
"$VENV/bin/pip" freeze > "$REPO/codebases/scripts/wisteria/requirements.lock"
echo "venv ready at $VENV; lockfile written."
