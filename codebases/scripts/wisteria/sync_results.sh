#!/bin/bash
# Pull run records from Wisteria to the laptop (run this ON THE LAPTOP).
#
# The canonical sync path is git: Wisteria sessions commit results/cache/ + results/index/
# and push to origin; `git pull` on the laptop brings them home. This script is the
# supplementary rsync for (a) peeking at records before a session commits, and (b) pulling
# selected raw diagnostic windows (results/raw/ stays on /work and out of git; see
# discussions/wisteria_notes.md §3 for paths).
#
# Usage:
#   sync_results.sh user@wisteria.cc.u-tokyo.ac.jp                  # cache + index
#   sync_results.sh user@wisteria.cc.u-tokyo.ac.jp --raw <run-id>   # one raw-window dir
set -euo pipefail
HOST="${1:?usage: sync_results.sh user@host [--raw run-id]}"
REMOTE_REPO="/work/02/gs26/s26001/river-valley-momentum"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "${2:-}" = "--raw" ]; then
    RID="${3:?run-id required after --raw}"
    rsync -av "$HOST:$REMOTE_REPO/codebases/results/raw/$RID/" \
        "$LOCAL_REPO/codebases/results/raw/$RID/"
else
    rsync -av \
        "$HOST:$REMOTE_REPO/codebases/results/cache/" "$LOCAL_REPO/codebases/results/cache/"
    rsync -av \
        "$HOST:$REMOTE_REPO/codebases/results/index/" "$LOCAL_REPO/codebases/results/index/"
fi
