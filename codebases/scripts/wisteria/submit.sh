#!/bin/bash
# Generate-and-submit wrapper for pjsub (plan_v5 §3.1 B1).
#
# Fills a template (@NAME@/@RSCGRP@/@GPUS@/@ELAPSE@/@LOG@/@REPO@/@PAYLOAD@), archives the
# concrete script beside its log under results/joblogs/, submits it, and appends one row to
# results/joblogs/submissions.tsv — so every submission is auditable after the fact.
#
# Usage:
#   submit.sh -n NAME [-t share|node] [-g GPUS] [-e ELAPSE] [-r RSCGRP] -- PAYLOAD...
#     -t share (default): share-family template, 1 node, GPUS gpus (rscgrp share|share-short|share-debug)
#     -t node:            regular-a full-node template (8 GPUs, torchrun payloads)
#   Defaults: -g 1, -e 0:30:00, -r share (share template) / regular-a (node template).
# Examples:
#   submit.sh -n smoke -r share-debug -e 0:20:00 -- python scripts/wisteria/smoke_gpu.py
#   submit.sh -n g1_scan -e 6:00:00 -- python scripts/run_g1_resnet.py --stage scan
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TPL_DIR="$REPO/codebases/scripts/wisteria"
JOBDIR="$REPO/codebases/results/joblogs"
mkdir -p "$JOBDIR/jobs"

kind=share; gpus=1; elapse=0:30:00; rscgrp=""; name=""
while [ $# -gt 0 ]; do
    case "$1" in
        -t) kind="$2"; shift 2 ;;
        -g) gpus="$2"; shift 2 ;;
        -e) elapse="$2"; shift 2 ;;
        -r) rscgrp="$2"; shift 2 ;;
        -n) name="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$name" ] || { echo "submit.sh: -n NAME is required" >&2; exit 2; }
[ $# -gt 0 ] || { echo "submit.sh: payload after -- is required" >&2; exit 2; }
payload="$*"

case "$kind" in
    share) tpl="$TPL_DIR/tpl_share_1gpu.sh"; [ -n "$rscgrp" ] || rscgrp=share ;;
    node)  tpl="$TPL_DIR/tpl_node_8gpu.sh";  [ -n "$rscgrp" ] || rscgrp=regular-a ;;
    *) echo "submit.sh: -t must be share or node" >&2; exit 2 ;;
esac

stamp="$(date +%Y%m%d_%H%M%S)"
script="$JOBDIR/jobs/${name}.${stamp}.sh"
log="$JOBDIR/${name}.${stamp}.log"

tplsrc="$(cat "$tpl")"
tplsrc="${tplsrc//@NAME@/$name}"
tplsrc="${tplsrc//@RSCGRP@/$rscgrp}"
tplsrc="${tplsrc//@GPUS@/$gpus}"
tplsrc="${tplsrc//@ELAPSE@/$elapse}"
tplsrc="${tplsrc//@LOG@/$log}"
tplsrc="${tplsrc//@REPO@/$REPO}"
tplsrc="${tplsrc//@PAYLOAD@/$payload}"
printf '%s\n' "$tplsrc" > "$script"

out="$(pjsub "$script" 2>&1)" || { echo "pjsub failed: $out" >&2; exit 1; }
jobid="$(printf '%s' "$out" | grep -oE '[0-9]+' | head -1)"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stamp" "$name" "$rscgrp" "$gpus" "$elapse" "$jobid" "$payload" >> "$JOBDIR/submissions.tsv"
echo "submitted $name -> job $jobid (rscgrp=$rscgrp gpus=$gpus elapse=$elapse)"
echo "  script $script"
echo "  log    $log"
