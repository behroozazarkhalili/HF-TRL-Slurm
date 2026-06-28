#!/bin/bash
# Recovery sequence for the OpenEnv smoke after the /scratch quota failure +
# login-node fork starvation. RUN THIS ON A FRESH (non-starved) LOGIN NODE.
#
# It (1) confirms the node can fork + measure FS, (2) measures /scratch quota,
# (3) removes the partial openenv_env venv that hit the quota wall, (4) relocates
# the venv target to /project (more headroom, survives /scratch purges), (5) leaves
# the resubmit to a deliberate manual step.
#
# Usage:  bash scripts/tmp/openenv_recover_on_new_node.sh
set -uo pipefail

echo "=== 0. node + fork sanity ==="
echo "host: $(hostname)   user: $(id -un)"
true && echo "[ok] fork works on this node"

echo ""
echo "=== 1. /scratch quota (fast lustre MDS query) ==="
timeout 30 lfs quota -hu "$USER" /scratch 2>&1 | head -6 || echo "lfs quota unavailable; trying diskusage_report"
timeout 30 diskusage_report 2>&1 | grep -iE "scratch|project|Filesystem" | head -6 || true

echo ""
echo "=== 2. partial venv that hit the wall (size + presence) ==="
if [[ -d /scratch/$USER/venvs/openenv_env ]]; then
    timeout 60 du -sh /scratch/$USER/venvs/openenv_env 2>&1 || echo "(du slow; dir exists)"
else
    echo "no /scratch/$USER/venvs/openenv_env (already gone or never created here)"
fi

echo ""
echo "=== 3. biggest /scratch reclaim candidates (top dirs) ==="
timeout 90 du -sh /scratch/$USER/* 2>/dev/null | sort -h | tail -15 || echo "(du timed out)"

echo ""
echo "=== DONE — review above before deleting anything ==="
echo "Next (manual, after review):"
echo "  rm -rf /scratch/$USER/venvs/openenv_env        # remove partial venv"
echo "  # then relocate venv to /project via the patched runner (see below)"
