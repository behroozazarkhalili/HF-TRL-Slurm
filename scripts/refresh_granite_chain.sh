#!/bin/bash
# =============================================================================
# Refresh atime/mtime on granite-chain artifact dirs to defer DRAC's 60-day
# /scratch purge while the dependency chain (stage-3 → 4 → 5) waits in the
# b5 queue. Idempotent; safe to run nightly. Writes a heartbeat to logs/.
# =============================================================================

set -euo pipefail

LOG_DIR="/project/6014832/ermia/HF-TRL/logs"
LOG_FILE="$LOG_DIR/refresh_granite_chain.log"
mkdir -p "$LOG_DIR"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

# Source of truth: the adapter pointer files in granite-chain/
CHAIN_DIR="/scratch/ermia/outputs/granite-chain"

# Paths to refresh — derive dynamically from the adapter pointers, plus the
# chain dir itself. If a pointer's target is gone, log it but don't fail.
DIRS=("$CHAIN_DIR")
if [[ -d "$CHAIN_DIR" ]]; then
    while IFS= read -r ptr; do
        target=$(<"$ptr")
        if [[ -d "$target" ]]; then
            DIRS+=("$target")
        else
            log "WARN: adapter pointer $ptr -> $target (missing)"
        fi
    done < <(find "$CHAIN_DIR" -maxdepth 1 -name 'stage-*.adapter' -type f 2>/dev/null)
fi

# Quick exit if chain dir is gone (chain done + cleaned up; nothing to refresh)
if [[ ! -d "$CHAIN_DIR" ]]; then
    log "INFO: $CHAIN_DIR not present — chain cleaned up. Sentinel can be removed."
    exit 0
fi

total_touched=0
for d in "${DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        n=$(find "$d" -type f 2>/dev/null | wc -l)
        find "$d" -type f -exec touch -a -m {} + 2>/dev/null || true
        log "touched $n files in $d"
        total_touched=$((total_touched + n))
    fi
done

# Compute the projected purge date (60 days from now) for the dirs we just refreshed
purge_date=$(date -d "+60 days" +%Y-%m-%d 2>/dev/null || date -v+60d +%Y-%m-%d)
log "DONE: refreshed ${#DIRS[@]} dir(s), $total_touched file(s). Purge deferred to $purge_date."

# Health check: warn if any pending chain stages are still in the queue
if command -v squeue >/dev/null 2>&1; then
    pending=$(squeue -u "$USER" -h --format='%j %T' 2>/dev/null | grep -cE 'granite-chain.*PENDING' || true)
    running=$(squeue -u "$USER" -h --format='%j %T' 2>/dev/null | grep -cE 'granite-chain.*RUNNING' || true)
    log "queue: pending=$pending running=$running"
    if (( pending == 0 && running == 0 )); then
        log "INFO: no granite-chain jobs in queue — chain may be done. Consider removing this sentinel."
    fi
fi
