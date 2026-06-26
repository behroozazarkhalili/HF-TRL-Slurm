#!/bin/bash
# Robust SLURM job state probe for use inside Monitor poll loops.
#
# Resolves a job's state by querying squeue first, then sacct, with explicit
# handling for transient accounting-database failures so a flaky `sacct` never
# masquerades as PENDING. Prints exactly one token to stdout:
#
#   PENDING | RUNNING | COMPLETED | FAILED | CANCELLED | TIMEOUT |
#   OUT_OF_MEMORY | NODE_FAIL | BOOT_FAIL | UNKNOWN_TERMINAL | SACCT_DOWN
#
# UNKNOWN_TERMINAL: not in squeue and sacct returns empty (exit 0). Job is
#                   gone but state can't be confirmed; treat as terminal so
#                   the loop exits and the caller can read filesystem signals.
# SACCT_DOWN:       sacct exited non-zero (DB unreachable). Caller should
#                   retry, NOT conflate with PENDING.
#
# Usage: scripts/slurm_state.sh <jobid>
set -u
JID="${1:?usage: slurm_state.sh JOBID}"

# 1) squeue is authoritative for live jobs and never queries the accounting DB.
SQ_STATE=$(squeue -h -j "$JID" -o "%T" 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "$SQ_STATE" ]]; then
    echo "$SQ_STATE"
    exit 0
fi

# 2) Not in queue. Ask sacct, but distinguish empty-output from DB-down.
SACCT_OUT=$(sacct -j "$JID" --format=State --noheader --parsable2 2>&1)
SACCT_RC=$?
if (( SACCT_RC != 0 )) || grep -qE "slurm_persist_conn_open|Resource temporarily unavailable|Problem talking to the database" <<<"$SACCT_OUT"; then
    echo "SACCT_DOWN"
    exit 0
fi

# 3) sacct healthy, but no row -> job is gone and accounting forgot it.
STATE=$(printf '%s\n' "$SACCT_OUT" | head -1 | tr -d ' ')
if [[ -z "$STATE" ]]; then
    echo "UNKNOWN_TERMINAL"
    exit 0
fi
echo "$STATE"
