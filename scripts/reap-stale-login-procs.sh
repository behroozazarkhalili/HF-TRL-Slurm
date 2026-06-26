#!/bin/bash
# Reap our own STALE, large-RSS, disposable login-node processes to relieve the
# per-user cgroup memory pressure that causes fork/posix_spawn EAGAIN on login1.
#
# Run ON DEMAND when forks start failing (`bash: fork: retry: Resource temporarily
# unavailable`). DRY-RUN by default — prints what it WOULD kill. Pass --kill to act.
#
# SAFETY (why this won't murder your editor / jobs):
#   - Targets ONLY known-disposable cmdline patterns (stale vscode-server node,
#     leftover copilot/node, orphaned venv python probes). Never a blanket "kill old".
#   - Requires age > THRESHOLD_H hours AND rss > MIN_RSS_MB.
#   - EXCLUDES: this script's own tree, the current shell, sshd, anything under an
#     salloc/srun allocation (SLURM_JOB_ID set), and the NEWEST process of each
#     disposable kind (so your active vscode-server session is preserved).
#
# Usage:
#   scripts/reap-stale-login-procs.sh            # dry-run (safe; shows candidates)
#   scripts/reap-stale-login-procs.sh --kill     # actually SIGTERM the candidates
#   THRESHOLD_H=12 scripts/reap-stale-login-procs.sh --kill   # override age threshold

set -uo pipefail

THRESHOLD_H="${THRESHOLD_H:-24}"          # kill only if older than this many hours
MIN_RSS_MB="${MIN_RSS_MB:-300}"           # ...and resident set larger than this
DO_KILL=0
[[ "${1:-}" == "--kill" ]] && DO_KILL=1

THRESH_S=$(( THRESHOLD_H * 3600 ))
ME=$$                                      # this script's pid
SELF_TREE="$ME $PPID"                       # don't touch our own chain

# Disposable cmdline patterns (extended-regex). Add new safe ones here only.
DISPOSABLE='vscode-server/.*/node|@github/copilot|copilot-linux|/\.copilot/|transformer-crosscoders/\.venv/bin/python.*(probe|audit|stats)'

CGDIR="/sys/fs/cgroup/user.slice/user-$(id -u).slice"
mem_mb() { local f="$CGDIR/memory.$1"; [[ -r "$f" ]] && echo $(( $(cat "$f") / 1024 / 1024 )) || echo "?"; }

echo "=========================================="
echo "Stale login-proc reaper  (mode: $([[ $DO_KILL == 1 ]] && echo KILL || echo DRY-RUN))"
echo "threshold: >${THRESHOLD_H}h AND >${MIN_RSS_MB}MB   user: $(id -un)   host: $(hostname)"
echo "cgroup mem: $(mem_mb current)MB / $(mem_mb max)MB"
echo "=========================================="

# If we're inside a SLURM allocation, refuse — these procs may be the job's.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[abort] running inside SLURM_JOB_ID=$SLURM_JOB_ID — reaper is login-node-only." >&2
    exit 2
fi

# Collect candidates: pid ppid etimes rss(KB) comm + full cmdline.
# Track the newest (smallest etimes) pid per disposable 'kind' to PRESERVE it.
declare -A newest_age   # kind -> smallest etimes seen
declare -A newest_pid   # kind -> pid of that newest

mapfile -t ROWS < <(ps -u "$(id -u)" -o pid=,ppid=,etimes=,rss=,comm= 2>/dev/null)

classify() {  # echo a 'kind' tag if cmdline matches a disposable pattern, else nothing
    local pid="$1" cl
    [[ -r "/proc/$pid/cmdline" ]] || return 1          # proc exited between snapshot & read
    cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
    [[ -z "$cl" ]] && return 1
    if   [[ "$cl" =~ vscode-server ]]; then echo "vscode-server"
    elif [[ "$cl" =~ copilot ]];        then echo "copilot"
    elif [[ "$cl" =~ \.venv/bin/python ]] && [[ "$cl" =~ (probe|audit|stats|token) ]]; then echo "venv-probe"
    else return 1; fi
}

# First pass: find the newest of each kind (to preserve the active session).
for row in "${ROWS[@]}"; do
    read -r pid ppid etimes rss comm <<<"$row"
    [[ " $SELF_TREE " == *" $pid "* ]] && continue
    kind=$(classify "$pid") || continue
    if [[ -z "${newest_age[$kind]:-}" || "$etimes" -lt "${newest_age[$kind]}" ]]; then
        newest_age[$kind]=$etimes; newest_pid[$kind]=$pid
    fi
done

# Second pass: select stale + large + not-newest candidates.
candidates=()
printf "%-8s %-14s %7s %8s  %s\n" PID KIND RSS_MB AGE_H CMD
printf -- "------------------------------------------------------------------\n"
for row in "${ROWS[@]}"; do
    read -r pid ppid etimes rss comm <<<"$row"
    [[ " $SELF_TREE " == *" $pid "* ]] && continue
    kind=$(classify "$pid") || continue
    [[ "$pid" == "${newest_pid[$kind]:-}" ]] && continue          # preserve newest of kind
    rss_mb=$(( rss / 1024 ))
    (( etimes < THRESH_S )) && continue                           # too young
    (( rss_mb < MIN_RSS_MB )) && continue                         # too small
    age_h=$(awk "BEGIN{printf \"%.1f\", $etimes/3600}")
    cl=$([[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-46)
    printf "%-8s %-14s %7s %8s  %s\n" "$pid" "$kind" "$rss_mb" "$age_h" "$cl"
    candidates+=("$pid")
done

echo "------------------------------------------------------------------"
echo "candidates: ${#candidates[@]}  (preserved newest-of-kind: ${newest_pid[*]:-none})"

if (( ${#candidates[@]} == 0 )); then
    echo "[ok] nothing stale to reap."
    exit 0
fi

if (( DO_KILL == 0 )); then
    echo "[dry-run] re-run with --kill to SIGTERM the ${#candidates[@]} candidate(s)."
    exit 0
fi

echo "[kill] sending SIGTERM to ${#candidates[@]} process(es)..."
for pid in "${candidates[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null    # children first
    kill -TERM "$pid" 2>/dev/null && echo "  TERM $pid"
done
sleep 3
echo "cgroup mem after: $(mem_mb current)MB / $(mem_mb max)MB"
echo "[done]"
