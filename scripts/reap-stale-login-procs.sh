#!/bin/bash
# Reap our own STALE, disposable login-node processes to stay under the cgroup
# pids/TasksMax ceiling that causes fork/posix_spawn EAGAIN on DRAC login nodes.
#
# THE REAL LIMIT (confirmed by DRAC support, 2026-06): the binding constraint is the
# cgroup PIDS controller, `pids.max` (TasksMax=512), counting THREADS (each process
# has >=1 task/thread). `nproc` is never reached first. Memory is a secondary signal.
# So this tool ranks candidates by THREADS contributed + age, reports against
# pids.max, and (in --auto) only acts when actually near the ceiling.
#
# Run ON DEMAND when forks start failing, or periodically with --auto. DRY-RUN by
# default; pass --kill to act.
#
# SAFETY (won't touch your editor / jobs):
#   - NEVER kills vscode-server (any age): killing even an old editor backend can
#     disconnect the live session and force a relogin. The editor is OFF-LIMITS.
#   - Targets ONLY known-disposable patterns: orphaned graphify AST workers and
#     our own venv-python probe/monitor/audit children.
#   - Requires age > THRESHOLD_H hours.
#   - EXCLUDES: this script's tree, sshd, anything under a SLURM allocation
#     (SLURM_JOB_ID set), and the NEWEST process of each disposable kind.
#   - --auto only reaps when pids.current/pids.max exceeds AUTO_PCT (default 70%).
#
# Usage:
#   scripts/reap-stale-login-procs.sh             # dry-run: report + candidates
#   scripts/reap-stale-login-procs.sh --kill      # SIGTERM stale candidates
#   scripts/reap-stale-login-procs.sh --auto --kill  # only if near pids.max
#   THRESHOLD_H=12 AUTO_PCT=60 scripts/reap-stale-login-procs.sh --auto --kill

set -uo pipefail

THRESHOLD_H="${THRESHOLD_H:-24}"   # only reap procs older than this many hours
AUTO_PCT="${AUTO_PCT:-70}"          # --auto acts only above this % of pids.max
DO_KILL=0; AUTO=0
for a in "$@"; do
    case "$a" in
        --kill) DO_KILL=1 ;;
        --auto) AUTO=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    esac
done

THRESH_S=$(( THRESHOLD_H * 3600 ))
ME=$$
SELF_TREE="$ME $PPID"

# Disposable cmdline patterns (extended-regex). Add new SAFE ones here only.
# (matched against /proc/<pid>/cmdline)

CGDIR="/sys/fs/cgroup/user.slice/user-$(id -u).slice"
read_cg() { local f="$CGDIR/$1"; [[ -r "$f" ]] && cat "$f" 2>/dev/null || echo "?"; }
mem_mb() { local v; v=$(read_cg "memory.$1"); [[ "$v" == "?" ]] && echo "?" || echo $(( v / 1024 / 1024 )); }

PIDS_MAX=$(read_cg pids.max)
PIDS_CUR=$(read_cg pids.current)
# our own live thread count (authoritative "task" count for this UID on this node)
MY_THREADS=$(ps -u "$(id -u)" -L --no-headers 2>/dev/null | wc -l)
PCT="?"
if [[ "$PIDS_MAX" =~ ^[0-9]+$ && "$PIDS_CUR" =~ ^[0-9]+$ && "$PIDS_MAX" -gt 0 ]]; then
    PCT=$(( PIDS_CUR * 100 / PIDS_MAX ))
fi

echo "=========================================="
echo "Stale login-proc reaper  (mode: $([[ $DO_KILL == 1 ]] && echo KILL || echo DRY-RUN)$([[ $AUTO == 1 ]] && echo ,AUTO))"
echo "host: $(hostname)   user: $(id -un)   age>${THRESHOLD_H}h"
echo "pids.current/pids.max : ${PIDS_CUR}/${PIDS_MAX}  (${PCT}% of TasksMax)   <-- the real ceiling"
echo "my live threads (tasks): ${MY_THREADS}            cgroup mem: $(mem_mb current)/$(mem_mb max) MB"
echo "=========================================="

# Never touch processes inside a SLURM job allocation.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[abort] inside SLURM_JOB_ID=$SLURM_JOB_ID; reaper is login-node-only." >&2
    exit 2
fi

# --auto gate: only proceed if we're actually near the pids ceiling.
if (( AUTO == 1 )) && [[ "$PCT" =~ ^[0-9]+$ ]] && (( PCT < AUTO_PCT )); then
    echo "[auto] ${PCT}% < ${AUTO_PCT}% of pids.max; nothing to do."
    exit 0
fi

# classify <pid> -> echoes 'kind' tag if disposable, else returns 1.
# (Runs in a subshell via $(), so no globals: caller reads cmdline/threads itself.)
classify() {
    local pid="$1" cl
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
    [[ -z "$cl" ]] && return 1
    # NEVER touch the editor. vscode-server is EXCLUDED outright — killing even an old
    # backend can disconnect the live session and force a relogin, which is unacceptable.
    [[ "$cl" =~ vscode-server ]] && return 1
    if   [[ "$cl" =~ graphifyy ]];      then echo "graphify"
    elif [[ "$cl" =~ \.venv/bin/python ]] || [[ "$cl" =~ /python3?\  ]]; then
        # only OUR disposable python: probes, monitors, audits, one-offs
        [[ "$cl" =~ (probe|audit|stats|token|build_fable|slurm_state|monitor|reap|-c\ ) ]] && echo "venv-probe" || return 1
    else return 1; fi
}
nthreads() { local n; n=$(awk '/^Threads:/{print $2}' "/proc/$1/status" 2>/dev/null); echo "${n:-1}"; }
cmdline()  { [[ -r "/proc/$1/cmdline" ]] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || echo ""; }

declare -A newest_age newest_pid
mapfile -t ROWS < <(ps -u "$(id -u)" -o pid=,etimes= 2>/dev/null)

# Pass 1: find the newest (smallest etimes) of each kind -> PRESERVE it.
for row in "${ROWS[@]}"; do
    read -r pid etimes <<<"$row"
    [[ " $SELF_TREE " == *" $pid "* ]] && continue
    kind=$(classify "$pid") || continue
    if [[ -z "${newest_age[$kind]:-}" || "$etimes" -lt "${newest_age[$kind]}" ]]; then
        newest_age[$kind]=$etimes; newest_pid[$kind]=$pid
    fi
done

# Pass 2: select stale + not-newest candidates; tally threads they'd reclaim.
candidates=(); reclaim_threads=0
printf "%-8s %-14s %7s %8s  %s\n" PID KIND THREADS AGE_H CMD
printf -- "------------------------------------------------------------------\n"
for row in "${ROWS[@]}"; do
    read -r pid etimes <<<"$row"
    [[ " $SELF_TREE " == *" $pid "* ]] && continue
    kind=$(classify "$pid") || continue
    [[ "$pid" == "${newest_pid[$kind]:-}" ]] && continue   # preserve newest of kind
    (( etimes < THRESH_S )) && continue                    # too young
    age_h=$(awk "BEGIN{printf \"%.1f\", $etimes/3600}")
    nt=$(nthreads "$pid"); cl=$(cmdline "$pid")
    printf "%-8s %-14s %7s %8s  %s\n" "$pid" "$kind" "$nt" "$age_h" "${cl:0:46}"
    candidates+=("$pid"); reclaim_threads=$(( reclaim_threads + nt ))
done
echo "------------------------------------------------------------------"
echo "candidates: ${#candidates[@]}   threads they hold: ${reclaim_threads}   (preserved newest: ${newest_pid[*]:-none})"

if (( ${#candidates[@]} == 0 )); then
    echo "[ok] nothing stale to reap."
    exit 0
fi
if (( DO_KILL == 0 )); then
    echo "[dry-run] re-run with --kill to SIGTERM the ${#candidates[@]} candidate(s) (~${reclaim_threads} threads)."
    exit 0
fi

echo "[kill] SIGTERM ${#candidates[@]} process(es)..."
for pid in "${candidates[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null   # children first
    kill -TERM "$pid" 2>/dev/null && echo "  TERM $pid"
done
sleep 3
NEW_CUR=$(read_cg pids.current); NEW_TH=$(ps -u "$(id -u)" -L --no-headers 2>/dev/null | wc -l)
echo "after: pids.current ${PIDS_CUR} -> ${NEW_CUR}   my threads ${MY_THREADS} -> ${NEW_TH}"
echo "[done]"
