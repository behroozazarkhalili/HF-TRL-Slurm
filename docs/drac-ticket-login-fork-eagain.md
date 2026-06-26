# DRAC support ticket draft: login-node fork/posix_spawn EAGAIN under cgroup memory pressure

**Recipient:** `support@tech.alliancecan.ca` (or via https://support.alliancecan.ca web portal)

**Cluster:** Fir (login1.int.fir.alliancecan.ca)
**User:** ermia
**Account:** def-maxwl
**Node in question:** login1 (login node)

---

## Subject
login1: intermittent `fork: retry: Resource temporarily unavailable` / `EAGAIN posix_spawn '/bin/sh'` despite proc count far below the nproc limit

## Summary
On **login1**, interactive shells and tooling intermittently fail to spawn child processes with:

```
EAGAIN: resource temporarily unavailable, posix_spawn '/bin/sh'
bash: fork: retry: Resource temporarily unavailable
```

The failures are **transient** (a retry seconds later usually succeeds) and correlate with
having one or more **large-RSS processes** (Node.js, Python/venv interpreters) resident in our
login-node cgroup. They are **not** caused by hitting the process-count limit — at the time of
failure we had ~180 user processes against an `ulimit -u` of 1,545,971.

This blocks routine login-node work (job submission wrappers, `squeue`/`sacct` polling loops,
small Python dataset probes) until memory pressure in our cgroup eases.

## Error signatures (this session, 2026-06-26)
```
EAGAIN: resource temporarily unavailable, posix_spawn '/bin/sh'
bash: fork: retry: Resource temporarily unavailable
```
A trivial command (`true`, `ls | wc -l`, `cat <file>`) returns a non-zero exit with these on
stderr, then succeeds on retry once a large process releases memory.

## Diagnostic data (login1, at time of failures)

Process / limit state — rules out nproc exhaustion:
```
hostname                : login1
ulimit -u (soft/hard)   : 1545971 / 1545971
user processes          : 180          # nowhere near the cap
system-wide threads     : 13645        # threads-max = 3091943
loadavg                 : 10.80 22.60 22.78
```

cgroup v2 memory cap — the actual constraint (a GENERAL per-user policy, not
account-specific: the limit is keyed on UID at the `user-<uid>.slice` level, and there is
no def-maxwl / SLURM-account cgroup on the login node — account cgroups exist only per-job
on compute nodes):
```
/sys/fs/cgroup/user.slice/user-3129465.slice/memory.max     = 17179869184   # 16 GiB per-user cap
/sys/fs/cgroup/user.slice/user-3129465.slice/memory.current =  7105867776   # ~6.6 GiB in use (~41%)
/sys/fs/cgroup/user.slice/memory.max                         = 303980744704 # ~283 GiB node total (all users)
vm.overcommit_memory                                          = 0 (heuristic)
vm.overcommit_ratio                                           = 50
```

Top resident processes in our cgroup at failure time:
```
1262 MB  node
1251 MB  node
1185 MB  node
 416 MB  node
 410 MB  claude
```

## Root-cause hypothesis
With `vm.overcommit_memory=0` (heuristic) and a **16 GiB cgroup memory cap**, a `fork()` /
`posix_spawn()` from a large-RSS parent (e.g. a ~1.2 GB Node process) requires the kernel to
make a copy-on-write memory **reservation** roughly equal to the parent's footprint against the
cgroup's remaining headroom. When several multi-hundred-MB-to-GB processes are co-resident, that
reservation intermittently cannot be satisfied within the 16 GiB cap and the spawn fails with
**EAGAIN**, even though the process-count and system-wide thread limits are nowhere near
exhausted. The condition clears as soon as one of the large parents exits or shrinks.

This is consistent with all observed behaviour:
- transient and self-clearing,
- triggered/worsened by running memory-heavy CLI tooling on the login node,
- never affects our SLURM batch jobs (they run inside per-job cgroups on compute nodes).

## Questions for support
1. We've confirmed the **16 GiB cap is a general per-user policy** (`user-<uid>.slice/memory.max`,
   not account-tagged) under a ~283 GiB node total. Is this the intended/current Fir login-node
   guardrail, and is the value tunable per-user or only node-wide?
2. Would raising `vm.overcommit_memory` to `1` (always overcommit) on login nodes — or increasing
   the per-user cgroup `memory.max` — be appropriate to avoid fork-reservation EAGAIN for
   legitimate interactive tooling? (We understand login nodes are not for compute; our usage is
   job orchestration + small dataset probes, not training.)
3. Is there a recommended pattern for memory-heavy interactive CLIs (e.g. Node-based tools) on
   login nodes, or should these be confined to an `salloc` interactive compute allocation?

## Our mitigations already in place (for reference)
- Heavy data wrangling / model loads run as **CPU or GPU SLURM jobs**, never on the login node.
- Thread-pinning before any numpy/polars import:
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`,
  plus `RAYON_NUM_THREADS` / `POLARS_MAX_THREADS` for Rust-backed libs.
- `HF_HUB_DISABLE_XET=1` (hf_xet's Rust thread-spawn was itself a source of WouldBlock panics on
  the login node).

## Severity / impact
medium-to-high: no data loss, but it interrupts interactive job-orchestration workflows and forces
manual retries. A clearer policy or a small cgroup/overcommit adjustment would remove the friction.
