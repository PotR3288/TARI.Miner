# C29 Mining Performance — Findings & Handoff

**Date:** 2026-09 (asimov hardware validation)
**Hardware:** asimov, 2× NVIDIA GeForce RTX 3060 (sm_86, 12 GB each), driver 595.84, CUDA 13.0 at `/usr/local/cuda-13`
**Repo state:** `main` @ `883a550` — Phase 3 GPU endpoint-compression is **opt-in** (`TARI_C29_GPU_COMPRESS=1`, default off), correct but only ~+10%.

This document records what was measured, where the time actually goes, and why the
obvious optimization (GPU compression) did not deliver. It is the starting point for a
ground-up re-optimization of Cuckaroo29 mining.

---

## 1. The per-attempt cost model (measured on real hardware)

A single solve attempt = **trim** (GPU, ~50 trim stages) + **host walk** (build graph + find cycles) +
**recovery** (GPU nonce resolution). The standalone solver pipelines `trim(N+1)` over `walk(N)`, so:

```
per-graph time ≈ max( trim_time , walk_time + recovery_time )
```

Measured on asimov, RTX 3060 sm_86, default build (`ntrims=50`):

| Component | Measured cost / attempt | Notes |
|---|---|---|
| **GPU trim** (all stages) | **~310 ms** | The floor. Confirmed by the skip-walk ceiling below. |
| **Host walk** (graph build + cycle find) | **~240–268 ms** | Dominant host cost; see §2 for breakdown. |
| — of which hash compression | ~11% (~30 ms) | 4.6M linear-probe lookups into two 8 MB tables. |
| — of which **sequential `add_edge` loop** | **~90% (~240 ms)** | 776K edges × random `adjlist[]`/`links[]` R/W, memory-latency bound on CPU. |
| Recovery (GPU) | ~43 ms when candidates exist | Amortized across all candidates in one pass; rare (~1/32 attempts). |

**Throughput:** 512 graphs / 217 s = **~2.35 g/s**, 14 cycles, 0 verify failures (baseline `ae44468`
matches: 2.437 g/s — the small delta is run-to-run variance).

### The skip-walk ceiling (the key experiment)

`TARI_C29_SKIP_WALK=1` skips the host walk entirely and measures pure trim+copy throughput:

```
skip-walk : 512 graphs / ~160 s ≈ 3.24 g/s   (GPU 0, warm)
normal    : 512 graphs / 217 s = 2.35 g/s
```

**This is the theoretical ceiling if walk were free: +38%.** It proves trim (~310 ms) is the floor and
that any real optimization must drive `walk_time` **below** ~310 ms to approach it.

---

## 2. Where the host walk time actually goes (the surprising part)

Instrumented with `TARI_C29_WALK_TIMING=1` (emits per-attempt stats via stderr, bypassing
`SQUASH_OUTPUT`). On a real trimmed graph (~776K edges):

```
host-walk edges 775958 nsols 0 time 268.3 ms probes 4.6M dfs_triggers ~4
```

- **probes = 4.6M** — linear-probing hash lookups for node-ID compression (two 8 MB tables).
- **dfs_triggers ≈ 4/attempt** — cycle-finding DFS fires only a handful of times; the graph is sparse
  (avg degree ~1) and `cycles_with_link` is bounded at depth 42. **Cycle detection itself is cheap.**

So the walk cost is *not* cycle-finding. It splits into:
1. **Hash compression** (~30 ms, 11%) — parallelizable on GPU (done in Phase 3).
2. **Sequential `add_edge` adjacency construction** (~240 ms, 90%) — the real bottleneck. Each of the
   776K edges does dependent random reads/writes into `adjlist[]` and `links[]` arrays; single-threaded,
   memory-latency bound on CPU. This is what remains after GPU compression.

---

## 3. Phase 3 (GPU endpoint→ID compression) — result: correct but only ~+10%

**What it does:** replaces the host hash compressor with a GPU radix-sort + unique that assigns dense
*injective* endpoint IDs, then a lean O(nedges) host adjacency build with pre-computed IDs. Node IDs are
pure labels (cycle detection uses only adjacency; recovery stores *edge indices*; `verify()` re-derives
every proof via SipHash), so an injective relabeling is strictly safer than the lossy host compressor
(~18% false merges).

**Correctness — verified:** at count=512, GPU-compress path found **exactly 14 cycles / 0 verify failures**,
identical to the default path. Host-side simulation: 776K values → 775,912 distinct dense IDs, 0 mismatches
vs reference.

**Performance — disappointing:**

| Config | g/s | walk ms | probes |
|---|---|---|---|
| A default (lossy host compress) | **2.352** | 268 | 4.6M |
| B GPU-compress (injective) | **2.526** | 239 | 0 |

Removing all 4.6M hash probes saved only ~29 ms of the 268 ms walk — confirming compression was never the
dominant cost. Walk is still ~240 ms, co-dominant with trim (~310 ms), so per-graph time barely moves.

**Bugs found & fixed during Phase 3 (all in `mean_c29.cu` / `graph.hpp`, all committed):**
- Inverted `cg.reset(!gpu_ids)` → skipped compressor-table reset on the default path → stale entries
  accumulated until a probe chain saturated → infinite loop at 100% CPU, idle GPU. Fixed to `reset(gpu_ids)`.
- Scatter wrote `comp[2*i+which]` (sorted-position index) instead of mapping each value back to its
  *original* edge via `perm_out[i]` → equal values split across IDs (no cycle could close: 0/512), distinct
  values collided (~190K spurious DFS triggers). Fixed.
- Rank used **exclusive** prefix sum (correct only for the first element of each value-group) instead of
  **inclusive − 1** → within a group `[a,a,b,c,c]` got IDs `[0,1,1,2,3]` not `[0,0,1,2,2]`. Fixed to
  `InclusiveSum - 1`.

These are the kind of off-by-one / permutation-direction bugs that only surface on real hardware with real
graphs — a strong argument for keeping asimov in the loop.

---

## 4. The real bottleneck & the path forward (for the new project)

**The dominant cost is sequential host-side graph construction (~240 ms/attempt), not compression and not
cycle-finding.** To reach the ~3.2 g/s ceiling (+38%), `walk_time` must drop well below trim time (~310 ms).

Options, in order of expected payoff:

1. **On-GPU cycle detection (the real Phase 3).** Build adjacency + run the bounded DFS on the GPU so the
   host walk becomes a thin DtoH of just the found cycles. This removes ~240 ms → per-graph ≈ max(310,
   ~50+recovery) ≈ 310 ms → **~3.2 g/s**. Substantial build: GPU adjacency construction (CSR or linked-list),
   GPU DFS with the depth-42 bound and MAXSOLS cap, careful handling of the `sols`/`visited` shared state that
   currently makes host `add_edge` sequential.

2. **Parallelized host edge insertion.** Split the 776K-edge loop across CPU cores / a GPU kernel that only
   builds adjacency (no DFS), then run cycle detection on the result. Smaller win than #1 and `add_edge` has
   shared mutable state (`nlinks`, `sols`) that makes safe parallelism non-trivial — but worth prototyping to
   measure its ceiling before committing to full GPU DFS.

3. **Reduce trim time itself.** Trim is the floor (~310 ms). If it can be cut (fewer stages, better kernels,
   larger batches), the whole per-graph time drops regardless of walk. Worth profiling independently — a 20%
   trim reduction is worth more than any walk optimization once walk < trim.

**Recommended first step for the new project:** prototype option 2 (parallel adjacency build) to measure how
much of the 240 ms is truly irreducible sequential work vs. parallelizable, *and* profile the trim stages to
see if the ~310 ms floor has slack. That data decides whether full on-GPU cycle detection (#1) is worth the
build effort or whether a cheaper host-side win gets most of the way there.

---

## 5. Reproduction & tooling notes

- **Build (asimov):** `export PATH=/usr/local/cuda-13/bin:$PATH && ./build_solver.sh sm_86 release` →
  `bin/tari_c29_solver_sm_86`. Local dev box uses nvcc 12.0; CI covers sm_86/sm_89/sm_120.
- **Env vars (all opt-in, default off):**
  - `TARI_C29_WALK_TIMING=1` — per-attempt walk/probe/DFS stats via stderr (`host-walk edges N nsols K time T ms probes P dfs D`).
  - `TARI_C29_GPU_COMPRESS=1` — Phase 3 GPU endpoint compression (correct, ~+10%).
  - `TARI_C29_SKIP_WALK=1` — skip the host walk; measures the trim-only ceiling (~3.24 g/s).
- **A/B pattern that works on asimov:** warm up (`--count 4`) first, then run count=512 per GPU in parallel
  (GPU 0 vs GPU 1) to avoid cross-contamination; `setsid nohup ... &` so the SSH session can drop.
- **Gotchas learned:**
  - `print_log()` is stripped in release builds (`SQUASH_OUTPUT=1`) — instrumentation must use `fprintf(stderr)` directly.
  - The standalone solver calls `findcycles_copied_status` directly (bypasses `solve()`), so per-graph timing
    must be added at the pipeline loop, not in `solve()`.
  - CUDA decays array kernel parameters to pointers — only **structs** are truly passed by value into device
    param space (this caused three asimov crashes in Phase 2; fixed with a struct wrapper).
  - CUB moved under CCCL in CUDA 13 (`.../include/cccl/cub/cub.cuh`); local nvcc 12.0 has it at `/usr/include/cub`.

## 6. Commit chain (this session)

```
ae44468  base (pre-Phase-1)
7a14f45  Phase 1: arch_tuning + --trim-tpb-r1/r23 CLI flags
f64f88d  Phase 2: batched Recovery initial
418e78e  recoverIndexes alloc fix (MAXSOLS*PROOFSIZE)
dbaa7f2  struct wrapper for by-value kernel param (THE critical Phase-2 fix)
b73cd0d  walk timing via stderr (print_log squashed in release)
569138a  profile host-walk internals: compression probes + DFS trigger counts
18ecfc1  TARI_C29_SKIP_WALK upper-bound experiment flag
07d0573  per-graph trim vs findcycles wall time under TARI_C29_WALK_TIMING
1978def  Phase 3: on-GPU endpoint->ID compression (opt-in)
561ffb3  fix inverted cg.reset() + interleaved ID layout
d98634b  diagnose gpu-compress fallback (log which CUB step fails)
d62c10a  fix scatter_ranks: map rank to ORIGINAL edge via perm_out
883a550  fix rank computation: INCLUSIVE scan minus 1, not exclusive   <- HEAD
```
