# TARI.Miner

TARI.Miner is an open-source NVIDIA CUDA miner for Tari (XTM) Cuckaroo29.
It connects to LuckyPool and has no developer fee, payout address, donation
schedule, or alternate mining connection.

## Download

Open the [latest release](https://github.com/JustAResearcher/TARI.Miner/releases/latest)
and download one file for your operating system:

- `TARI.Miner-v1.1.6-windows.zip`
- `TARI.Miner-v1.1.6-linux.tar.gz`
- `tari-miner-hiveos-1.1.6.tar.gz` for HiveOS

Each package contains all supported GPU backends. The starter detects every
NVIDIA GPU and selects the correct backend for each card:

- Compute capability 8.6: RTX 30 series
- Compute capability 8.9: RTX 40 series
- Compute capability 12.0: RTX 50 series

Mixed rigs are supported. Each GPU runs in its own process with a worker name
such as `RIG01-gpu0` or `RIG01-gpu1`. Pipeline depth is also selected inside
each process from the free VRAM remaining after its first solver context starts.
Automatic selection can use up to five contexts on Linux and Windows TCC, and
up to four on Windows WDDM. An explicit `--pipeline` value from 1 through 5
overrides the automatic platform cap; allocation still stops safely if VRAM
runs out.

The starter does not change GPU clocks, voltage, fans, or power limits.

## Windows

1. Extract `TARI.Miner-v1.1.6-windows.zip`.
2. Run `start-c29.bat`.
3. Paste your Tari wallet address when prompted.
4. Leave the starter window open while mining.

`start-c29.bat` runs `start-c29.ps1`, which does the work. Both files must stay
in the same folder. Windows PowerShell 5.1 is included with Windows, so nothing
needs installing.

Every GPU worker runs inside the starter's own window rather than a window of
its own, and the starter stays open until the last worker has exited. Press
Ctrl+C there to stop them all at once: workers share the starter's console so
that a single interrupt reaches every one of them.

The BAT file is preconfigured for `taric29-ca.luckypool.io:3111` and uses the
Windows computer name as the base worker name. To save settings, edit the
optional variables near the top of `start-c29.bat`.

To mine only on selected GPUs, set a comma-separated device list:

```bat
set "TARI_DEVICES=0,2"
start-c29.bat
```

To write per-GPU output to files instead of the console, set a log directory:

```bat
set "TARI_LOG_DIR=%CD%\logs"
start-c29.bat
```

Each worker writes progress to `gpu0.log`, `gpu1.log`, and so on, including the
periodic speed report. Warnings and connection errors go to `gpu0.err.log`
alongside it. The two streams are separate files on Windows; the Linux starter
combines them into one.

Pools that expect `wallet/worker` rather than `wallet.worker` need the login
separator set alongside the pool:

```bat
set "TARI_POOL=POOL_HOST:PORT"
set "TARI_LOGIN_SEPARATOR=/"
start-c29.bat
```

Extra miner options are applied to every selected GPU:

```bat
start-c29.bat --pipeline 1
```

## Linux

Extract `TARI.Miner-v1.1.6-linux.tar.gz`, then run:

```bash
./start-c29.sh
```

The starter prompts for a wallet in an interactive terminal. Environment
variables can provide permanent settings:

```bash
TARI_POOL=taric29-ca.luckypool.io:3111 \
TARI_WORKER=RIG01 \
TARI_WALLET=YOUR_TARI_WALLET \
TARI_DEVICES=all \
TARI_LOGIN_SEPARATOR=. \
./start-c29.sh
```

Use `TARI_DEVICES=0,2` to select specific GPUs. Press Ctrl+C to stop all GPU
workers started by the script.

## HiveOS

Create a Custom miner in the Flight Sheet with these values:

```text
Miner name: tari-miner-hiveos
Installation URL: https://github.com/JustAResearcher/TARI.Miner/releases/download/v1.1.6/tari-miner-hiveos-1.1.6.tar.gz
Hash algorithm: cuckaroo29
Wallet and worker template: %WAL%.%WORKER_NAME%
Pool URL: stratum+tcp://taric29-ca.luckypool.io:3111
Pass: x
```

The HiveOS launcher uses every supported GPU unless Extra config arguments
contains a selector such as `TARI_DEVICES=0,2`. Other extra arguments are
passed to each miner process, for example `--pipeline 1`. Per-GPU graph rates,
temperatures, fans, and share counters are reported to the HiveOS agent.

## Miner Options

Options placed after the starter command are passed to every selected GPU:

```text
--pool host:port        Pool endpoint
--wallet WALLET         Required Tari wallet address
--worker NAME           Worker name
--pass VALUE            Pool password; defaults to x
--login-separator S     Joins wallet and worker in the pool login; defaults to .
--intensity N           Duty cycle from 1 to 100 percent; defaults to 100
--pipeline N            Overlapped solver contexts; defaults automatically
--max-runtime-sec N     Stop after N seconds
--version               Print version and exit
```

### Intensity

`--intensity` sets how much of the time the miner works. At 100, the default, it
never pauses. At 50 it idles for about as long as it works, roughly halving both
the graph rate and the load on the card. Lower values scale the same way, so 25
works about a quarter of the time. Useful for sharing a GPU with something else,
or for keeping a laptop cooler and quieter:

```bash
./start-c29.sh --intensity 50
```

```bat
start-c29.bat --intensity 50
```

This is not the same as `--pipeline`. Pipeline depth controls how many solver
contexts overlap, which is a memory and latency tuning knob, and lowering it to
fit VRAM does not reduce how hard the GPU is driven. Intensity inserts idle time
between graphs and is the setting to reach for when the goal is less load.

The starter supplies `--device`, `--pool`, `--wallet`, and `--worker` after
user options so each GPU always receives its detected device index and unique
worker name. If automatic allocation does not fit in VRAM, retry with
`--pipeline 1`.

### Pool login format

The login sent to the pool is the wallet address, the separator, then the
worker name. LuckyPool expects `wallet.worker`, which is the default. Pools
expecting `wallet/worker` need the separator changed:

```bash
TARI_WALLET=YOUR_TARI_WALLET \
TARI_POOL=POOL_HOST:PORT \
TARI_LOGIN_SEPARATOR=/ \
./start-c29.sh
```

`TARI_LOGIN_SEPARATOR` sits alongside `TARI_POOL` because the two belong
together: a pool defines both the endpoint and the login format it accepts.
Passing `--login-separator` after the starter command works as well.

The wallet is checked before the first connection. Whitespace, control
characters, and inputs larger than the longest supported Tari text encoding are
rejected outright. A login with one of the two Base58 address lengths that uses
a `0`, `O`, `I`, or `l` emits a typo warning before mining starts. It is not
rejected solely for that warning, because pools exist that expect a username
rather than an address.

The pool connection is plain TCP. Do not use a sensitive password for
`--pass`; the default `x` is sufficient for LuckyPool.

### Exit Codes

A miner worker keeps running through anything it can recover from, including a
dropped connection and a pool outage. It exits non-zero only for a condition
that needs a restart or an operator, so a rig supervisor can act on the code:

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown, or `--max-runtime-sec` elapsed |
| 1 | Startup failure: sockets unavailable, no such CUDA device, or not enough VRAM for one solver |
| 2 | Invalid command line |
| 3 | A GPU solution failed host verification; the GPU or its tuning is suspect |
| 4 | The pool rejected the login repeatedly; check wallet, worker, password, and separator |
| 5 | Solver failure: a CUDA error, or three consecutive graphs with no surviving edges |
| 6 | The pool accepted the connection but never sent a job |
| 7 | The pool repeatedly sent invalid protocol data |

The starters propagate these. When a GPU worker exits non-zero, the starter
stops the remaining workers and exits with that same code, rather than carrying
on with its healthy GPUs — otherwise HiveOS never sees the failure. A starter
that fails before any worker runs uses its own codes: 2 for a missing wallet, 3
when `nvidia-smi` finds no GPU, 4 when `TARI_DEVICES` matches none, 5 for a
missing backend binary, and 130 for Ctrl+C.

## Test The Solver

The standalone solver checks GPU results with an independent CPU verifier.
Choose the backend matching the test GPU:

```bat
bin\tari_c29_solver_sm_89.exe --device 0 --count 320 --pipeline 2
```

```bash
bin/tari_c29_solver_sm_89 --device 0 --count 320 --pipeline 2
```

A correct run ends with `verify failures: 0`.

### Exact GPU recall regression

The GPU recall test compares exact `(header nonce, proof hash)` sets, not just
cycle counts. It independently packs every 42-edge proof, recalculates its
BLAKE2b-256 hash and difficulty, repeats both builds, and checks pipeline
parity. Build the shipped and conservative reference profiles for the GPU
architecture, then run:

```bat
build_solver.bat sm_120
build_solver.bat sm_120 reference
python tests\tari_c29_gpu_recall.py run --candidate bin\tari_c29_solver_sm_120.exe --reference bin\validation\tari_c29_solver_sm_120_reference.exe --arch sm_120 --output-dir validation --parity-pipeline 4
```

```bash
./build_solver.sh sm_89
./build_solver.sh sm_89 reference
python3 tests/tari_c29_gpu_recall.py run --candidate bin/tari_c29_solver_sm_89 --reference bin/validation/tari_c29_solver_sm_89_reference --arch sm_89 --output-dir validation --parity-pipeline 2
```

The default sequence tests 4,200 fixed graphs and saves the raw logs, JSONL
proof records, invoked commands, and comparison report. The reference binaries
remain under `bin/validation` and are not included in release packages.
The runner also rejects a binary whose embedded build target, runtime GPU
architecture, or compiled trim default does not match `--arch`. Throughput
measurements are a separate performance check and do not replace the exact
recall gate. Hosted CI compiles both profiles and exercises the verifier
fixtures; the full sequence runs only on a trusted host with the matching GPU.

### Trim tuning

The miner selects trim parameters at runtime from the detected GPU compute
capability (`tari_miner::arch_tuning` in `tari_miner_pipeline.h`). Only
architectures with benchmarked values deviate from the reference profile; all
others keep it, so a single binary is safe on any supported card. Explicit CLI
flags always win over the table:

```text
--ntrims N            trim rounds (even)
--trim-tpb-r1 N       round 1 threads/block
--trim-tpb-r23 N      rounds 2-3 threads/block
--gena-blocks/--gena-tpb/--genb-tpb/--trim-tpb/--tail-tpb/--recover-*
```

To benchmark a new GPU model, run the sweep harness on that card and record the
best row in `arch_tuning`:

```bash
tests/tune_sweep.sh bin/tari_c29_solver_sm_86 --count 3000 \
    --ntrims 44,46,48,50 --r1 768,960,1024 --r23 768,960,1024
```

It prints a ranked table and CSV; rows with `verify failures > 0` are marked
INVALID and must be discarded. After changing the table for an architecture,
re-run the exact GPU recall regression above before shipping.

### Recovery (nonce resolution)

When the host-side cycle finder reports candidate cycles, their edge nonces are
resolved by a single batched full-graph scan (`RecoveryBatch` in `mean_c29.cu`)
instead of one 2^EDGEBITS-edge scan per candidate. Candidates that do not fully
resolve are dropped; `verify()` re-derives each proof from the recovered nonces
independently, so nothing half-resolved can be submitted.

## Build From Source

The repository includes its required third-party source under
`third_party/cuckoo`.

Windows requires Visual Studio C++ tools and NVIDIA CUDA Toolkit 13.2. From an
x64 Native Tools Command Prompt:

```bat
build.bat
build_all.bat
```

Linux requires a C++ toolchain and an NVIDIA CUDA toolkit that supports all
three target architectures:

```bash
./build_all.sh
```

`build_all` also compiles the non-release reference solvers used by the recall
test. Individual backends can be built with `build_solver` or
`build_pool_miner` and one of `sm_86`, `sm_89`, or `sm_120`; pass `reference`
as the second `build_solver` argument to build only that validation profile.

## License

TARI.Miner is GPL-3.0-or-later. Required upstream licenses and attribution are
included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`third_party/cuckoo/LICENSE.txt`.

This is independent community software and is not affiliated with or endorsed
by Tari or LuckyPool.
