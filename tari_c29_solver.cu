// tari_c29_solver.cu - standalone Tari (XTM) Cuckaroo29 GPU solver / benchmark.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Drives the reference Cuckaroo29 mean trimmer (cuckoo-ref/src/cuckaroo/mean.cu,
// here the MSVC-portable copy mean_c29.cu) with the Tari-specific
// key-derivation / proof-packing / difficulty layer (tari_c29).
//
// Per header nonce it: derives Tari siphash keys (BLAKE2b of nonce||mining_hash),
// injects them into the GPU trimmer (bypassing the reference setheader), runs the
// full trim + cycle-find on the GPU, then INDEPENDENTLY re-verifies any 42-cycle
// on the host and prints its C29 difficulty. Reports graphs/s.
//
// Finding a 42-cycle is ~1/42 per graph, so use --count a few hundred to see
// verified cycles. This is a smoke test + benchmark, NOT a pool miner.
//
// Build: build_solver.bat   (nvcc -arch=sm_120 for the RTX 5090)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <future>
#include <fstream>

#include "tari_miner_pipeline.h"
#include "tari_miner_reliability.h"
#include "version.h"

// --- MSVC host-compiler shims for the reference solver's GNU-isms ---
#if defined(_MSC_VER)
#define __builtin_prefetch(...) ((void)0)   // bitmap.hpp perf hint -> no-op

// Pre-empt cuckoo's portable_endian.h: its __WINDOWS__ branch assumes MinGW
// (needs <sys/param.h> and BYTE_ORDER). Define the guard + correct LE x64 macros.
#ifndef PORTABLE_ENDIAN_H__
#define PORTABLE_ENDIAN_H__
#include <stdlib.h>
#define htole16(x) (x)
#define htole32(x) (x)
#define htole64(x) (x)
#define le16toh(x) (x)
#define le32toh(x) (x)
#define le64toh(x) (x)
#define htobe16(x) _byteswap_ushort(x)
#define htobe32(x) _byteswap_ulong(x)
#define htobe64(x) _byteswap_uint64(x)
#define be16toh(x) _byteswap_ushort(x)
#define be32toh(x) _byteswap_ulong(x)
#define be64toh(x) _byteswap_uint64(x)
#endif
#endif

// Silence the reference solver's internal logging by default and rename its main().
#ifndef TARI_C29_SQUASH_OUTPUT
#define TARI_C29_SQUASH_OUTPUT 1
#endif
#ifndef SOLVER_CPULOAD
#define SOLVER_CPULOAD 0
#endif
#ifndef TARI_C29_REFERENCE_BUILD
#define TARI_C29_REFERENCE_BUILD 0
#endif
#ifndef TARI_C29_BUILD_ARCH
#define TARI_C29_BUILD_ARCH 0
#endif
#define SQUASH_OUTPUT TARI_C29_SQUASH_OUTPUT
#define main reference_mean_main_unused
#include "mean_c29.cu"      // brings in solver_ctx, create_solver_ctx, etc.
#undef main

#include "tari_c29.h"

static double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

static tari_miner::DriverMode driver_mode(const cudaDeviceProp &prop) {
#if defined(_WIN32)
    return prop.tccDriver
        ? tari_miner::DriverMode::WindowsTcc
        : tari_miner::DriverMode::WindowsWddm;
#else
    (void)prop;
    return tari_miner::DriverMode::Linux;
#endif
}

static size_t solver_context_bytes(const SolverCtx *ctx) {
    size_t bytes = ctx->trimmer.globalbytes();
#if RECOVERY_SMALL_OUTPUT
    bytes += PROOFSIZE * sizeof(u32);
#endif
    return bytes;
}

static int parse_hex32(const char *hex, uint8_t out[32]) {
    if ((int)strlen(hex) != 64) return -1;
    for (int i = 0; i < 32; i++) {
        unsigned v;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
    }
    return 0;
}

static void report_cuda_failure(int device, int context, const char *phase, cudaError_t error) {
    fprintf(stderr,
            "fatal: CUDA %s failure on device %d solver context %d: %s (%s); "
            "exiting for supervisor restart\n",
            phase, device, context, cudaGetErrorName(error), cudaGetErrorString(error));
}

static void report_zero_yield_failure(int device, int context) {
    fprintf(stderr,
            "fatal: device %d solver context %d returned zero surviving edges for "
            "%u consecutive graphs; exiting for supervisor restart\n",
            device, context, tari_miner::MAX_CONSECUTIVE_ZERO_YIELDS);
}

int main(int argc, char **argv) {
    int device = 0;
    uint64_t start_nonce = 0;
    uint64_t count = 256;
    uint64_t target = 1;                 // mainnet C29 min_difficulty = 1
    int pipeline = 2;
    bool pipeline_set = false;
    const char *recall_path = nullptr;
    int ntrims_override = 0;
    int gena_blocks = -1, gena_tpb = -1, genb_tpb = -1;
    int trim_tpb = -1, tail_tpb = -1, recover_blocks = -1, recover_tpb = -1;
    int trim_tpb_r1 = -1, trim_tpb_r23 = -1;
    uint8_t mining_hash[32];
    for (int i = 0; i < 32; i++) mining_hash[i] = (uint8_t)i;  // deterministic default

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--nonce") && i + 1 < argc) start_nonce = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) target = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--pipeline") && i + 1 < argc) { pipeline = atoi(argv[++i]); pipeline_set = true; }
        else if (!strcmp(argv[i], "--recall-jsonl") && i + 1 < argc) recall_path = argv[++i];
        else if (!strcmp(argv[i], "--ntrims") && i + 1 < argc) ntrims_override = atoi(argv[++i]) & -2;
        else if (!strcmp(argv[i], "--gena-blocks") && i + 1 < argc) gena_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gena-tpb") && i + 1 < argc) gena_tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--genb-tpb") && i + 1 < argc) genb_tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trim-tpb") && i + 1 < argc) trim_tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tail-tpb") && i + 1 < argc) tail_tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--recover-blocks") && i + 1 < argc) recover_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--recover-tpb") && i + 1 < argc) recover_tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trim-tpb-r1") && i + 1 < argc) trim_tpb_r1 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trim-tpb-r23") && i + 1 < argc) trim_tpb_r23 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mining-hash") && i + 1 < argc) {
            if (parse_hex32(argv[++i], mining_hash) != 0) {
                fprintf(stderr, "bad --mining-hash (need 64 hex chars)\n"); return 2;
            }
        } else if (!strcmp(argv[i], "--version")) {
            printf("TARI.Miner C29 solver %s\n", TARI_MINER_VERSION);
            return 0;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: tari_c29_solver [--device N] [--count N] [--nonce N] "
                   "[--target D] [--pipeline N] [--ntrims N] [--mining-hash <64hex>] "
                   "[--recall-jsonl FILE] "
                   "[--gena-blocks N] [--gena-tpb N] [--genb-tpb N] "
                   "[--trim-tpb N] [--tail-tpb N] [--recover-blocks N] [--recover-tpb N] "
                   "[--trim-tpb-r1 N] [--trim-tpb-r23 N] "
                   "[--version]\n");
            return 0;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }

    std::ofstream recall;
    if (recall_path) {
        recall.open(recall_path, std::ios::out | std::ios::trunc);
        if (!recall) {
            fprintf(stderr, "cannot open --recall-jsonl output: %s\n", recall_path);
            return 2;
        }
    }

    cudaError_t select_rc = cudaSetDevice(device);
    if (select_rc != cudaSuccess) {
        fprintf(stderr, "no CUDA device %d: %s\n", device, cudaGetErrorString(select_rc));
        return 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        fprintf(stderr, "no CUDA device %d\n", device); return 1;
    }
    printf("TARI.Miner C29 solver %s on %s (%.0f GB, sm_%d%d)\n",
           TARI_MINER_VERSION, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    char mining_hash_hex[65];
    for (int i = 0; i < 32; i++)
        sprintf(mining_hash_hex + 2 * i, "%02x", mining_hash[i]);
    mining_hash_hex[64] = 0;
    printf("mining_hash = %s\n", mining_hash_hex);
    printf("nonces      = %llu .. %llu  (%llu graphs)\n",
           (unsigned long long)start_nonce,
           (unsigned long long)(start_nonce + count - 1),
           (unsigned long long)count);
    printf("target diff = %llu\n\n", (unsigned long long)target);

    SolverParams params;
    fill_default_params(&params);
    params.device = device;
    params.mutate_nonce = false;         // we inject keys ourselves; no header mutation
#if SOLVER_CPULOAD
    params.cpuload = true;
#endif
#if !TARI_C29_REFERENCE_BUILD
    // Runtime per-arch tuning: the reference profile stays bit-exact with the
    // upstream defaults so validation runs are comparable.
    const tari_miner::ArchTuning arch = tari_miner::arch_tuning(prop.major, prop.minor);
    if (arch.ntrims > 0) params.ntrims = (uint16_t)arch.ntrims;
    if (arch.trim_tpb_r1 > 0) params.trimtpb_r1 = (uint16_t)arch.trim_tpb_r1;
    if (arch.trim_tpb_r23 > 0) params.trimtpb_r23 = (uint16_t)arch.trim_tpb_r23;
#endif
    if (ntrims_override > 0) params.ntrims = (uint16_t)ntrims_override;
    if (gena_blocks > 0) params.genablocks = (uint16_t)gena_blocks;
    if (gena_tpb > 0) params.genatpb = (uint16_t)gena_tpb;
    if (genb_tpb > 0) params.genbtpb = (uint16_t)genb_tpb;
    if (trim_tpb > 0) params.trimtpb = (uint16_t)trim_tpb;
    if (tail_tpb > 0) params.tailtpb = (uint16_t)tail_tpb;
    if (recover_blocks > 0) params.recoverblocks = (uint16_t)recover_blocks;
    if (recover_tpb > 0) params.recovertpb = (uint16_t)recover_tpb;
    if (trim_tpb_r1 > 0) params.trimtpb_r1 = (uint16_t)trim_tpb_r1;
    if (trim_tpb_r23 > 0) params.trimtpb_r23 = (uint16_t)trim_tpb_r23;
    SolverCtx *ctx = create_solver_ctx(&params);
    if (!ctx || !ctx->trimmer.initsuccess) {
        fprintf(stderr, "failed to init solver (need ~6GB VRAM). reason: %s\n", LAST_ERROR_REASON);
        return 1;
    }
    size_t free_after_first = 0, total_after_first = 0;
    cudaError_t memory_rc = cudaMemGetInfo(&free_after_first, &total_after_first);
    if (memory_rc != cudaSuccess)
        cudaGetLastError();
    pipeline = tari_miner::choose_pipeline_depth(
        pipeline_set,
        pipeline,
        driver_mode(prop),
        free_after_first,
        solver_context_bytes(ctx),
        memory_rc == cudaSuccess
    );

    uint64_t graphs = 0, cycles = 0, shares = 0, bugs = 0;
    int exit_code = 0;
    double t0 = 0.0;
    std::vector<SolverCtx*> extra_contexts;
    bool recall_run_written = false;

    auto write_recall_run = [&]() {
        if (!recall.is_open() || recall_run_written)
            return;
        recall
            << "{\"type\":\"run\",\"mining_hash\":\"" << mining_hash_hex
            << "\",\"start_nonce\":" << (unsigned long long)start_nonce
            << ",\"count\":" << (unsigned long long)count
            << ",\"pipeline\":" << pipeline
            << ",\"ntrims\":" << (unsigned)params.ntrims
            << ",\"compiled_ntrims\":" << TARI_C29_DEFAULT_NTRIMS
            << ",\"arch\":\"sm_" << TARI_C29_BUILD_ARCH
            << "\",\"device_arch\":\"sm_" << prop.major << prop.minor
            << "\",\"profile\":\""
            << (TARI_C29_REFERENCE_BUILD ? "reference" : "release")
            << "\"}\n";
        recall_run_written = true;
    };

    auto inject_keys = [&](SolverCtx *c, uint64_t nonce) {
        tari_siphash_keys k;
        tari_c29_derive_keys(nonce, mining_hash, &k);
        c->trimmer.sipkeys.k0 = k.k0;
        c->trimmer.sipkeys.k1 = k.k1;
        c->trimmer.sipkeys.k2 = k.k2;
        c->trimmer.sipkeys.k3 = k.k3;
        c->sols.clear();
    };

    auto consume_solutions = [&](SolverCtx *c, uint64_t nonce) {
        int nsols = (int)(c->sols.size() / TARI_C29_PROOFSIZE);
        for (int s = 0; s < nsols; s++) {
            uint32_t edges[TARI_C29_PROOFSIZE];
            for (int i = 0; i < TARI_C29_PROOFSIZE; i++)
                edges[i] = c->sols[s * TARI_C29_PROOFSIZE + i];

            // (2)+(3) Independent host verify + pack + difficulty (end-to-end check).
            int rc = tari_c29_verify(nonce, mining_hash, edges);
            if (rc == TARI_C29_OK) {
                uint64_t diff = tari_c29_check(nonce, mining_hash, edges);
                cycles++;
                bool is_share = diff >= target;
                if (is_share) shares++;
                printf("  nonce %llu: 42-cycle VERIFIED, C29 difficulty = %llu%s\n",
                       (unsigned long long)nonce, (unsigned long long)diff,
                       is_share ? "   *** SHARE (>= target) ***" : "");
                if (recall.is_open()) {
                    recall
                        << "{\"type\":\"cycle\",\"nonce\":"
                        << (unsigned long long)nonce
                        << ",\"difficulty\":" << (unsigned long long)diff
                        << ",\"edges\":[";
                    for (int i = 0; i < TARI_C29_PROOFSIZE; i++) {
                        if (i) recall << ',';
                        recall << edges[i];
                    }
                    recall << "]}\n";
                }
            } else {
                bugs++;
                printf("  nonce %llu: GPU returned a cycle but host verify FAILED (rc=%d) "
                       "<-- edge-math mismatch BUG\n", (unsigned long long)nonce, rc);
            }
        }
    };

    auto observe_trim = [&](tari_miner::SolverWatchdog &watchdog,
                            int context,
                            const SolverTrimResult &trim) {
        if (watchdog.observe(trim.nedges, trim.cuda_error == cudaSuccess))
            return true;
        if (trim.cuda_error != cudaSuccess)
            report_cuda_failure(device, context, "trim", trim.cuda_error);
        else
            report_zero_yield_failure(device, context);
        exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
        return false;
    };

    if (pipeline <= 1) {
      tari_miner::SolverWatchdog watchdog;
      write_recall_run();
      t0 = now_sec();
      for (uint64_t r = 0; r < count; r++) {
        uint64_t nonce = start_nonce + r;

        // (1) Tari key derivation -> inject straight into the GPU trimmer.
        inject_keys(ctx, nonce);

        SolverTrimResult trim = ctx->trim_copy_checked(device);
        if (trim.cuda_error == cudaSuccess)
            graphs++;
        if (!observe_trim(watchdog, 0, trim))
            break;
        if (trim.nedges) {
            int cycle_rc = ctx->findcycles_copied_status(trim.nedges);
            if (cycle_rc != cudaSuccess) {
                report_cuda_failure(device, 0, "cycle recovery", (cudaError_t)cycle_rc);
                exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
                break;
            }
        }
        consume_solutions(ctx, nonce);
      }
    } else {
        std::vector<SolverCtx*> contexts;
        contexts.push_back(ctx);
        for (int i = 1; i < pipeline; i++) {
            if (!have_device_memory_for_extra_solver_ctx(contexts[0])) {
                fprintf(stderr, "warning: not enough free VRAM for pipeline solver %d; using pipeline=%zu\n",
                        i, contexts.size());
                break;
            }
            SolverCtx *extra = create_solver_ctx(&params);
            if (!extra || !extra->trimmer.initsuccess) {
                fprintf(stderr, "warning: pipeline solver %d init failed; using pipeline=%zu. reason: %s\n",
                        i, contexts.size(), LAST_ERROR_REASON);
                cudaGetLastError();
                if (extra) destroy_solver_ctx(extra);
                break;
            }
            contexts.push_back(extra);
        }
        pipeline = (int)contexts.size();
        printf("solver pipeline=%d context%s\n", pipeline, pipeline == 1 ? "" : "s");
        write_recall_run();
        std::vector<tari_miner::SolverWatchdog> watchdogs((size_t)pipeline);
        for (size_t i = 1; i < contexts.size(); i++)
            extra_contexts.push_back(contexts[i]);
        t0 = now_sec();

        struct PendingTrim {
            std::future<SolverTrimResult> future;
            uint64_t nonce = 0;
            bool active = false;
        };
        std::vector<PendingTrim> pending((size_t)pipeline);
        uint64_t next = 0;
        auto launch_trim = [&](int slot) {
            SolverCtx *c = contexts[(size_t)slot];
            uint64_t nonce = start_nonce + next++;
            inject_keys(c, nonce);
            pending[(size_t)slot].nonce = nonce;
            pending[(size_t)slot].active = true;
            pending[(size_t)slot].future = std::async(std::launch::async, [c, device]() {
                return c->trim_copy_checked(device);
            });
        };

        auto drain_pending = [&]() {
            for (PendingTrim &item : pending) {
                if (item.active) {
                    item.future.get();
                    item.active = false;
                }
            }
        };

        int initial = (count < (uint64_t)pipeline) ? (int)count : pipeline;
        for (int slot = 0; slot < initial; slot++) launch_trim(slot);

        for (uint64_t done = 0; done < count; done++) {
            int slot = (int)(done % (uint64_t)pipeline);
            SolverCtx *c = contexts[(size_t)slot];
            SolverTrimResult trim = pending[(size_t)slot].future.get();
            pending[(size_t)slot].active = false;

            if (trim.cuda_error == cudaSuccess)
                graphs++;
            if (!observe_trim(watchdogs[(size_t)slot], slot, trim))
                break;
            if (trim.nedges) {
                int cycle_rc = c->findcycles_copied_status(trim.nedges);
                if (cycle_rc != cudaSuccess) {
                    report_cuda_failure(device, slot, "cycle recovery", (cudaError_t)cycle_rc);
                    exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
                    break;
                }
            }
            consume_solutions(c, pending[(size_t)slot].nonce);
            if (next < count)
                launch_trim(slot);
        }
        drain_pending();
    }

    double elapsed = now_sec() - t0;
    printf("\n--- summary ---\n");
    printf("graphs solved  : %llu in %.2f s  =>  %.3f graphs/s\n",
           (unsigned long long)graphs, elapsed, graphs / elapsed);
    printf("42-cycles found: %llu  (expected ~%.1f at 1/42 per graph)\n",
           (unsigned long long)cycles, graphs / 42.0);
    printf("verify failures: %llu  (MUST be 0)\n", (unsigned long long)bugs);
    printf("shares (>=%llu)  : %llu\n", (unsigned long long)target, (unsigned long long)shares);
    if (recall.is_open()) {
        recall
            << "{\"type\":\"summary\",\"graphs\":"
            << (unsigned long long)graphs
            << ",\"cycles\":" << (unsigned long long)cycles
            << ",\"verify_failures\":" << (unsigned long long)bugs
            << "}\n";
        recall.flush();
        if (!recall) {
            fprintf(stderr, "failed writing --recall-jsonl output: %s\n", recall_path);
            if (!exit_code)
                exit_code = 2;
        }
    }

    for (SolverCtx *extra : extra_contexts)
        destroy_solver_ctx(extra);
    destroy_solver_ctx(ctx);
    if (exit_code)
        return exit_code;
    return bugs ? 3 : 0;
}
