#pragma once

#include <cstddef>

namespace tari_miner {

// Runtime per-architecture trim tuning. The build scripts used to bake the
// sm_120 tuning into compile flags; these tables let one binary select a
// tuned parameter set at runtime from the detected compute capability, with
// explicit CLI overrides still winning over both. Only architectures with
// benchmarked values deviate from the reference profile — run
// tests/tune_sweep.sh on each GPU generation and add entries here when
// measurements justify it. Late-round TPB is already runtime via
// trimparams::trim.tpb; only the compile-time-pinned rounds need entries.
struct ArchTuning {
    int ntrims = 0;        // 0 = keep the compiled default (TARI_C29_DEFAULT_NTRIMS)
    int trim_tpb_r1 = 0;   // threads/block for round 1, 0 = keep build default
    int trim_tpb_r23 = 0;  // threads/block for rounds 2-3, 0 = keep build default
};

inline ArchTuning arch_tuning(int major, int minor) {
    ArchTuning t;
    const int cc = major * 10 + minor;
    if (cc >= 120) {
        // Blackwell: the tuning set previously baked into build flags.
        t.ntrims = 48;
        t.trim_tpb_r1 = 1024;
        t.trim_tpb_r23 = 960;
    } else if (cc >= 86) {
        // Ampere/Ada: keep the reference profile for now. Fill these in from
        // tests/tune_sweep.sh results per GPU model before shipping a change.
    }
    return t;
}

enum class DriverMode {
    Linux,
    WindowsWddm,
    WindowsTcc,
};

constexpr int MAX_PIPELINE = 5;
constexpr int AUTO_FALLBACK_PIPELINE = 2;
constexpr int WDDM_AUTO_CAP = 4;
constexpr size_t PIPELINE_MEMORY_RESERVE = 256ull << 20;

inline int clamp_pipeline_depth(int requested) {
    if (requested < 1) return 1;
    if (requested > MAX_PIPELINE) return MAX_PIPELINE;
    return requested;
}

inline int choose_pipeline_depth(
    bool explicit_set,
    int requested,
    DriverMode mode,
    size_t free_after_first_context,
    size_t bytes_per_context,
    bool memory_known = true
) {
    if (explicit_set)
        return clamp_pipeline_depth(requested);

    const int cap =
        mode == DriverMode::WindowsWddm ? WDDM_AUTO_CAP : MAX_PIPELINE;
    if (!memory_known || bytes_per_context == 0)
        return AUTO_FALLBACK_PIPELINE < cap ? AUTO_FALLBACK_PIPELINE : cap;

    size_t extra_contexts = 0;
    if (free_after_first_context > PIPELINE_MEMORY_RESERVE) {
        extra_contexts =
            (free_after_first_context - PIPELINE_MEMORY_RESERVE - 1) /
            bytes_per_context;
    }
    const size_t capped_extras =
        extra_contexts < (size_t)(cap - 1) ? extra_contexts : (size_t)(cap - 1);
    return 1 + (int)capped_extras;
}

} // namespace tari_miner
