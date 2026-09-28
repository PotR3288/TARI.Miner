#!/usr/bin/env bash
# Tuning sweep for TARI.Miner C29 trim parameters on a specific GPU.
#
# Runs the solver binary with each (ntrims, r1-tpb, r23-tpb) combination in
# NTRIMS_GRID x R1_GRID x R23_GRID order, parses graphs/s from the summary,
# and prints a ranked table plus CSV. verify failures MUST stay 0 for a row
# to be considered valid; rows with >0 are marked INVALID.
#
# Usage:
#   tests/tune_sweep.sh <solver-binary> [options]
#     --count N        nonce iterations per run (default 2000)
#     --ntrims a,b,c   grid of ntrims values (default "46,48,50")
#     --r1 a,b         round-1 trim TPB grid (default "768,960,1024"; 0 = build default)
#     --r23 a,b        rounds 2-3 trim TPB grid (default "768,960,1024"; 0 = build default)
#     --pipeline N     pipeline depth (default: binary's own auto value; pass to override)
#     --out FILE       CSV output path (default /tmp/tune_sweep_<ts>.csv)
#
# Example on a fleet GPU box:
#   tests/tune_sweep.sh bin/tari_c29_solver_sm_86 --count 3000 \
#       --ntrims 44,46,48,50 --r1 768,960,1024 --r23 768,960,1024

set -euo pipefail

BINARY="${1:-}"
if [[ -z "$BINARY" || ! -x "$BINARY" ]]; then
    echo "usage: $0 <solver-binary> [--count N] [--ntrims a,b,c] [--r1 a,b] [--r23 a,b] [--pipeline N] [--out FILE]" >&2
    exit 2
fi
shift

COUNT=2000
NTRIMS_GRID="46,48,50"
R1_GRID="768,960,1024"
R23_GRID="768,960,1024"
PIPELINE=""
OUT="/tmp/tune_sweep_$(date +%s).csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --count) COUNT="$2"; shift 2 ;;
        --ntrims) NTRIMS_GRID="$2"; shift 2 ;;
        --r1) R1_GRID="$2"; shift 2 ;;
        --r23) R23_GRID="$2"; shift 2 ;;
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -ra NTRIMS_ARR <<< "$NTRIMS_GRID"
IFS=',' read -ra R1_ARR <<< "$R1_GRID"
IFS=',' read -ra R23_ARR <<< "$R23_GRID"

echo "sweep: $BINARY  count=$COUNT  ntrims={$NTRIMS_GRID} r1={$R1_GRID} r23={$R23_GRID}"
echo "csv -> $OUT"
echo "ntrims,r1_tpb,r23_tpb,graphs_per_s,cycles,verify_failures,status" > "$OUT"

best_rate=0
best_row=""
total=$(( ${#NTRIMS_ARR[@]} * ${#R1_ARR[@]} * ${#R23_ARR[@]} ))
i=0
for ntrims in "${NTRIMS_ARR[@]}"; do
    for r1 in "${R1_ARR[@]}"; do
        for r23 in "${R23_ARR[@]}"; do
            i=$((i + 1))
            args=(--count "$COUNT" --ntrims "$ntrims")
            [[ "$r1" != "0" ]] && args+=(--trim-tpb-r1 "$r1")
            [[ "$r23" != "0" ]] && args+=(--trim-tpb-r23 "$r23")
            [[ -n "$PIPELINE" ]] && args+=(--pipeline "$PIPELINE")

            echo "[${i}/${total}] ntrims=$ntrims r1=${r1:-def} r23=${r23:-def}"
            log="/tmp/tune_sweep_run_${i}.log"
            if ! "$BINARY" "${args[@]}" > "$log" 2>&1; then
                echo "  RUN FAILED (see $log)"
                echo "$ntrims,$r1,$r23,0,0,?,FAILED" >> "$OUT"
                continue
            fi

            rate=$(grep -oP '=>\s+\K[0-9.]+' "$log" | head -1 || true)
            cycles=$(grep -oP '42-cycles found:\s+\K[0-9]+' "$log" | head -1 || echo 0)
            bugs=$(grep -oP 'verify failures:\s+\K[0-9]+' "$log" | head -1 || echo 0)

            if [[ -z "${rate:-}" ]]; then
                echo "  no rate parsed (see $log)"
                echo "$ntrims,$r1,$r23,0,${cycles:-0},${bugs:-?},NOPARSE" >> "$OUT"
                continue
            fi

            status=OK
            if [[ "${bugs:-0}" != "0" ]]; then status="INVALID"; fi
            printf '  rate=%s graphs/s cycles=%s verify_failures=%s %s\n' \
                "$rate" "${cycles:-?}" "${bugs:-?}" "$status"
            echo "$ntrims,$r1,$r23,$rate,${cycles:-0},${bugs:-0},$status" >> "$OUT"

            if [[ "$status" == "OK" ]]; then
                better=$(awk -v a="$rate" -v b="$best_rate" 'BEGIN{print (a+0 > b+0) ? 1 : 0}')
                if [[ "$better" == "1" ]]; then
                    best_rate="$rate"
                    best_row="ntrims=$ntrims r1=${r1:-def} r23=${r23:-def}"
                fi
            fi
        done
    done
done

echo
echo "--- ranked results ---"
awk -F, 'NR>1 && $7=="OK"' "$OUT" | sort -t, -k4 -rn | \
    awk -F, '{printf "  %6s graphs/s   ntrims=%-3s r1_tpb=%-5s r23_tpb=%s\n", $4, $1, $2, $3}'
echo
if [[ -n "$best_row" ]]; then
    echo "BEST: ${best_row}  (${best_rate} graphs/s)"
else
    echo "no valid rows — check the run logs in /tmp/tune_sweep_run_*.log"
fi
