// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2020 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include "cuckaroo.hpp"
#include "graph.hpp"
#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"
// [tari-c29] Phase 3: on-GPU endpoint->ID compression (opt-in TARI_C29_GPU_COMPRESS=1).
// CUDA 13 ships CUB under CCCL; keep the classic include for older toolchains.
#if __CUDACC_VER_MAJOR__ >= 13
#include <cccl/cub/cub.cuh>
#else
#include <cub/cub.cuh>
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint64_t u64; // save some typing

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#ifndef IDXSHIFT
#define IDXSHIFT 9
#endif
#endif

const u32 MAXEDGES = NEDGES >> IDXSHIFT;

#ifndef XBITS
#ifndef XBITS
#define XBITS 6
#endif
#endif

const u32 NX        = 1 << XBITS;
const u32 NX2       = NX * NX;
const u32 XMASK     = NX - 1;
const u32 YBITS     = XBITS;
const u32 NY        = 1 << YBITS;
const u32 YZBITS    = EDGEBITS - XBITS;
const u32 ZBITS     = YZBITS - YBITS;
const u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;

#ifndef NEPS_A
#define NEPS_A 133
#endif
#ifndef NEPS_B
#define NEPS_B 88
#endif
#define NEPS 128

const u32 EDGES_A = NZ * NEPS_A / NEPS;
const u32 EDGES_B = NZ * NEPS_B / NEPS;

const u32 ROW_EDGES_A = EDGES_A * NY;
const u32 ROW_EDGES_B = EDGES_B * NY;

#ifndef WARP_DST_ATOMICS
#define WARP_DST_ATOMICS 1
#endif
#ifndef WARP_DST_ATOMICS_LATE
#define WARP_DST_ATOMICS_LATE 0
#endif

#ifndef ROUND_U64_EDGE_LOAD
#define ROUND_U64_EDGE_LOAD 0
#endif
#ifndef ROUND1_COUNT_NORMAL_LOAD
#define ROUND1_COUNT_NORMAL_LOAD 0
#endif
#ifndef ROUND23_COUNT_NORMAL_LOAD
#define ROUND23_COUNT_NORMAL_LOAD 0
#endif
#ifndef FUSE_FINAL_TAIL_COUNT_NORMAL_LOAD
#define FUSE_FINAL_TAIL_COUNT_NORMAL_LOAD 0
#endif
#ifndef TARI_C29_DEFAULT_NTRIMS
#define TARI_C29_DEFAULT_NTRIMS 50
#endif
#ifndef ROUND_U64_EDGE_STORE
#define ROUND_U64_EDGE_STORE 1
#endif

#ifndef ROUND0_TPB
#define ROUND0_TPB 1024
#endif
#ifndef ROUND1_TPB
#define ROUND1_TPB 1024
#endif
#ifndef ROUND23_TPB
#define ROUND23_TPB 1024
#endif
#ifndef ROUND_LATE_TPB
#define ROUND_LATE_TPB 0
#endif

#ifndef SKIP_LATE_NULL_CHECKS
#define SKIP_LATE_NULL_CHECKS 1
#endif

#ifndef PINNED_EDGE_HOST
#define PINNED_EDGE_HOST 1
#endif
#ifndef TARI_TEST_FORCE_ZERO_YIELD
#define TARI_TEST_FORCE_ZERO_YIELD 0
#endif
#ifndef TARI_TEST_FORCE_CUDA_ERROR
#define TARI_TEST_FORCE_CUDA_ERROR 0
#endif

#ifndef RECOVERY_SMALL_OUTPUT
#define RECOVERY_SMALL_OUTPUT 0
#endif

#ifndef SEEDA_REHASH
#define SEEDA_REHASH 0
#endif

#ifndef GPUASSERT_RESET_ON_ERROR
#define GPUASSERT_RESET_ON_ERROR 0
#endif

#ifndef TRIM_STAGE_TIMING
#define TRIM_STAGE_TIMING 0
#endif

#ifndef SEEDB_REVERSE_LOOP
#define SEEDB_REVERSE_LOOP 0
#endif
#ifndef ROUND0_DST_HASH_DYNAMIC_BITS
#define ROUND0_DST_HASH_DYNAMIC_BITS 0
#endif
#ifndef ROUND0_DST_HASH_DYNAMIC_PROBES
#define ROUND0_DST_HASH_DYNAMIC_PROBES 8
#endif
#ifndef ROUND0_DST_HASH_FALLBACK_PLAIN
#define ROUND0_DST_HASH_FALLBACK_PLAIN 0
#endif
#ifndef ROUND0_DST_HASH_REPLAY_NORMAL_LOAD
#define ROUND0_DST_HASH_REPLAY_NORMAL_LOAD 0
#endif
#ifndef ROUND0_DST_HASH_REVERSE_INSERT
#define ROUND0_DST_HASH_REVERSE_INSERT 0
#endif
#ifndef FUSE_FINAL_TAIL_CURRENT
#define FUSE_FINAL_TAIL_CURRENT 0
#endif
// Number of Parts of BufferB, all but one of which will overlap BufferA
#ifndef NB
#define NB 2
#endif

#ifndef NA
#define NA  ((NB * NEPS_A + NEPS_B-1) / NEPS_B)
#endif

__constant__ uint2 e0 = {0,0};

__device__ u64 dipblock(const siphash_keys &keys, const word_t edge, u64 *buf) {
  diphash_state<> shs(keys);
  word_t edge0 = edge & ~EDGE_BLOCK_MASK;
  u32 i;
  for (i=0; i < EDGE_BLOCK_MASK; i++) {
    shs.hash24(edge0 + i);
    buf[i] = shs.xor_lanes();
  }
  shs.hash24(edge0 + i);
  buf[i] = 0;
  return shs.xor_lanes();
}

__device__ u32 endpoint(uint2 nodes, int uorv) {
  return uorv ? nodes.y : nodes.x;
}

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 8
#endif

template<int maxOut>
__global__ void SeedA(const siphash_keys sipkeys, ulonglong4 * __restrict__ buffer, u32 * __restrict__ indexes) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = gridDim.x * dim;
  const int FLUSHA2 = 2*FLUSHA;

  __shared__ uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  __shared__ int counters[NX];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);
#if !SEEDA_REHASH
  u64 buf[EDGE_BLOCK_SIZE];
#endif

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  __syncthreads();

  const int col = group % NX;
  const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
#if SEEDA_REHASH
    const word_t edge0 = nonce0 & ~EDGE_BLOCK_MASK;
    diphash_state<> lastState(sipkeys);
    for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++)
      lastState.hash24(edge0 + e);
    const u64 last = lastState.xor_lanes();
    diphash_state<> shs(sipkeys);
    for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++) {
      u64 edge;
      if (e < EDGE_BLOCK_MASK) {
        shs.hash24(edge0 + e);
        edge = shs.xor_lanes() ^ last;
      } else {
        edge = last;
      }
#else
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++) {
      u64 edge = buf[e] ^ last;
#endif
      u32 node0 = edge & EDGEMASK;
      u32 node1 = (edge >> 32) & EDGEMASK;
      int row = node0 >> YZBITS;
      int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
      tmp[row][counter] = make_uint2(node0, node1);
      __syncthreads();
      if (counter == FLUSHA-1) {
        int localIdx = min(FLUSHA2, counters[row]);
        int newCount = localIdx % FLUSHA;
        int nflush = localIdx - newCount;
        u32 grp = row * NX + col;
        int cnt = min((int)atomicAdd(indexes + grp, nflush), (int)(maxOut - nflush));
        for (int i = 0; i < nflush; i += TMPPERLL4)
          buffer[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
        for (int t = 0; t < newCount; t++) {
          tmp[row][t] = tmp[row][t + nflush];
        }
        counters[row] = newCount;
      }
      __syncthreads();
    }
  }
  uint2 zero = make_uint2(0, 0);
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    u32 grp = row * NX + col;
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      buffer[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
    }
  }
}

template <typename Edge> __device__ bool null(Edge e);

__device__ bool null(u32 nonce) {
  return nonce == 0;
}

__device__ bool null(uint2 nodes) {
  return nodes.x == 0 && nodes.y == 0;
}

#ifndef FLUSHB
#define FLUSHB 8
#endif

template<int maxOut>
__global__ void SeedB(const uint2 * __restrict__ source, ulonglong4 * __restrict__ destination, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int FLUSHB2 = 2 * FLUSHB;

  __shared__ uint2 tmp[NX][FLUSHB2];
  const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);
  __shared__ int counters[NX];

  for (int col = lid; col < NX; col += dim)
    counters[col] = 0;
  __syncthreads();
  const int row = group / NX;
  const int bucketEdges = min((int)srcIdx[group], (int)maxOut);
  const int loops = (bucketEdges + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    int col;
    int counter = 0;
#if SEEDB_REVERSE_LOOP
    const int edgeIndex = (loops - 1 - loop) * dim + lid;
#else
    const int edgeIndex = loop * dim + lid;
#endif
    if (edgeIndex < bucketEdges) {
      const int index = group * maxOut + edgeIndex;
      uint2 edge = __ldg(&source[index]);
      if (!null(edge)) {
        u32 node1 = edge.x;
        col = (node1 >> ZBITS) & XMASK;
        counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1)); // assuming COLS_LIMIT_LOSSES checked
        tmp[col][counter] = edge;
        }
    }
    __syncthreads();
    if (counter == FLUSHB-1) {
      int localIdx = min(FLUSHB2, counters[col]);
      int newCount = localIdx % FLUSHB;
      int nflush = localIdx - newCount;
      u32 grp = row * NX + col;
#ifdef SYNCBUG
      if (grp==0x2d6) printf("group %x size %d lid %d nflush %d\n", group, bucketEdges, lid, nflush);
#endif
      int cnt = min((int)atomicAdd(dstIdx + grp, nflush), (int)(maxOut - nflush));
      for (int i = 0; i < nflush; i += TMPPERLL4)
        destination[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
      for (int t = 0; t < newCount; t++) {
        tmp[col][t] = tmp[col][t + nflush];
      }
      counters[col] = newCount;
    }
    __syncthreads();
  }
  uint2 zero = make_uint2(0, 0);
  for (int col = lid; col < NX; col += dim) {
    int localIdx = min(FLUSHB2, counters[col]);
    u32 grp = row * NX + col;
#ifdef SYNCBUG
    if (group==0x2f2 && grp==0x2d6) printf("group %x size %d lid %d localIdx %d\n", group, bucketEdges, lid, localIdx);
#endif
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[col][j] = zero;
    for (int i = 0; i < localIdx; i += TMPPERLL4) {
      int cnt = min((int)atomicAdd(dstIdx + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
      destination[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
    }
  }
}

#ifndef ROUND_COUNTER_PAD
#define ROUND_COUNTER_PAD 1
#endif

constexpr int DUP_COUNTER_OFFSET = NZ / 32 + ROUND_COUNTER_PAD;
constexpr int ROUND_COUNTER_WORDS = NZ / 16 + ROUND_COUNTER_PAD;

__device__ __forceinline__  void Increase2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  u32 old = atomicOr(ecounters + word, mask) & mask;
  if (old)
    atomicOr(ecounters + word + DUP_COUNTER_OFFSET, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;

  return (ecounters[word + DUP_COUNTER_OFFSET] >> bit) & 1;
}

__device__ __forceinline__ uint2 LoadRoundEdge(const uint2 *src, const int index) {
#if ROUND_U64_EDGE_LOAD
  const u64 raw = __ldg(reinterpret_cast<const u64 *>(src) + index);
  return make_uint2((u32)raw, (u32)(raw >> 32));
#else
  return __ldg(&src[index]);
#endif
}

__device__ __forceinline__ void StoreRoundEdge(uint2 *dst, const int index, const uint2 edge) {
#if ROUND_U64_EDGE_STORE
  reinterpret_cast<u64 *>(dst)[index] = ((u64)edge.y << 32) | edge.x;
#else
  dst[index] = edge;
#endif
}

#if ROUND0_DST_HASH_DYNAMIC_BITS
constexpr u32 ROUND0_DST_HASH_EMPTY = 0xffffffffu;
constexpr int ROUND0_DST_HASH_DYNAMIC_SLOTS = 1 << ROUND0_DST_HASH_DYNAMIC_BITS;
constexpr int ROUND0_DST_HASH_DYNAMIC_WORDS = ROUND_COUNTER_WORDS + 3 * ROUND0_DST_HASH_DYNAMIC_SLOTS;

__device__ __forceinline__ u32 Round0DstHashBucket(const u32 bucket) {
  u32 h = bucket ^ (bucket >> ROUND0_DST_HASH_DYNAMIC_BITS);
  return (h * 0x9e3779b1u) & (ROUND0_DST_HASH_DYNAMIC_SLOTS - 1);
}

__device__ __forceinline__ int Round0DstHashFindOrInsert(u32 *keys, const u32 bucket) {
  u32 slot0 = Round0DstHashBucket(bucket);
#pragma unroll
  for (int probe = 0; probe < ROUND0_DST_HASH_DYNAMIC_PROBES; probe++) {
    const u32 slot = (slot0 + probe) & (ROUND0_DST_HASH_DYNAMIC_SLOTS - 1);
    const u32 old = atomicCAS(keys + slot, ROUND0_DST_HASH_EMPTY, bucket);
    if (old == ROUND0_DST_HASH_EMPTY || old == bucket)
      return (int)slot;
  }
  return -1;
}

__device__ __forceinline__ int Round0DstHashFind(const u32 *keys, const u32 bucket) {
  u32 slot0 = Round0DstHashBucket(bucket);
#pragma unroll
  for (int probe = 0; probe < ROUND0_DST_HASH_DYNAMIC_PROBES; probe++) {
    const u32 slot = (slot0 + probe) & (ROUND0_DST_HASH_DYNAMIC_SLOTS - 1);
    const u32 key = keys[slot];
    if (key == bucket)
      return (int)slot;
    if (key == ROUND0_DST_HASH_EMPTY)
      return -1;
  }
  return -1;
}

__device__ __forceinline__ void StoreRoundDirectWarp(uint2 *dst, u32 *dstIdx, const int maxOut,
                                                     const u32 bucket, const uint2 edge,
                                                     const unsigned activeMask) {
#if WARP_DST_ATOMICS
  const unsigned lane = threadIdx.x & 31;
  const unsigned mask = __match_any_sync(activeMask, bucket);
  const unsigned rank = __popc(mask & ((1u << lane) - 1));
  const unsigned count = __popc(mask);
  const unsigned leader = __ffs(mask) - 1;
  u32 base = 0;
  if (lane == leader)
    base = atomicAdd(dstIdx + bucket, count);
  base = __shfl_sync(mask, base, leader);
  const u32 bktIdx = min(base + rank, (u32)(maxOut - 1));
  StoreRoundEdge(dst, bucket * maxOut + bktIdx, edge);
#else
  const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
  StoreRoundEdge(dst, bucket * maxOut + bktIdx, edge);
#endif
}

template<int maxIn, int maxOut>
__global__ void Round0DstHashDynamic(const uint2 * __restrict__ src, uint2 * __restrict__ dst,
                                     const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;

  extern __shared__ u32 smem[];
  u32 *ecounters = smem;
  u32 *keys = ecounters + ROUND_COUNTER_WORDS;
  u32 *counts = keys + ROUND0_DST_HASH_DYNAMIC_SLOTS;
  u32 *bases = counts + ROUND0_DST_HASH_DYNAMIC_SLOTS;

  for (int i = lid; i < ROUND_COUNTER_WORDS; i += dim)
    ecounters[i] = 0;
  for (int i = lid; i < ROUND0_DST_HASH_DYNAMIC_SLOTS; i += dim) {
    keys[i] = ROUND0_DST_HASH_EMPTY;
    counts[i] = 0;
    bases[i] = 0;
  }
  __syncthreads();

  const int edgesInBucket = min(srcIdx[group], (u32)maxIn);
  const int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = LoadRoundEdge(src, index);
      if (null(edge)) continue;
      Increase2bCounter(ecounters, edge.x & ZMASK);
    }
  }

  __syncthreads();

  for (int loop = 0; loop < loops; loop++) {
    const int pos = loop * dim + lid;
    const int lindex =
#if ROUND0_DST_HASH_REVERSE_INSERT
      edgesInBucket - 1 - pos;
#else
      pos;
#endif
    bool fallback = false;
    u32 bucket = 0;
    uint2 outEdge = make_uint2(0, 0);
    if (pos < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = LoadRoundEdge(src, index);
      if (!null(edge) && Read2bCounter(ecounters, edge.x & ZMASK)) {
        bucket = edge.y >> ZBITS;
        outEdge = make_uint2(edge.x, edge.y);
        const int slot = Round0DstHashFindOrInsert(keys, bucket);
        if (slot >= 0)
          atomicAdd(counts + slot, 1);
        else
          fallback = true;
      }
    }
#if ROUND0_DST_HASH_FALLBACK_PLAIN
    if (fallback) {
      const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), (u32)(maxOut - 1));
      StoreRoundEdge(dst, bucket * maxOut + bktIdx, outEdge);
    }
#else
    const unsigned fallbackMask = __ballot_sync(__activemask(), fallback);
    if (fallback)
      StoreRoundDirectWarp(dst, dstIdx, maxOut, bucket, outEdge, fallbackMask);
#endif
  }

  __syncthreads();

  for (int i = lid; i < ROUND0_DST_HASH_DYNAMIC_SLOTS; i += dim) {
    const u32 bucket = keys[i];
    if (bucket != ROUND0_DST_HASH_EMPTY) {
      const u32 count = counts[i];
      const u32 base = atomicAdd(dstIdx + bucket, count);
      bases[i] = base;
      counts[i] = 0;
    }
  }

  __syncthreads();

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
#if ROUND0_DST_HASH_REPLAY_NORMAL_LOAD
      uint2 edge = src[index];
#else
      uint2 edge = LoadRoundEdge(src, index);
#endif
      if (!null(edge) && Read2bCounter(ecounters, edge.x & ZMASK)) {
        const u32 bucket = edge.y >> ZBITS;
        const int slot = Round0DstHashFind(keys, bucket);
        if (slot >= 0) {
          const u32 rank = atomicAdd(counts + slot, 1);
          const u32 base = (u32)bases[slot];
          const u32 bktIdx = min(base + rank, (u32)(maxOut - 1));
          StoreRoundEdge(dst, bucket * maxOut + bktIdx, make_uint2(edge.x, edge.y));
        }
      }
    }
  }
}

#endif

template<int NP, int maxIn, int maxOut, int UORV, int CHECK_NULL>
__global__ void Round(const uint2 * __restrict__ src, uint2 * __restrict__ dst, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;

  __shared__ u32 ecounters[ROUND_COUNTER_WORDS];
  for (int i = lid; i < ROUND_COUNTER_WORDS; i += dim)
    ecounters[i] = 0;
  __syncthreads();

  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group], maxIn);
    // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
    const int loops = (edgesInBucket + dim-1) / dim;

    for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
        const int index = maxIn * group + lindex;
#if ROUND1_COUNT_NORMAL_LOAD || ROUND23_COUNT_NORMAL_LOAD
        uint2 edge;
        if constexpr ((ROUND1_COUNT_NORMAL_LOAD &&
                       NP == NB && maxIn == EDGES_B/NB && maxOut == EDGES_B/2 && UORV == 1) ||
                      (ROUND23_COUNT_NORMAL_LOAD &&
                       NP == 1 && ((maxIn == EDGES_B/2 && maxOut == EDGES_A/4 && UORV == 0) ||
                                   (maxIn == EDGES_A/4 && maxOut == EDGES_B/4 && UORV == 1))))
          edge = src[index];
        else
          edge = LoadRoundEdge(src, index);
#else
        uint2 edge = LoadRoundEdge(src, index);
#endif
        if constexpr (CHECK_NULL) {
          if (null(edge)) continue;
        }
        u32 node = UORV ? edge.y : edge.x;
        Increase2bCounter(ecounters, node & ZMASK);
      }
    }
  }

  __syncthreads();

  src -= NP * NX2 * maxIn; srcIdx -= NP * NX2;
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group], maxIn);
    const int loops = (edgesInBucket + dim-1) / dim;
    for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
        const int index = maxIn * group + lindex;
        uint2 edge = LoadRoundEdge(src, index);
        if constexpr (CHECK_NULL) {
          if (null(edge)) continue;
        }
        u32 node0 = UORV ? edge.y : edge.x;
        if (Read2bCounter(ecounters, node0 & ZMASK)) {
          u32 node1 = UORV ? edge.x : edge.y;
          const int bucket = node1 >> ZBITS;
          const uint2 outEdge = UORV ? make_uint2(node1, node0) : make_uint2(node0, node1);
#if WARP_DST_ATOMICS
          if constexpr (maxIn > EDGES_B/4 || WARP_DST_ATOMICS_LATE) {
          const unsigned lane = threadIdx.x & 31;
          const unsigned mask = __match_any_sync(__activemask(), bucket);
          const unsigned rank = __popc(mask & ((1u << lane) - 1));
          const unsigned count = __popc(mask);
          const unsigned leader = __ffs(mask) - 1;
          u32 base = 0;
          if (lane == leader)
            base = atomicAdd(dstIdx + bucket, count);
          base = __shfl_sync(mask, base, leader);
          const u32 bktIdx = min(base + rank, (u32)(maxOut - 1));
          StoreRoundEdge(dst, bucket * maxOut + bktIdx, outEdge);
          } else {
          const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
          StoreRoundEdge(dst, bucket * maxOut + bktIdx, outEdge);
          }
#else
          const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
          StoreRoundEdge(dst, bucket * maxOut + bktIdx, outEdge);
#endif
        }
      }
    }
  }
}

template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const u32 *srcIdx, u32 *dstIdx) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  int myEdges = srcIdx[group];
  __shared__ int destIdx;

  if (lid == 0)
    destIdx = atomicAdd(dstIdx, myEdges);
  __syncthreads();
  for (int i = lid; i < myEdges; i += dim)
    destination[destIdx + i] = source[group * maxIn + i];
}

#if FUSE_FINAL_TAIL_CURRENT
template<int maxIn>
__global__ void FusedFinalTail(const uint2 * __restrict__ src, uint2 * __restrict__ dst,
                               const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;

  __shared__ u32 ecounters[ROUND_COUNTER_WORDS];
  for (int i = lid; i < ROUND_COUNTER_WORDS; i += dim)
    ecounters[i] = 0;
  __syncthreads();

  const int edgesInBucket = min(srcIdx[group], (u32)maxIn);
  const int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
#if FUSE_FINAL_TAIL_COUNT_NORMAL_LOAD
      uint2 edge = src[index];
#else
      uint2 edge = LoadRoundEdge(src, index);
#endif
      Increase2bCounter(ecounters, edge.y & ZMASK);
    }
  }

  __syncthreads();

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    bool keep = false;
    uint2 edge = make_uint2(0, 0);
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      edge = LoadRoundEdge(src, index);
      keep = Read2bCounter(ecounters, edge.y & ZMASK);
    }
    const unsigned activeMask = __activemask();
    const unsigned keepMask = __ballot_sync(activeMask, keep);
    if (keep) {
      const unsigned lane = threadIdx.x & 31;
      const unsigned rank = __popc(keepMask & ((1u << lane) - 1));
      const unsigned count = __popc(keepMask);
      const unsigned leader = __ffs(keepMask) - 1;
      u32 base = 0;
      if (lane == leader)
        base = atomicAdd(dstIdx, count);
      base = __shfl_sync(keepMask, base, leader);
      dst[base + rank] = edge;
    }
  }
}

#endif

// [tari-c29 portability patch] GNU statement-expressions -> do/while for MSVC host compiler
#define checkCudaErrors_V(ans) do { if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return; } while(0)
#define checkCudaErrors_N(ans) do { if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return NULL; } while(0)
#define checkCudaErrors(ans) do { int retval = gpuAssert((ans), __FILE__, __LINE__); if (retval != cudaSuccess) return retval; } while(0)

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  int device_id;
  cudaGetDevice(&device_id);
  if (code != cudaSuccess) {
    snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d", device_id, cudaGetErrorString(code), file, line);
#if GPUASSERT_RESET_ON_ERROR
    cudaDeviceReset();
#endif
    if (abort) return code;
  }
  return code;
}

// [tari-c29] Batched recovery: one full-graph scan resolves the nonces for
// ALL candidate cycles at once, instead of launching a separate 2^EDGEBITS-edge
// scan per candidate (up to MAXSOLS redundant scans). Output slots are
// pre-filled with NONCE_UNRESOLVED; resolved positions store nonce+1 so that a
// genuine edge with nonce 0 is distinguishable from an unresolved position.
#define RECOVERY_NONCE_UNRESOLVED 0xFFFFFFFFu
// Candidate edges travel BY VALUE inside this struct (1344B; total param
// space ~1.4KB < 4KB limit). A bare array parameter would decay to a host
// pointer and the kernel would dereference it in device address space —
// structs are copied into kernel param space, arrays are not.
struct RecoveryCandEdges { uint2 e[MAXSOLS][PROOFSIZE]; };
__global__ void RecoveryBatch(const siphash_keys sipkeys, ulonglong4 *buffer,
                              const RecoveryCandEdges cand, int ncands, u32 *noncesOut) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __shared__ u32 nonces[MAXSOLS][PROOFSIZE];
  u64 buf[EDGE_BLOCK_SIZE];
  for (int c = 0; c < MAXSOLS; c++) {
    if (lid < PROOFSIZE) nonces[c][lid] = 0;
  }
  __syncthreads();
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    u32 nonce0 = gid * loops + blk;
    const u64 last = dipblock(sipkeys, nonce0, buf);
    for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
      u64 edge = buf[i] ^ last;
      u32 u = edge & EDGEMASK;
      u32 v = (edge >> 32) & EDGEMASK;
      for (int c = 0; c < ncands; c++) { //YO
        const uint2 *ce = &cand.e[c][0];
        for (int p = 0; p < PROOFSIZE; p++) {
          if (ce[p].x == u && ce[p].y == v) {
            nonces[c][p] = nonce0 + i + 1; // +1: keep nonce 0 distinguishable from "no match"
          }
        }
      }
    }
  }
  __syncthreads();
  if (lid < PROOFSIZE) {
    // Only the block whose partition contains the matching edge has a nonzero
    // value; zero-check keeps other blocks from clobbering it in global memory.
    for (int c = 0; c < ncands; c++) {
      if (nonces[c][lid] > 0)
        noncesOut[c * PROOFSIZE + lid] = nonces[c][lid] - 1;
    }
  }
}

struct blockstpb {
  u16 blocks;
  u16 tpb;
};

struct trimparams {
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;
  // Runtime per-arch overrides for the compile-time-pinned round TPBs.
  // 0 = fall back to the ROUND1_TPB / ROUND23_TPB build defaults.
  u16 trim_tpb_r1 = 0;
  u16 trim_tpb_r23 = 0;

  trimparams() {
    ntrims              = TARI_C29_DEFAULT_NTRIMS;
    genA.blocks         = 32768;
    genA.tpb            =  128;
    genB.blocks         =  NX2;
    genB.tpb            =  128;
    trim.blocks         =  NX2;
    trim.tpb            =  320;
    tail.blocks         =  NX2;
    tail.tpb            = 1024;
    recover.blocks      = 1024;
    recover.tpb         = 1024;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt = nullptr;
  size_t sizeA, sizeB;
  const size_t indexesSize = NX * NY * sizeof(u32);
  u8 *bufferA = nullptr;
  u8 *bufferB = nullptr;
  u8 *bufferAB = nullptr;
  u32 *indexesE[1+NB] = {};
  u32 nedges = 0;
  cudaError_t trim_error = cudaSuccess;
  siphash_keys sipkeys;
  bool abort = false;
  bool initsuccess = false;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
    checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
    for (int i = 0; i < 1+NB; i++) {
      checkCudaErrors_V(cudaMalloc((void**)&indexesE[i], indexesSize));
    }
    sizeA = ROW_EDGES_A * NX * sizeof(uint2);
    sizeB = ROW_EDGES_B * NX * sizeof(uint2);
    const size_t bufferSize = sizeA + sizeB / NB;
    assert(bufferSize >= sizeB + sizeB / NB / 2); // ensure enough space for Round 1
    checkCudaErrors_V(cudaMalloc((void**)&bufferA, bufferSize));
    bufferAB = bufferA + sizeB / NB;
    bufferB  = bufferA + bufferSize - sizeB;
    assert(bufferA + sizeA == bufferB + sizeB * (NB-1) / NB); // ensure alignment of overlap
#if ROUND0_DST_HASH_DYNAMIC_BITS
    checkCudaErrors_V(cudaFuncSetAttribute(Round0DstHashDynamic<EDGES_A, EDGES_B/NB>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           ROUND0_DST_HASH_DYNAMIC_WORDS * sizeof(u32)));
#endif
    checkCudaErrors_V(cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice));
    initsuccess = true;
  }
  u64 globalbytes() const {
    return (sizeA+sizeB/NB) + (1+NB) * indexesSize + sizeof(edgetrimmer);
  }
  ~edgetrimmer() {
    if (bufferA)
      checkCudaErrors_V(cudaFree(bufferA));
    for (int i = 0; i < 1+NB; i++) {
      if (indexesE[i])
        checkCudaErrors_V(cudaFree(indexesE[i]));
    }
    if (dt)
      checkCudaErrors_V(cudaFree(dt));
  }
  u32 trim() {
    nedges = 0;
    trim_error = cudaSuccess;
    u32 qA = 0;
    u32 qE = 0;
#if TRIM_STAGE_TIMING
    cudaEvent_t timingStart, timingStop;
    checkCudaErrors(cudaEventCreate(&timingStart));
    checkCudaErrors(cudaEventCreate(&timingStop));
    float timingSeedA = 0.0f, timingSeedB = 0.0f, timingRound0 = 0.0f;
    float timingRound1 = 0.0f, timingRound2 = 0.0f, timingRound3 = 0.0f;
    float timingLate = 0.0f, timingTail = 0.0f;
#define TARI_TIMING_BEGIN() cudaEventRecord(timingStart, 0)
#define TARI_TIMING_END(dst) do { \
      cudaEventRecord(timingStop, 0); \
      cudaEventSynchronize(timingStop); \
      cudaEventElapsedTime(&(dst), timingStart, timingStop); \
    } while (0)
#else
#define TARI_TIMING_BEGIN() do {} while (0)
#define TARI_TIMING_END(dst) do {} while (0)
#endif
#if SQUASH_OUTPUT
    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[1], 0, indexesSize);
    SeedA<EDGES_A><<<tp.genA.blocks, tp.genA.tpb>>>(sipkeys, (ulonglong4*)bufferAB, indexesE[1]);
    TARI_TIMING_END(timingSeedA);
    if (abort) return false;

    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[0], 0, indexesSize);
    qA = sizeA/NA;
    qE = NX2 / NA;
    for (u32 i = 0; i < NA; i++) {
      SeedB<EDGES_A><<<tp.genB.blocks/NA, tp.genB.tpb>>>((uint2*)(bufferAB+i*qA), (ulonglong4*)(bufferA+i*qA), indexesE[1]+i*qE, indexesE[0]+i*qE);
      if (abort) return false;
    }
    TARI_TIMING_END(timingSeedB);
#else
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));

    cudaDeviceSynchronize();
    float durationA, durationB;
    cudaEventRecord(start, NULL);

    cudaMemset(indexesE[1], 0, indexesSize);

    SeedA<EDGES_A><<<tp.genA.blocks, tp.genA.tpb>>>(sipkeys, (ulonglong4*)bufferAB, indexesE[1]);

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
    if (abort) return false;
    cudaEventRecord(start, NULL);

    cudaMemset(indexesE[0], 0, indexesSize);

    qA = sizeA/NA;
    qE = NX2 / NA;
    for (u32 i = 0; i < NA; i++) {
      SeedB<EDGES_A><<<tp.genB.blocks/NA, tp.genB.tpb>>>((uint2*)(bufferAB+i*qA), (ulonglong4*)(bufferA+i*qA), indexesE[1]+i*qE, indexesE[0]+i*qE);
      if (abort) return false;
    }

    checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
    checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(stop));
    print_log("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
    if (abort) return false;
#endif

    TARI_TIMING_BEGIN();
    for (u32 i = 0; i < NB; i++) cudaMemset(indexesE[1+i], 0, indexesSize);

    qA = sizeA/NB;
    const size_t qB = sizeB/NB;
    qE = NX2 / NB;
    for (u32 i = NB; i--; ) {
#if ROUND0_DST_HASH_DYNAMIC_BITS
      Round0DstHashDynamic<EDGES_A, EDGES_B/NB><<<tp.trim.blocks/NB, ROUND0_TPB ? ROUND0_TPB : tp.trim.tpb, ROUND0_DST_HASH_DYNAMIC_WORDS * sizeof(u32)>>>((uint2*)(bufferA+i*qA), (uint2*)(bufferB+i*qB), indexesE[0]+i*qE, indexesE[1+i]);
#else
      Round<1, EDGES_A, EDGES_B/NB, 0, 1><<<tp.trim.blocks/NB, ROUND0_TPB ? ROUND0_TPB : tp.trim.tpb>>>((uint2*)(bufferA+i*qA), (uint2*)(bufferB+i*qB), indexesE[0]+i*qE, indexesE[1+i]); // to .632
#endif
      if (abort) return false;
    }
    TARI_TIMING_END(timingRound0);

    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[0], 0, indexesSize);

    Round<NB, EDGES_B/NB, EDGES_B/2, 1, !SKIP_LATE_NULL_CHECKS><<<tp.trim.blocks, tp.trim_tpb_r1 ? tp.trim_tpb_r1 : (ROUND1_TPB ? ROUND1_TPB : tp.trim.tpb)>>>((const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .296
    if (abort) return false;
    TARI_TIMING_END(timingRound1);

    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[1], 0, indexesSize);

    Round<1, EDGES_B/2, EDGES_A/4, 0, !SKIP_LATE_NULL_CHECKS><<<tp.trim.blocks, tp.trim_tpb_r23 ? tp.trim_tpb_r23 : (ROUND23_TPB ? ROUND23_TPB : tp.trim.tpb)>>>((const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]); // to .176
    if (abort) return false;
    TARI_TIMING_END(timingRound2);

    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[0], 0, indexesSize);

    Round<1, EDGES_A/4, EDGES_B/4, 1, !SKIP_LATE_NULL_CHECKS><<<tp.trim.blocks, tp.trim_tpb_r23 ? tp.trim_tpb_r23 : (ROUND23_TPB ? ROUND23_TPB : tp.trim.tpb)>>>((const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .117
    if (abort) return false;
    TARI_TIMING_END(timingRound3);

#if !SQUASH_OUTPUT
    cudaDeviceSynchronize();
#endif

    TARI_TIMING_BEGIN();
    for (int round = 4; round < tp.ntrims; round += 2) {
      cudaMemset(indexesE[1], 0, indexesSize);
      Round<1, EDGES_B/4, EDGES_B/4, 0, !SKIP_LATE_NULL_CHECKS><<<tp.trim.blocks, ROUND_LATE_TPB ? ROUND_LATE_TPB : tp.trim.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]);
      if (abort) return false;
#if FUSE_FINAL_TAIL_CURRENT
      if (round + 2 >= tp.ntrims) {
        cudaMemset(indexesE[0], 0, indexesSize);
        FusedFinalTail<EDGES_B/4><<<tp.trim.blocks, ROUND_LATE_TPB ? ROUND_LATE_TPB : tp.trim.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]);
        if (abort) return false;
        break;
      }
#endif
      cudaMemset(indexesE[0], 0, indexesSize);
      Round<1, EDGES_B/4, EDGES_B/4, 1, !SKIP_LATE_NULL_CHECKS><<<tp.trim.blocks, ROUND_LATE_TPB ? ROUND_LATE_TPB : tp.trim.tpb>>>((const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]);
      if (abort) return false;
    }
    TARI_TIMING_END(timingLate);

#if FUSE_FINAL_TAIL_CURRENT
    TARI_TIMING_BEGIN();
    trim_error =
        cudaMemcpy(&nedges, indexesE[0], sizeof(u32), cudaMemcpyDeviceToHost);
    TARI_TIMING_END(timingTail);
#else
    TARI_TIMING_BEGIN();
    cudaMemset(indexesE[1], 0, indexesSize);
#if !SQUASH_OUTPUT
    cudaDeviceSynchronize();
#endif
    Tail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]);
    trim_error =
        cudaMemcpy(&nedges, indexesE[1], sizeof(u32), cudaMemcpyDeviceToHost);
    TARI_TIMING_END(timingTail);
#endif
#if !SQUASH_OUTPUT
    cudaDeviceSynchronize();
#endif
#if TRIM_STAGE_TIMING
    printf("stage-ms SeedA %.3f SeedB %.3f R0 %.3f R1 %.3f R2 %.3f R3 %.3f late %.3f tail %.3f\n",
           timingSeedA, timingSeedB, timingRound0, timingRound1, timingRound2,
           timingRound3, timingLate, timingTail);
    checkCudaErrors(cudaEventDestroy(timingStart));
    checkCudaErrors(cudaEventDestroy(timingStop));
#endif
#undef TARI_TIMING_BEGIN
#undef TARI_TIMING_END
    return trim_error == cudaSuccess ? nedges : 0;
  }
};

// [tari-c29] Phase 3: on-GPU endpoint->ID compression. The host walk's cost is
// dominated by sequential linear-probing hash compression (measured ~4.6M probes,
// ~250 ms/attempt), NOT cycle-finding (DFS triggers only a few times per attempt).
// Node IDs are pure labels: cycles_with_link uses only adjacency structure, recovery
// stores edge indices (not node IDs), and verify() re-derives every proof via SipHash.
// So any consistent endpoint->ID relabeling is safe; an INJECTIVE one is strictly
// safer than the lossy host compressor (~18% false merges). Per partition, distinct
// endpoints <= nedges <= MAXEDGES == MAXNODES, so dense injective IDs always fit.
#define GPU_COMPRESS_MIN_EDGES 50000

__global__ void split_edge_side(const uint2 *edges, u32 *vals, int which, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) vals[i] = which ? edges[i].y : edges[i].x;
}

__global__ void init_perm(u32 *perm, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) perm[i] = (u32)i;
}

// flag[i] = 1 iff sorted[i] starts a new distinct value (first or differs from prev)
__global__ void make_flags(const u32 *sorted, u32 *flags, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) flags[i] = (i == 0 || sorted[i] != sorted[i-1]) ? 1u : 0u;
}

// comp[2*i + which] = rank_at_pos[i]: scatter dense ranks back to original edge
// order in an INTERLEAVED {x_id, y_id} layout so one nedges*sizeof(uint2) DtoH
// copy carries both partitions.
__global__ void scatter_ranks(const u32 *perm_out, const u32 *rank_at_pos, u32 *comp, int which, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) comp[2*i + which] = rank_at_pos[i];
}

struct SolverTrimResult {
  u32 nedges = 0;
  cudaError_t cuda_error = cudaSuccess;
};

struct solver_ctx {
  edgetrimmer trimmer;
  bool mutatenonce;
  uint2 *edges;
  bool pinned_edges;
  graph<word_t> cg;
  std::vector<u32> sols; // concatenation of all proof's indices
#if RECOVERY_SMALL_OUTPUT
  u32 *recoverIndexes;
#endif

  // [tari-c29] Phase 3: on-GPU endpoint->ID compression scratch. Allocated only
  // when TARI_C29_GPU_COMPRESS=1 (default off). d_comp is INTERLEAVED:
  // d_comp[2*i] = dense injective ID for edge[i].x, d_comp[2*i+1] = ID for .y.
  bool gpu_compression = false;
  u32 *d_side_vals = nullptr, *d_sorted = nullptr, *d_perm = nullptr, *d_perm_out = nullptr, *d_rank = nullptr;
  u32 *d_comp = nullptr;
  void *d_cub_temp = nullptr; size_t cub_sort_bytes = 0;
  void *d_scan_temp = nullptr; size_t scan_bytes = 0;
  u32 *h_comp_pinned = nullptr;

  bool gpu_compress_init() {
    if (getenv("TARI_C29_GPU_COMPRESS") == nullptr)
      return false; // default off: legacy host compression path
    const size_t b = (size_t)MAXEDGES * sizeof(u32);
    cudaError_t rc;
    rc  = cudaMalloc((void**)&d_side_vals, b);
    if (!rc) rc = cudaMalloc((void**)&d_sorted, b);
    if (!rc) rc = cudaMalloc((void**)&d_perm, b);
    if (!rc) rc = cudaMalloc((void**)&d_perm_out, b);
    if (!rc) rc = cudaMalloc((void**)&d_rank, b);
    if (!rc) rc = cudaMalloc((void**)&d_comp, 2 * b);
    // size CUB temp storage once at the worst case (n = MAXEDGES), then reuse
    if (!rc) {
      cub::DeviceRadixSort::SortPairs<u32,u32,int>(nullptr, cub_sort_bytes,
                                                   d_side_vals, d_sorted, d_perm, d_perm_out,
                                                   MAXEDGES, 0, 32);
      rc = cudaMalloc(&d_cub_temp, cub_sort_bytes);
    }
    if (!rc) {
      size_t sb = 0;
      cub::DeviceScan::ExclusiveSum<u32*,u32*>(nullptr, sb, d_rank, d_rank, MAXEDGES);
      rc = cudaMalloc(&d_scan_temp, sb);
      scan_bytes = sb;
    }
    if (!rc)
      rc = cudaMallocHost((void**)&h_comp_pinned, 2 * b);
    if (rc != cudaSuccess) {
      // partial-init cleanup: free whatever was allocated so far
      gpu_compression = false;
      cudaFree(d_side_vals); d_side_vals = nullptr;
      cudaFree(d_sorted); d_sorted = nullptr;
      cudaFree(d_perm); d_perm = nullptr;
      cudaFree(d_perm_out); d_perm_out = nullptr;
      cudaFree(d_rank); d_rank = nullptr;
      cudaFree(d_comp); d_comp = nullptr;
      cudaFree(d_cub_temp); d_cub_temp = nullptr;
      cudaFree(d_scan_temp); d_scan_temp = nullptr;
      if (h_comp_pinned) { cudaFreeHost(h_comp_pinned); h_comp_pinned = nullptr; }
      return false;
    }
    gpu_compression = true;
    return true;
  }

  // Fills h_comp_pinned with dense injective endpoint IDs in INTERLEAVED
  // {x_id, y_id} order (one uint2 per edge). Returns false (caller falls back to
  // the host compression path) when the feature is off, the graph is too small
  // to be worth it, or any CUDA/CUB step fails.
  bool gpu_compress(const uint2 *edges, u32 nedges) {
    if (!gpu_compression || nedges < GPU_COMPRESS_MIN_EDGES)
      return false;
    const int blocks = (nedges + 255) / 256;
    cudaError_t rc;
    for (int which = 0; which < 2; which++) {
      split_edge_side<<<blocks, 256>>>(edges, d_side_vals, which, nedges);
      init_perm<<<blocks, 256>>>(d_perm, nedges);
      rc = cub::DeviceRadixSort::SortPairs<u32,u32,int>(d_cub_temp, cub_sort_bytes,
                                                        d_side_vals, d_sorted, d_perm, d_perm_out,
                                                        nedges, 0, 32);
      if (rc != cudaSuccess) return false;
      make_flags<<<blocks, 256>>>(d_sorted, d_rank, nedges);
      rc = cub::DeviceScan::ExclusiveSum<u32*,u32*>(d_scan_temp, scan_bytes, d_rank, d_rank, nedges);
      if (rc != cudaSuccess) return false;
      scatter_ranks<<<blocks, 256>>>(d_perm_out, d_rank, d_comp, which, nedges);
    }
    rc = cudaMemcpy(h_comp_pinned, d_comp, (size_t)nedges * sizeof(uint2), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) return false;
    return true;
  }

  solver_ctx(const trimparams tp, bool mutate_nonce) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
    pinned_edges = false;
#if PINNED_EDGE_HOST
    cudaError_t pin_rc = cudaMallocHost((void **)&edges, sizeof(uint2) * (size_t)MAXEDGES);
    if (pin_rc == cudaSuccess) {
      pinned_edges = true;
    } else {
      // Pageable memory is an intentional fallback, so do not let the handled
      // allocation error poison the first per-trim CUDA health check.
      cudaGetLastError();
      edges = new uint2[MAXEDGES];
    }
#else
    edges = new uint2[MAXEDGES];
#endif
#if RECOVERY_SMALL_OUTPUT
    recoverIndexes = nullptr;
    // Batched recovery writes up to MAXSOLS proofs (one per candidate cycle).
    checkCudaErrors_V(cudaMalloc((void **)&recoverIndexes, (size_t)MAXSOLS * PROOFSIZE * sizeof(u32)));
#endif
    gpu_compress_init(); // [tari-c29] Phase 3; no-op unless TARI_C29_GPU_COMPRESS=1
    mutatenonce = mutate_nonce;
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    if (mutatenonce)
      ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer.sipkeys);
    sols.clear();
  }
  ~solver_ctx() {
#if RECOVERY_SMALL_OUTPUT
    cudaFree(recoverIndexes);
#endif
    if (gpu_compression) { // [tari-c29] Phase 3 scratch
      cudaFree(d_side_vals);
      cudaFree(d_sorted);
      cudaFree(d_perm);
      cudaFree(d_perm_out);
      cudaFree(d_rank);
      cudaFree(d_comp);
      cudaFree(d_cub_temp);
      cudaFree(d_scan_temp);
      if (h_comp_pinned)
        cudaFreeHost(h_comp_pinned);
    }
    if (pinned_edges)
      cudaFreeHost(edges);
    else
      delete[] edges;
  }

  int findcycles_with_keys(uint2 *edges, u32 nedges, const siphash_keys &keys, std::vector<u32> &outSols) {
    // [tari-c29] Upper-bound experiment: skip the host walk entirely to measure
    // how much of the current rate is CPU-walk-limited vs GPU-trim-limited.
    static const bool skip_walk = getenv("TARI_C29_SKIP_WALK") != nullptr;
    if (skip_walk)
      return 0;
    // [tari-c29] Host-walk timing: TARI_C29_WALK_TIMING=1 logs per-attempt
    // graph-construction cost (the candidate for on-GPU cycle-finding).
    static const bool walk_timing = getenv("TARI_C29_WALK_TIMING") != nullptr;
    u64 t_walk0 = 0, t_walk1 = 0;
    if (walk_timing) t_walk0 = timestamp();
    // [tari-c29] Phase 3: on-GPU endpoint->ID compression. When enabled and it
    // succeeds, the host walk becomes a plain O(nedges) adjacency build with
    // pre-computed dense injective IDs — no per-edge hash lookups (the measured
    // ~4.6M-probe / ~250 ms cost). Falls back to the legacy path on any failure.
    bool gpu_ids = gpu_compress(edges, nedges);
    cg.reset(gpu_ids); // skip the two 8MB compressor-table resets only on the GPU-ID path (they are unused there)
    if (gpu_ids) {
      for (u32 i = 0; i < nedges; i++)
        cg.add_edge(h_comp_pinned[2*i], h_comp_pinned[2*i + 1]); // interleaved {x_id, y_id}
    } else {
      for (u32 i = 0; i < nedges; i++)
        cg.add_compress_edge(edges[i].x, edges[i].y);
    }
    if (!cg.nsols) {
      if (walk_timing && nedges > 100000) { // only log non-trivial graphs
        t_walk1 = timestamp();
        fprintf(stderr, "host-walk edges %d nsols 0 time %.2f ms probes %.3fM dfs %u\n",
                nedges, (t_walk1 - t_walk0) / 1e6,
                (cg.compressu->probes + cg.compressv->probes) / 1e6, (unsigned)cg.dfs_triggers);
      }
      return 0;
    }
    if (walk_timing) t_walk1 = timestamp(); // host walk done; recovery phase starts here

    // [tari-c29] Batched recovery: collect every candidate cycle first, then
    // resolve all of their nonces in a single full-graph scan. The old code
    // launched one 2^EDGEBITS-edge Recovery pass per candidate (up to MAXSOLS
    // redundant scans) with a host round-trip between each. Candidates that do
    // not fully resolve are dropped here; run_solver's verify() re-derives the
    // proof from nonces independently, so nothing half-resolved can leak out.
    RecoveryCandEdges cand; // passed BY VALUE — arrays decay to host pointers
    const int ncands = (int)(cg.nsols < MAXSOLS ? cg.nsols : MAXSOLS);
    for (u32 s = 0; s < (u32)ncands; s++) {
      for (u32 j = 0; j < PROOFSIZE; j++)
        cand.e[s][j] = edges[cg.sols[s][j]];
    }

#if RECOVERY_SMALL_OUTPUT
    const size_t outBytes = (size_t)MAXSOLS * PROOFSIZE * sizeof(u32);
#else
    const size_t outBytes = (size_t)ncands * PROOFSIZE * sizeof(u32);
#endif
    cudaError_t rc = cudaSuccess; // sentinel fill below primes every output slot
    {
      u32 *fill = new u32[outBytes / sizeof(u32)];
      for (size_t i = 0; i < outBytes / sizeof(u32); i++) fill[i] = RECOVERY_NONCE_UNRESOLVED;
#if RECOVERY_SMALL_OUTPUT
      rc = cudaMemcpy(recoverIndexes, fill, outBytes, cudaMemcpyHostToDevice);
#else
      rc = cudaMemcpy(trimmer.indexesE[1], fill, outBytes, cudaMemcpyHostToDevice);
#endif
      delete[] fill;
    }
    u32 *noncesOut;
#if RECOVERY_SMALL_OUTPUT
    noncesOut = recoverIndexes;
#else
    noncesOut = (u32 *)trimmer.indexesE[1];
#endif
    if (rc == cudaSuccess) {
      RecoveryBatch<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>(keys, (ulonglong4*)trimmer.bufferA, cand, ncands, noncesOut);
      rc = cudaGetLastError();
    }
    // Recovery uses the calling thread's default stream. Synchronizing that
    // stream preserves overlap with trims running in other host threads.
    if (rc == cudaSuccess)
      rc = cudaStreamSynchronize(0);
    if (rc != cudaSuccess)
      return gpuAssert(rc, __FILE__, __LINE__);

    u32 *noncesHost = new u32[(size_t)MAXSOLS * PROOFSIZE]; // always host-side, flat
#if RECOVERY_SMALL_OUTPUT
    rc = cudaMemcpy(noncesHost, recoverIndexes, outBytes, cudaMemcpyDeviceToHost);
#else
    rc = cudaMemcpy(noncesHost, trimmer.indexesE[1], outBytes, cudaMemcpyDeviceToHost);
#endif
    if (rc != cudaSuccess) {
      delete[] noncesHost;
      return gpuAssert(rc, __FILE__, __LINE__);
    }

    for (u32 s = 0; s < (u32)ncands; s++) {
      bool complete = true;
      for (u32 j = 0; j < PROOFSIZE; j++) {
        if (noncesHost[s * PROOFSIZE + j] == RECOVERY_NONCE_UNRESOLVED) { // position never matched: not a real cycle
          complete = false;
          break;
        }
      }
      if (!complete)
        continue;
      const size_t solbase = outSols.size();
      outSols.resize(solbase + PROOFSIZE);
      memcpy(&outSols[solbase], &noncesHost[s * PROOFSIZE], PROOFSIZE * sizeof(u32));
      qsort(&outSols[solbase], PROOFSIZE, sizeof(u32), cg.nonce_cmp);
    }
    delete[] noncesHost;
    if (walk_timing) {
      u64 t_end = timestamp();
      fprintf(stderr, "host-walk edges %d nsols %u time %.2f ms recovery %.2f ms probes %.3fM dfs %u\n",
              nedges, cg.nsols, (t_walk1 - t_walk0) / 1e6, (t_end - t_walk1) / 1e6,
              (cg.compressu->probes + cg.compressv->probes) / 1e6, (unsigned)cg.dfs_triggers);
    }
    return 0;
  }

  int findcycles(uint2 *edges, u32 nedges) {
    return findcycles_with_keys(edges, nedges, trimmer.sipkeys, sols);
  }

  int solve() {
    u64 time0, time1;
    u32 timems,timems2;
    // [tari-c29] per-graph trim vs findcycles breakdown (TARI_C29_WALK_TIMING)
    static const bool solve_timing = getenv("TARI_C29_WALK_TIMING") != nullptr;

    time0 = timestamp();
    u32 nedges = trim_copy();
    if (!nedges)
      return 0;
    time1 = timestamp(); timems  = (time1 - time0) / 1000000;
    time0 = timestamp();
    if (findcycles_copied_status(nedges) != cudaSuccess)
      return 0;
    time1 = timestamp(); timems2 = (time1 - time0) / 1000000;
    print_log("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
    if (solve_timing)
      fprintf(stderr, "solve edges %u trim %.1f ms findcycles %.1f ms\n",
              nedges, (double)timems, (double)timems2);
    return sols.size() / PROOFSIZE;
  }

  u32 trim_copy() {
    return trim_copy_to(edges);
  }

  u32 trim_copy_to(uint2 *outEdges, cudaError_t *copy_error = nullptr) {
#if TRIM_STAGE_TIMING
    cudaEvent_t copyStart, copyStop;
    checkCudaErrors(cudaEventCreate(&copyStart));
    checkCudaErrors(cudaEventCreate(&copyStop));
#endif
    if (copy_error)
      *copy_error = cudaSuccess;
    trimmer.abort = false;
    u32 nedges = trimmer.trim();
    if (trimmer.trim_error != cudaSuccess) {
      if (copy_error)
        *copy_error = trimmer.trim_error;
      return 0;
    }
    if (!nedges) {
#if TRIM_STAGE_TIMING
      checkCudaErrors(cudaEventDestroy(copyStart));
      checkCudaErrors(cudaEventDestroy(copyStop));
#endif
      return 0;
    }
    if (nedges > MAXEDGES) {
      fprintf(stderr, "OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
    }
#if TRIM_STAGE_TIMING
    cudaEventRecord(copyStart, 0);
#endif
    cudaError_t edge_copy_error = cudaMemcpy(outEdges,
#if FUSE_FINAL_TAIL_CURRENT
               trimmer.bufferA,
#else
               trimmer.bufferB,
#endif
               sizeof(uint2) * (size_t)nedges, cudaMemcpyDeviceToHost); // [tari-c29 patch] VLA-sizeof -> explicit
    if (copy_error)
      *copy_error = edge_copy_error;
#if TRIM_STAGE_TIMING
    cudaEventRecord(copyStop, 0);
    cudaEventSynchronize(copyStop);
    float copyMs = 0.0f;
    cudaEventElapsedTime(&copyMs, copyStart, copyStop);
    printf("stage-ms EdgeCopy %.3f edges %u\n", copyMs, nedges);
    checkCudaErrors(cudaEventDestroy(copyStart));
    checkCudaErrors(cudaEventDestroy(copyStop));
#endif
    return edge_copy_error == cudaSuccess ? nedges : 0;
  }

  SolverTrimResult trim_copy_checked(int device) {
    return trim_copy_to_checked(edges, device);
  }

  SolverTrimResult trim_copy_to_checked(uint2 *outEdges, int device) {
    SolverTrimResult result;
    cudaError_t rc = cudaSetDevice(device);
    if (rc != cudaSuccess) {
      result.cuda_error = rc;
      return result;
    }

    // CUDA's last-error state belongs to the calling host thread. Inspect and
    // clear it here, before this same thread launches the trim.
    rc = cudaGetLastError();
    if (rc != cudaSuccess) {
      result.cuda_error = rc;
      return result;
    }

    cudaError_t copy_rc = cudaSuccess;
    result.nedges = trim_copy_to(outEdges, &copy_rc);

    // All trim work uses the calling thread's default stream. The release
    // build selects per-thread default streams, so this detects asynchronous
    // execution failures without serializing other pipeline slots.
    cudaError_t sync_rc = cudaStreamSynchronize(0);
    cudaError_t last_rc = cudaGetLastError();
    result.cuda_error = copy_rc != cudaSuccess
        ? copy_rc
        : (sync_rc != cudaSuccess ? sync_rc : last_rc);
#if TARI_TEST_FORCE_ZERO_YIELD
    result.nedges = 0;
#endif
#if TARI_TEST_FORCE_CUDA_ERROR
    // Exercise the same fatal path with a real CUDA runtime error rather than
    // assigning a synthetic status value. This hook is compiled out normally.
    result.cuda_error = cudaMemset(nullptr, 0, 1);
    if (result.cuda_error == cudaSuccess)
      result.cuda_error = cudaErrorInvalidValue;
#endif
    return result;
  }

  int findcycles_copied_status(u32 nedges) {
    return findcycles(edges, nedges);
  }

  int findcycles_copied(u32 nedges) {
    if (findcycles_copied_status(nedges) != cudaSuccess)
      return 0;
    return sols.size() / PROOFSIZE;
  }

  void abort() {
    trimmer.abort = true;
  }
};

static bool have_device_memory_for_extra_solver_ctx(const solver_ctx *ctx) {
  size_t freeMem = 0, totalMem = 0;
  cudaError_t rc = cudaMemGetInfo(&freeMem, &totalMem);
  if (rc != cudaSuccess) {
    cudaGetLastError();
    return true;
  }
  size_t need = ctx->trimmer.globalbytes();
#if RECOVERY_SMALL_OUTPUT
  need += PROOFSIZE * sizeof(u32);
#endif
  const size_t margin = 256ull << 20;
  return freeMem > need + margin;
}

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

typedef solver_ctx SolverCtx;

CALL_CONVENTION int run_solver(SolverCtx* ctx,
                               char* header,
                               int header_length,
                               u32 nonce,
                               u32 range,
                               SolverSolutions *solutions,
                               SolverStats *stats
                               )
{
  u64 time0, time1;
  u32 timems;
  u32 sumnsols = 0;
  int device_id;
  if (stats != NULL) {
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, stats->device_id);
    stats->device_id = device_id;
    stats->edge_bits = EDGEBITS;
    strncpy(stats->device_name, props.name, MAX_NAME_LEN);
  }

  if (ctx == NULL || !ctx->trimmer.initsuccess){
    print_log("Error initialising trimmer. Aborting.\n");
    print_log("Reason: %s\n", LAST_ERROR_REASON);
    if (stats != NULL) {
       stats->has_errored = true;
       strncpy(stats->error_reason, LAST_ERROR_REASON, MAX_NAME_LEN);
    }
    return 0;
  }

  for (u32 r = 0; r < range; r++) {
    time0 = timestamp();
    ctx->setheadernonce(header, header_length, nonce + r);
    print_log("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx->trimmer.sipkeys.k0, ctx->trimmer.sipkeys.k1, ctx->trimmer.sipkeys.k2, ctx->trimmer.sipkeys.k3);
    u32 nsols = ctx->solve();
    time1 = timestamp();
    timems = (time1 - time0) / 1000000;
    print_log("Time: %d ms\n", timems);
    for (unsigned s = 0; s < nsols; s++) {
      print_log("Solution");
      u32* prf = &ctx->sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        print_log(" %jx", (uintmax_t)prf[i]);
      print_log("\n");
      if (solutions != NULL){
        solutions->edge_bits = EDGEBITS;
        solutions->num_sols++;
        solutions->sols[sumnsols+s].nonce = nonce + r;
        for (u32 i = 0; i < PROOFSIZE; i++)
          solutions->sols[sumnsols+s].proof[i] = (u64) prf[i];
      }
      int pow_rc = verify(prf, ctx->trimmer.sipkeys);
      if (pow_rc == POW_OK) {
        print_log("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          print_log("%02x", cyclehash[i]);
        print_log("\n");
      } else {
        print_log("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
    if (stats != NULL) {
      stats->last_start_time = time0;
      stats->last_end_time = time1;
      stats->last_solution_time = time1 - time0;
    }
  }
  print_log("%d total solutions\n", sumnsols);
  return sumnsols > 0;
}

CALL_CONVENTION SolverCtx* create_solver_ctx(SolverParams* params) {
  trimparams tp;
  tp.ntrims = params->ntrims;
  tp.genA.blocks = params->genablocks;
  tp.genA.tpb = params->genatpb;
  tp.genB.tpb = params->genbtpb;
  tp.trim.tpb = params->trimtpb;
  tp.tail.tpb = params->tailtpb;
  tp.recover.blocks = params->recoverblocks;
  tp.recover.tpb = params->recovertpb;
  tp.trim_tpb_r1 = (u16)params->trimtpb_r1;
  tp.trim_tpb_r23 = (u16)params->trimtpb_r23;

  cudaDeviceProp prop;
  checkCudaErrors_N(cudaGetDeviceProperties(&prop, params->device));

  assert(tp.genA.tpb <= prop.maxThreadsPerBlock);
  assert(tp.genB.tpb <= prop.maxThreadsPerBlock);
  assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
  if (tp.trim_tpb_r1)
    assert(tp.trim_tpb_r1 <= prop.maxThreadsPerBlock);
  if (tp.trim_tpb_r23)
    assert(tp.trim_tpb_r23 <= prop.maxThreadsPerBlock);
  // assert(tp.tailblocks <= prop.threadDims[0]);
  assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
  assert(tp.recover.tpb <= prop.maxThreadsPerBlock);

  assert(tp.genA.blocks * tp.genA.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES
  assert(tp.recover.blocks * tp.recover.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES
  assert(NEDGES % (tp.genA.blocks * tp.genA.tpb) == 0); // SeedA partitions graph exactly
  assert((NEDGES / (tp.genA.blocks * tp.genA.tpb)) % EDGE_BLOCK_SIZE == 0); // SeedA loops cover full sipblocks
  assert(tp.genA.tpb / NX <= FLUSHA); // check ROWS_LIMIT_LOSSES
  assert(tp.genB.tpb / NX <= FLUSHB); // check COLS_LIMIT_LOSSES

  cudaSetDevice(params->device);
  if (!params->cpuload)
    checkCudaErrors_N(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

  SolverCtx* ctx = new SolverCtx(tp, params->mutate_nonce);

  return ctx;
}

CALL_CONVENTION void destroy_solver_ctx(SolverCtx* ctx) {
  delete ctx;
}

CALL_CONVENTION void stop_solver(SolverCtx* ctx) {
  ctx->abort();
}

CALL_CONVENTION void fill_default_params(SolverParams* params) {
  trimparams tp;
  params->device = 0;
  params->ntrims = tp.ntrims;
  params->genablocks = min(tp.genA.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.genA.tpb);
  params->genatpb = tp.genA.tpb;
  params->genbtpb = tp.genB.tpb;
  params->trimtpb = tp.trim.tpb;
  params->tailtpb = tp.tail.tpb;
  params->recoverblocks = min(tp.recover.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.recover.tpb);
  params->recovertpb = tp.recover.tpb;
  params->cpuload = false;
}

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  // set defaults
  SolverParams params;
  fill_default_params(&params);

  memset(header, 0, sizeof(header));
  while ((c = getopt(argc, argv, "scb:d:h:k:m:n:r:U:u:v:w:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedAblocks] [-u seedAthreads] [-v seedBthreads] [-w Trimthreads] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", EDGEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -u %d -v %d -w %d -y %d -Z %d -z %d\n", EDGEBITS, device, tp.ntrims, nonce, range, tp.genA.blocks, tp.genA.tpb, tp.genB.tpb, tp.trim.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'c':
        params.cpuload = false;
        break;
      case 'd':
        device = params.device = atoi(optarg);
        break;
      case 'h':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        params.ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'U':
        params.genablocks = atoi(optarg);
        break;
      case 'u':
        params.genatpb = atoi(optarg);
        break;
      case 'v':
        params.genbtpb = atoi(optarg);
        break;
      case 'w':
        params.trimtpb = atoi(optarg);
        break;
      case 'y':
        params.tailtpb = atoi(optarg);
        break;
      case 'Z':
        params.recoverblocks = atoi(optarg);
        break;
      case 'z':
        params.recovertpb = atoi(optarg);
        break;
    }
  }

  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  assert(device < nDevices);
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  u64 dbytes = prop.totalGlobalMem;
  int dunit;
  for (dunit=0; dbytes >= 102040; dbytes>>=10,dunit++) ;
  print_log("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, 0/*[tari-c29 patch] memoryClockRate removed in CUDA 13*/);
  // cudaSetDevice(device);

  print_log("Looking for %d-cycle on cuckaroo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, params.ntrims, NX);

  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 102400; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
