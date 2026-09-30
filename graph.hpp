// Derived from tromp/cuckoo. SPDX-License-Identifier: GPL-2.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bitmap.hpp"
#include "compress.hpp"
#include <new>

#ifndef GRAPH_UNION_SKIP
#define GRAPH_UNION_SKIP 0
#endif

typedef word_t proof[PROOFSIZE];

// cuck(ar)oo graph with given limit on number of edges (and on single partition nodes)
template <typename word_t>
class graph {
public:
  // terminates adjacency lists
  static const word_t NIL = ~(word_t)0;

  struct link { // element of adjacency list
    word_t next;
    word_t to;
  };

  word_t MAXEDGES;
  word_t MAXNODES;
  word_t nlinks; // aka halfedges, twice number of edges
  word_t *adjlist; // index into links array
  link *links;
#if GRAPH_UNION_SKIP
  word_t *ufparent;
  word_t *ufsize;
#endif
  bool sharedmem;
  compressor<word_t> *compressu;
  compressor<word_t> *compressv;
  bitmap<u32> visited;
  u32 MAXSOLS;
  proof *sols;
  u32 nsols;
  // [tari-c29] DFS-trigger accounting for walk-cost profiling (TARI_C29_WALK_TIMING)
  u64 dfs_triggers = 0;

  graph(word_t maxedges, word_t maxnodes, u32 maxsols) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new word_t[2*MAXNODES]; // index into links array
    links   = new link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = compressv = 0;
    sharedmem = false;
    sols    = new proof[MAXSOLS+1]; // extra one for current path
    visited.clear();
  }

  ~graph() {
    if (!sharedmem) {
      delete[] adjlist;
      delete[] links;
    }
#if GRAPH_UNION_SKIP
    delete[] ufparent;
    delete[] ufsize;
#endif
    delete[] sols;
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, u32 compressbits) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new word_t[2*MAXNODES]; // index into links array
    links   = new link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = new compressor<word_t>(EDGEBITS, compressbits);
    compressv = new compressor<word_t>(EDGEBITS, compressbits);
    sharedmem = false;
    sols    = new  proof[MAXSOLS];
    visited.clear();
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, char *bytes) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new (bytes) word_t[2*MAXNODES]; // index into links array
    links   = new (bytes += (sizeof(word_t)*(size_t)(2*MAXNODES))) link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = compressv = 0;
    sharedmem = true;
    sols    = new  proof[MAXSOLS];
    visited.clear();
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, u32 compressbits, char *bytes) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new (bytes) word_t[2*MAXNODES]; // index into links array
    links   = new (bytes += (sizeof(word_t)*(size_t)(2*MAXNODES))) link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = new compressor<word_t>(EDGEBITS, compressbits, bytes += (sizeof(link)*(size_t)(2*MAXEDGES)));
    compressv = new compressor<word_t>(EDGEBITS, compressbits, bytes + compressu->bytes());
    sharedmem = true;
    sols    = new  proof[MAXSOLS];
    visited.clear();
  }

  // total size of new-operated data, excludes sols and visited bitmap of MAXEDGES bits
  uint64_t bytes() {
    return (sizeof(word_t)*(size_t)(2*MAXNODES)) + (sizeof(link)*(size_t)(2*MAXEDGES)) + (compressu ? 2 * compressu->bytes() : 0);
  }

  void reset(bool skip_compress = false) {
    memset(adjlist, (char)NIL, (sizeof(word_t)*(size_t)(2*MAXNODES)));
#if GRAPH_UNION_SKIP
    memset(ufparent, (char)NIL, (sizeof(word_t)*(size_t)(2*MAXNODES)));
#endif
    if (compressu && !skip_compress) { // [tari-c29] fast-walk path skips the two 8MB table resets
      compressu->reset();
      compressv->reset();
    }
    resetcounts();
  }

  void resetcounts() {
    nlinks = nsols = 0;
    dfs_triggers = 0; // [tari-c29] profiling
    if (compressu) compressu->probes = 0;
    if (compressv) compressv->probes = 0;
    // visited has entries set only during cycles() call
  }

  static int nonce_cmp(const void *a, const void *b) {
    return *(word_t *)a - *(word_t *)b;
  }

#if GRAPH_UNION_SKIP
  word_t uf_find(word_t x) {
    word_t r = x;
    while (ufparent[r] != r)
      r = ufparent[r];
    while (ufparent[x] != x) {
      word_t p = ufparent[x];
      ufparent[x] = r;
      x = p;
    }
    return r;
  }

  void uf_touch(word_t x) {
    if (ufparent[x] == NIL) {
      ufparent[x] = x;
      ufsize[x] = 1;
    }
  }

  void uf_join(word_t a, word_t b) {
    uf_touch(a);
    uf_touch(b);
    word_t ra = uf_find(a);
    word_t rb = uf_find(b);
    if (ra == rb)
      return;
    if (ufsize[ra] < ufsize[rb]) {
      word_t t = ra;
      ra = rb;
      rb = t;
    }
    ufparent[rb] = ra;
    ufsize[ra] += ufsize[rb];
  }
#endif

  void cycles_with_link(u32 len, word_t u, word_t dest) {
    // printf("cycles_with_link(%d, %x, %x)\n", len, u, dest);
    if (visited.test(u))
      return;
    if (u == dest) {
#if !SQUASH_OUTPUT
      print_log("  %d-cycle found\n", len);
#endif
      if (len == PROOFSIZE && nsols < MAXSOLS) {
        qsort(sols[nsols++], PROOFSIZE, sizeof(word_t), nonce_cmp);
        memcpy(sols[nsols], sols[nsols-1], sizeof(sols[0]));
      }
      return;
    }
    if (len == PROOFSIZE)
      return;
    word_t au1 = adjlist[u];
    if (au1 != NIL) {
      visited.set(u);
      for (; au1 != NIL; au1 = links[au1].next) {
        sols[nsols][len] = au1/2;
        cycles_with_link(len+1, links[au1 ^ 1].to, dest);
      }
      visited.reset(u);
    }
  }

  void add_edge(word_t u, word_t v) {
    assert(u < MAXNODES);
    assert(v < MAXNODES);
    v += MAXNODES; // distinguish partitions
#if GRAPH_UNION_SKIP
    bool maybeCycle = ufparent[u] != NIL && ufparent[v] != NIL && uf_find(u) == uf_find(v);
    if (maybeCycle) {
      ++dfs_triggers; // [tari-c29] profiling
#else
    if (adjlist[u] != NIL && adjlist[v] != NIL) { // possibly part of a cycle
      ++dfs_triggers; // [tari-c29] profiling
#endif
      sols[nsols][0] = nlinks/2;
      assert(!visited.test(u));
      cycles_with_link(1, u, v);
    }
    word_t ulink = nlinks++;
    word_t vlink = nlinks++; // the two halfedges of an edge differ only in last bit
    assert(vlink != NIL);    // avoid confusing links with NIL; guaranteed if bits in word_t > EDGEBITS + 1
    links[ulink].next = adjlist[u];
    links[vlink].next = adjlist[v];
    links[adjlist[u] = ulink].to = u;
    links[adjlist[v] = vlink].to = v;
#if GRAPH_UNION_SKIP
    uf_join(u, v);
#endif
  }

  void add_compress_edge(word_t u, word_t v) {
    add_edge(compressu->compress(u), compressv->compress(v));
  }
};
