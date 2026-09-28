// tari_c29_pool_miner.cu - LuckyPool-compatible Tari C29 pool miner.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This is intentionally small and explicit: it wraps the verified Tari C29
// proof code and the reference CUDA Cuckaroo29 trimmer with the LuckyPool
// JSON-RPC dialect captured from lolMiner 1.98.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <future>

#include "tari_miner_pipeline.h"
#include "tari_miner_reliability.h"
#include "tari_pool_protocol.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using socket_t = SOCKET;
static const socket_t INVALID_SOCK = INVALID_SOCKET;
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>
extern "C" int close(int);
extern "C" int gethostname(char *, size_t);
using socket_t = int;
static const socket_t INVALID_SOCK = -1;
#endif

// --- MSVC host-compiler shims for the reference solver's GNU-isms ---
#if defined(_MSC_VER)
#define __builtin_prefetch(...) ((void)0)
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

#define SQUASH_OUTPUT 1
#define main reference_mean_main_unused
#include "mean_c29.cu"
#undef main

#include "tari_c29.h"
#include "version.h"

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

static double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

static void close_socket(socket_t s) {
    if (s == INVALID_SOCK) return;
#if defined(_WIN32)
    closesocket(s);
#else
    close(s);
#endif
}

static void shutdown_socket(socket_t s) {
    if (s == INVALID_SOCK) return;
#if defined(_WIN32)
    shutdown(s, SD_BOTH);
#else
    shutdown(s, SHUT_RDWR);
#endif
}

static int socket_init() {
#if defined(_WIN32)
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    return 0;
#endif
}

static void socket_cleanup() {
#if defined(_WIN32)
    WSACleanup();
#endif
}

static std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if ((unsigned char)c < 0x20) {
            char buf[7];
            snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
            out += buf;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static bool split_host_port(const std::string &pool, std::string &host, std::string &port) {
    size_t p = pool.rfind(':');
    if (p == std::string::npos || p == 0 || p + 1 >= pool.size()) return false;
    host = pool.substr(0, p);
    port = pool.substr(p + 1);
    return true;
}

static socket_t connect_tcp(const std::string &host, const std::string &port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return INVALID_SOCK;
    socket_t sock = INVALID_SOCK;
    for (addrinfo *rp = res; rp; rp = rp->ai_next) {
        sock = (socket_t)socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == INVALID_SOCK) continue;
        if (connect(sock, rp->ai_addr, (int)rp->ai_addrlen) == 0) break;
        close_socket(sock);
        sock = INVALID_SOCK;
    }
    freeaddrinfo(res);
    return sock;
}

static bool send_all(socket_t sock, const std::string &s) {
    const char *p = s.data();
    size_t left = s.size();
    while (left) {
        int n = send(sock, p, (int)left, 0);
        if (n <= 0) return false;
        p += n;
        left -= (size_t)n;
    }
    return true;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool parse_hex_bytes(const std::string &hex, uint8_t *out, size_t out_len) {
    if (hex.size() != out_len * 2) return false;
    for (size_t i = 0; i < out_len; ++i) {
        int hi = hexval(hex[2 * i]);
        int lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static std::string hex_bytes(const uint8_t *p, size_t n) {
    static const char hexdig[] = "0123456789abcdef";
    std::string out;
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = hexdig[p[i] >> 4];
        out[2 * i + 1] = hexdig[p[i] & 15];
    }
    return out;
}

static bool json_get_string_from(const std::string &line, const char *key, std::string &out, size_t start = 0) {
    size_t p = 0;
    if (!tari_pool::json_find_value_from(line, key, p, start) ||
        p == line.size() || line[p] != '"') {
        return false;
    }
    p++;
    std::string val;
    bool esc = false;
    for (; p < line.size(); ++p) {
        char c = line[p];
        if (esc) {
            val.push_back(c);
            esc = false;
        } else if (c == '\\') {
            esc = true;
        } else if (c == '"') {
            out = val;
            return true;
        } else {
            val.push_back(c);
        }
    }
    return false;
}

static bool json_get_uint_from(const std::string &line, const char *key, uint64_t &out, size_t start = 0) {
    return tari_pool::json_uint_from(line, key, out, start);
}

static uint64_t nonce_prefix_base(const std::string &xn_hex, uint64_t *counter_mask) {
    size_t nbytes = xn_hex.size() / 2;
    if ((xn_hex.size() % 2) || nbytes > 8) {
        *counter_mask = ~0ULL;
        return 0;
    }
    uint64_t prefix = 0;
    for (size_t i = 0; i < nbytes; ++i) {
        int hi = hexval(xn_hex[2 * i]);
        int lo = hexval(xn_hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            *counter_mask = ~0ULL;
            return 0;
        }
        prefix = (prefix << 8) | (uint64_t)((hi << 4) | lo);
    }
    int free_bits = (int)(64 - 8 * nbytes);
    uint64_t base = free_bits == 64 ? 0 : (prefix << free_bits);
    *counter_mask = free_bits == 64 ? ~0ULL : ((1ULL << free_bits) - 1ULL);
    return base;
}

static std::string nonce_hex_be(uint64_t nonce) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(nonce >> (56 - 8 * i));
    return hex_bytes(b, sizeof(b));
}

struct Job {
    std::string job_id;
    std::string blob_hex;
    std::string target_hex;
    std::string xn_hex;
    uint8_t mining_hash[32]{};
    uint64_t target_diff = 1;
    uint64_t height = 0;
    uint64_t seq = 0;
};

static bool parse_job_line(
    const std::string &line,
    Job &current,
    Job &out,
    bool *invalid_target
) {
    *invalid_target = false;
    size_t start = 0;
    if (!tari_pool::json_find_value_from(line, "job", start) &&
        !tari_pool::json_find_value_from(line, "params", start)) {
        return false;
    }

    Job j = current;
    std::string blob, job_id, target, xn;
    if (!json_get_string_from(line, "blob", blob, start)) return false;
    if (!json_get_string_from(line, "job_id", job_id, start)) return false;
    if (!json_get_string_from(line, "target", target, start)) target = j.target_hex;
    if (json_get_string_from(line, "xn", xn, start)) j.xn_hex = xn;
    uint64_t height = 0;
    if (json_get_uint_from(line, "height", height, start)) j.height = height;

    if (!parse_hex_bytes(blob, j.mining_hash, 32)) return false;
    j.blob_hex = blob;
    j.job_id = job_id;
    j.target_hex = target;
    if (!tari_pool::target_hex_to_diff(target, j.target_diff)) {
        *invalid_target = true;
        return false;
    }
    j.seq = current.seq + 1;
    out = j;
    return true;
}

class PoolClient {
public:
    bool connect_login(const std::string &pool, const std::string &login, const std::string &pass) {
        std::string host, port;
        if (!split_host_port(pool, host, port)) {
            fprintf(stderr, "bad --pool, expected host:port\n");
            return false;
        }
        socket_t socket = connect_tcp(host, port);
        if (socket == INVALID_SOCK) return false;
        socket_.set(socket);
        running_.store(true);
        reader_ = std::thread([this]() { read_loop(); });

        std::string req =
            "{\"id\":1,\"jsonrpc\":\"2.0\",\"method\":\"login\",\"params\":{\"agent\":\"tari-miner/" TARI_MINER_VERSION "\","
            "\"login\":\"" + json_escape(login) + "\",\"pass\":\"" + json_escape(pass) + "\"}}\n";
        return send_line(req);
    }

    void stop() {
        running_.store(false);
        socket_.stop(
            [](socket_t socket) { shutdown_socket(socket); },
            [this]() {
                if (reader_.joinable()) reader_.join();
            },
            [](socket_t socket) { close_socket(socket); }
        );
    }

    ~PoolClient() {
        stop();
    }

    bool alive() const {
        return running_.load();
    }

    tari_miner::JobWaitOutcome wait_for_job(Job &job, int timeout_ms) {
        double end = now_sec() + timeout_ms / 1000.0;
        uint64_t last = 0;
        while (now_sec() < end) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (job_.seq != 0 && job_.seq != last) {
                    job = job_;
                    return tari_miner::JobWaitOutcome::Job;
                }
                last = job_.seq;
            }
            if (login_failed_.load())
                return tari_miner::JobWaitOutcome::LoginRejected;
            if (protocol_error_.load())
                return tari_miner::JobWaitOutcome::ProtocolError;
            if (!alive())
                return tari_miner::JobWaitOutcome::Disconnected;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (login_failed_.load())
            return tari_miner::JobWaitOutcome::LoginRejected;
        if (protocol_error_.load())
            return tari_miner::JobWaitOutcome::ProtocolError;
        if (!alive())
            return tari_miner::JobWaitOutcome::Disconnected;
        return tari_miner::JobWaitOutcome::Timeout;
    }

    Job current_job() {
        std::lock_guard<std::mutex> lk(mu_);
        return job_;
    }

    bool submit_share(const Job &job, uint64_t nonce, const uint32_t edges[TARI_C29_PROOFSIZE]) {
        std::string id;
        uint64_t request_id;
        {
            std::lock_guard<std::mutex> lk(mu_);
            id = login_id_;
            if (id.empty()) return false;
            request_id = responses_.begin_submit();
        }

        uint8_t packed[TARI_C29_PACKED_BYTES];
        uint8_t result[32];
        tari_c29_pack(edges, packed);
        blake2b(result, 32, packed, TARI_C29_PACKED_BYTES, nullptr, 0);

        std::string req = "{\"id\":" + std::to_string(request_id) +
            ",\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{\"id\":\"" +
            json_escape(id) + "\",\"job_id\":\"" + json_escape(job.job_id) + "\",\"nonce\":\"" +
            nonce_hex_be(nonce) + "\",\"pow\":[";
        for (int i = 0; i < TARI_C29_PROOFSIZE; ++i) {
            if (i) req.push_back(',');
            char buf[32];
            snprintf(buf, sizeof(buf), "%u", edges[i]);
            req += buf;
        }
        req += "],\"result\":\"" + hex_bytes(result, sizeof(result)) + "\"}}\n";
        if (send_line(req)) return true;
        {
            std::lock_guard<std::mutex> lk(mu_);
            responses_.cancel_submit(request_id);
        }
        return false;
    }

    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }
    bool protocol_error() const { return protocol_error_.load(); }

private:
    bool send_line(const std::string &line) {
        return socket_.with_socket([&](socket_t socket) {
            return send_all(socket, line);
        });
    }

    void read_loop() {
        tari_pool::LineBuffer lines;
        char tmp[4096];
        while (running_.load()) {
            socket_t socket = socket_.load();
            if (socket == INVALID_SOCK) break;
            int n = recv(socket, tmp, sizeof(tmp), 0);
            if (n <= 0) break;
            if (!lines.append(tmp, (size_t)n, [this](const std::string &line) {
                    if (running_.load()) handle_line(line);
                })) {
                fprintf(stderr, "pool sent a line larger than %zu bytes; disconnecting\n",
                        tari_pool::MAX_LINE_BYTES);
                protocol_error_.store(true);
                break;
            }
        }
        running_.store(false);
    }

    void handle_line(const std::string &line) {
        if (!tari_pool::json_root_object_is_valid(line)) {
            fprintf(stderr, "pool sent invalid JSON; disconnecting\n");
            protocol_error_.store(true);
            running_.store(false);
            shutdown_socket(socket_.load());
            return;
        }
        uint64_t response_id = 0;
        bool has_id = tari_pool::json_root_uint(line, "id", response_id);
        size_t error_position = 0;
        bool has_error_field =
            tari_pool::json_find_root_value(line, "error", error_position);
        bool has_error =
            has_error_field && !tari_pool::json_root_literal(line, "error", "null");
        size_t result_position = 0;
        bool has_result =
            tari_pool::json_find_root_value(line, "result", result_position);
        bool result_true = tari_pool::json_root_literal(line, "result", "true");
        bool result_false = tari_pool::json_root_literal(line, "result", "false");
        // Some deployments answer a submit with the same {"status":"OK"} object
        // they use for login rather than a bare true. Only a response whose id
        // matches an outstanding submit is counted, so the login reply itself
        // cannot be mistaken for an accepted share.
        bool result_status_ok = tari_pool::json_result_status_ok(line);
        tari_miner::PoolResponseKind response;
        {
            std::lock_guard<std::mutex> lk(mu_);
            bool login_pending = login_id_.empty() && job_.seq == 0;
            response = responses_.classify(
                has_id, response_id, has_error, has_result,
                result_true, result_false,
                login_pending, result_status_ok
            );
        }

        if (response == tari_miner::PoolResponseKind::LoginError) {
            std::string message;
            if (!json_get_string_from(line, "message", message) &&
                !json_get_string_from(line, "error", message)) {
                message = "pool rejected the login";
            }
            std::string safe = tari_pool::sanitize_for_terminal(message);
            fprintf(stderr, "pool login rejected: %s\n", safe.c_str());
            login_failed_.store(true);
            running_.store(false);
            shutdown_socket(socket_.load());
            return;
        }
        if (response == tari_miner::PoolResponseKind::ShareAccepted) {
            accepted_.fetch_add(1);
            printf("share accepted (%llu total)\n", (unsigned long long)accepted_.load());
            return;
        }
        if (response == tari_miner::PoolResponseKind::ShareRejected) {
            rejected_.fetch_add(1);
            std::string safe = tari_pool::sanitize_for_terminal(line);
            printf("share rejected: %s\n", safe.c_str());
            return;
        }
        if (response == tari_miner::PoolResponseKind::OtherError) {
            std::string safe = tari_pool::sanitize_for_terminal(line);
            printf("pool error: %s\n", safe.c_str());
            return;
        }

        std::lock_guard<std::mutex> lk(mu_);
        size_t result_pos = 0;
        std::string id;
        if (tari_pool::json_find_root_value(line, "result", result_pos) &&
            json_get_string_from(line, "id", id, result_pos)) {
            login_id_ = id;
        }

        Job parsed;
        bool invalid_target = false;
        if (parse_job_line(line, job_, parsed, &invalid_target)) {
            job_ = parsed;
            std::string safe_job_id = tari_pool::sanitize_for_terminal(job_.job_id);
            std::string safe_xn = tari_pool::sanitize_for_terminal(job_.xn_hex);
            printf("new job height=%llu id=%s diff=%llu xn=%s\n",
                   (unsigned long long)job_.height, safe_job_id.c_str(),
                   (unsigned long long)job_.target_diff, safe_xn.c_str());
        } else if (invalid_target) {
            fprintf(stderr, "invalid pool target; disconnecting\n");
            protocol_error_.store(true);
            running_.store(false);
            shutdown_socket(socket_.load());
        }
    }

    tari_pool::SocketState<socket_t, INVALID_SOCK> socket_;
    std::thread reader_;
    mutable std::mutex mu_;
    Job job_;
    std::string login_id_;
    tari_miner::PoolResponseTracker responses_;
    std::atomic<bool> running_{false};
    std::atomic<bool> login_failed_{false};
    std::atomic<bool> protocol_error_{false};
    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
};

struct Options {
    std::string pool = "taric29-ca.luckypool.io:3111";
    std::string wallet;
    std::string worker;
    std::string pass = "x";
    // Character(s) joining wallet and worker in the stratum login. The default
    // is "wallet.worker"; other pool deployments may expect "wallet/worker".
    std::string login_separator = ".";
    int intensity = 100;
    int device = 0;
    int pipeline = 2;
    bool pipeline_set = false;
    int max_runtime_sec = 0;
    int ntrims = -1;
    int gena_blocks = -1;
    int gena_tpb = -1;
    int genb_tpb = -1;
    int trim_tpb = -1;
    int tail_tpb = -1;
    int recover_blocks = -1;
    int recover_tpb = -1;
    int trim_tpb_r1 = -1;
    int trim_tpb_r23 = -1;
};

static std::string default_worker() {
    char name[128] = {0};
#if defined(_WIN32)
    DWORD n = sizeof(name);
    if (GetComputerNameA(name, &n)) return name;
#else
    if (gethostname(name, sizeof(name) - 1) == 0) return name;
#endif
    return "worker";
}

static void usage() {
    printf("usage: tari_c29_pool_miner --wallet WALLET [options]\n"
           "options:\n"
           "  --pool host:port        default taric29-ca.luckypool.io:3111\n"
           "  --worker name           default hostname\n"
           "  --pass x                default x\n"
    "  --login-separator s     joins wallet and worker in the pool login, default \".\"\n"
           "                          (use \"/\" for pools expecting wallet/worker)\n"
           "  --intensity N           1-100 percent duty cycle, default 100 (no throttle)\n"
           "  --device N              default 0\n"
           "  --pipeline N            solver contexts to overlap GPU trim and CPU cycle search, default auto\n"
           "  --max-runtime-sec N     stop after N seconds (test helper)\n"
           "  --version               print version and exit\n"
           "tuning:\n"
           "  --ntrims N              even trim-round count, build default\n"
           "  --gena-blocks N         SeedA blocks, default 32768 bounded by graph size\n"
           "  --gena-tpb N            SeedA threads/block, default 128\n"
           "  --genb-tpb N            SeedB threads/block, default 128\n"
           "  --trim-tpb N            trim rounds threads/block, default 320\n"
           "  --tail-tpb N            tail threads/block, default 1024\n"
           "  --recover-blocks N      recovery blocks, default 1024 bounded by graph size\n"
           "  --recover-tpb N         recovery threads/block, default 1024\n"
           "  --trim-tpb-r1 N         round-1 trim threads/block, per-GPU auto default\n"
           "  --trim-tpb-r23 N        rounds 2-3 trim threads/block, per-GPU auto default\n");
}

static bool parse_args(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char *name) -> char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--pool")) { char *v = need(argv[i]); if (!v) return false; o.pool = v; }
        else if (!strcmp(argv[i], "--wallet")) { char *v = need(argv[i]); if (!v) return false; o.wallet = v; }
        else if (!strcmp(argv[i], "--worker")) { char *v = need(argv[i]); if (!v) return false; o.worker = v; }
        else if (!strcmp(argv[i], "--pass")) { char *v = need(argv[i]); if (!v) return false; o.pass = v; }
        else if (!strcmp(argv[i], "--login-separator")) { char *v = need(argv[i]); if (!v) return false; o.login_separator = v; }
        else if (!strcmp(argv[i], "--intensity")) { char *v = need(argv[i]); if (!v) return false; o.intensity = atoi(v); }
        else if (!strcmp(argv[i], "--device")) { char *v = need(argv[i]); if (!v) return false; o.device = atoi(v); }
        else if (!strcmp(argv[i], "--pipeline")) { char *v = need(argv[i]); if (!v) return false; o.pipeline = atoi(v); o.pipeline_set = true; }
        else if (!strcmp(argv[i], "--max-runtime-sec")) { char *v = need(argv[i]); if (!v) return false; o.max_runtime_sec = atoi(v); }
        else if (!strcmp(argv[i], "--ntrims")) { char *v = need(argv[i]); if (!v) return false; o.ntrims = atoi(v); }
        else if (!strcmp(argv[i], "--gena-blocks")) { char *v = need(argv[i]); if (!v) return false; o.gena_blocks = atoi(v); }
        else if (!strcmp(argv[i], "--gena-tpb")) { char *v = need(argv[i]); if (!v) return false; o.gena_tpb = atoi(v); }
        else if (!strcmp(argv[i], "--genb-tpb")) { char *v = need(argv[i]); if (!v) return false; o.genb_tpb = atoi(v); }
        else if (!strcmp(argv[i], "--trim-tpb")) { char *v = need(argv[i]); if (!v) return false; o.trim_tpb = atoi(v); }
        else if (!strcmp(argv[i], "--tail-tpb")) { char *v = need(argv[i]); if (!v) return false; o.tail_tpb = atoi(v); }
        else if (!strcmp(argv[i], "--recover-blocks")) { char *v = need(argv[i]); if (!v) return false; o.recover_blocks = atoi(v); }
        else if (!strcmp(argv[i], "--recover-tpb")) { char *v = need(argv[i]); if (!v) return false; o.recover_tpb = atoi(v); }
        else if (!strcmp(argv[i], "--trim-tpb-r1")) { char *v = need(argv[i]); if (!v) return false; o.trim_tpb_r1 = atoi(v); }
        else if (!strcmp(argv[i], "--trim-tpb-r23")) { char *v = need(argv[i]); if (!v) return false; o.trim_tpb_r23 = atoi(v); }
        else if (!strcmp(argv[i], "--version")) { printf("TARI.Miner C29 %s\n", TARI_MINER_VERSION); exit(0); }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); exit(0); }
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return false;
        }
    }
    if (o.worker.empty()) o.worker = default_worker();
    if (o.wallet.empty()) {
        fprintf(stderr, "--wallet is required\n");
        return false;
    }
    tari_miner::WalletValidationError wallet_error =
        tari_miner::validate_wallet(o.wallet);
    switch (wallet_error) {
        case tari_miner::WalletValidationError::None:
            break;
        case tari_miner::WalletValidationError::Empty:
            fprintf(stderr, "--wallet is required\n");
            break;
        case tari_miner::WalletValidationError::WhitespaceOrControl:
            fprintf(stderr, "--wallet must not contain whitespace or control characters\n");
            break;
        case tari_miner::WalletValidationError::TooLong:
            fprintf(stderr, "--wallet is %zu bytes; the supported maximum is %zu\n",
                    o.wallet.size(), tari_miner::MAX_WALLET_LENGTH);
            break;
        case tari_miner::WalletValidationError::TariAddressCharset:
            fprintf(stderr,
                    "warning: --wallet has a Tari address length but contains a "
                    "character Base58 never uses (0, O, I or l); check it for a typo\n");
            break;
    }
    if (tari_miner::wallet_validation_is_fatal(wallet_error))
        return false;
    if (o.intensity < 1) o.intensity = 1;
    if (o.intensity > 100) o.intensity = 100;
    if (o.pipeline_set)
        o.pipeline = tari_miner::clamp_pipeline_depth(o.pipeline);
    return true;
}

int main(int argc, char **argv) {
    // stdout is fully buffered when it is not a console, so redirected progress
    // lines - including the periodic speed report - would sit unflushed for
    // several kilobytes before reaching a log file, and would be lost entirely
    // if the process were killed. Win32 treats _IOLBF as full buffering, so use
    // unbuffered output there and line buffering on platforms that support it.
#ifdef _WIN32
    setvbuf(stdout, nullptr, _IONBF, 0);
#else
    setvbuf(stdout, nullptr, _IOLBF, 0);
#endif

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return 2;
    }
    if (socket_init() != 0) {
        fprintf(stderr, "socket init failed\n");
        return 1;
    }

    cudaError_t select_rc = cudaSetDevice(opt.device);
    if (select_rc != cudaSuccess) {
        fprintf(stderr, "no CUDA device %d: %s\n",
                opt.device, cudaGetErrorString(select_rc));
        socket_cleanup();
        return 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, opt.device) != cudaSuccess) {
        fprintf(stderr, "no CUDA device %d\n", opt.device);
        socket_cleanup();
        return 1;
    }
    printf("TARI.Miner C29 %s on %s (%.0f GB, sm_%d%d)\n",
           TARI_MINER_VERSION, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    printf("pool=%s worker=%s\n", opt.pool.c_str(), opt.worker.c_str());

    SolverParams params;
    fill_default_params(&params);
    params.device = opt.device;
    params.mutate_nonce = false;
#if !TARI_C29_REFERENCE_BUILD
    // Runtime per-arch tuning (see tari_miner_pipeline.h). Explicit CLI
    // overrides below still win over the table.
    const tari_miner::ArchTuning arch = tari_miner::arch_tuning(prop.major, prop.minor);
    if (arch.ntrims > 0) params.ntrims = (uint16_t)arch.ntrims;
    if (arch.trim_tpb_r1 > 0) params.trimtpb_r1 = (uint16_t)arch.trim_tpb_r1;
    if (arch.trim_tpb_r23 > 0) params.trimtpb_r23 = (uint16_t)arch.trim_tpb_r23;
#endif
    if (opt.ntrims > 0) params.ntrims = opt.ntrims & -2;
    if (opt.gena_blocks > 0) params.genablocks = opt.gena_blocks;
    if (opt.gena_tpb > 0) params.genatpb = opt.gena_tpb;
    if (opt.genb_tpb > 0) params.genbtpb = opt.genb_tpb;
    if (opt.trim_tpb > 0) params.trimtpb = opt.trim_tpb;
    if (opt.tail_tpb > 0) params.tailtpb = opt.tail_tpb;
    if (opt.recover_blocks > 0) params.recoverblocks = opt.recover_blocks;
    if (opt.recover_tpb > 0) params.recovertpb = opt.recover_tpb;
    if (opt.trim_tpb_r1 > 0) params.trimtpb_r1 = opt.trim_tpb_r1;
    if (opt.trim_tpb_r23 > 0) params.trimtpb_r23 = opt.trim_tpb_r23;
    std::vector<SolverCtx*> contexts;
    SolverCtx *ctx = create_solver_ctx(&params);
    if (!ctx || !ctx->trimmer.initsuccess) {
        fprintf(stderr, "failed to init solver (need ~6GB VRAM). reason: %s\n", LAST_ERROR_REASON);
        socket_cleanup();
        return 1;
    }
    contexts.push_back(ctx);
    size_t free_after_first = 0, total_after_first = 0;
    cudaError_t memory_rc = cudaMemGetInfo(&free_after_first, &total_after_first);
    if (memory_rc != cudaSuccess)
        cudaGetLastError();
    opt.pipeline = tari_miner::choose_pipeline_depth(
        opt.pipeline_set,
        opt.pipeline,
        driver_mode(prop),
        free_after_first,
        solver_context_bytes(ctx),
        memory_rc == cudaSuccess
    );
    for (int i = 1; i < opt.pipeline; ++i) {
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
    opt.pipeline = (int)contexts.size();
    printf("solver pipeline=%d context%s\n", opt.pipeline, opt.pipeline == 1 ? "" : "s");

    uint64_t graphs = 0, cycles = 0, submitted = 0, verify_failures = 0;
    int exit_code = 0;
    tari_miner::LoginFailurePolicy login_failures;
    tari_miner::PoolSilencePolicy pool_silence;
    tari_miner::ProtocolErrorPolicy protocol_errors;
    std::vector<tari_miner::SolverWatchdog> solver_watchdogs(contexts.size());
    double start = now_sec();
    double last_report = start;

    auto observe_trim = [&](int context, const SolverTrimResult &trim) {
        tari_miner::SolverWatchdog &watchdog = solver_watchdogs[(size_t)context];
        if (watchdog.observe(trim.nedges, trim.cuda_error == cudaSuccess))
            return true;
        if (trim.cuda_error != cudaSuccess)
            report_cuda_failure(opt.device, context, "trim", trim.cuda_error);
        else
            report_zero_yield_failure(opt.device, context);
        exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
        return false;
    };

    auto record_protocol_error = [&]() {
        bool fatal = protocol_errors.record_failure();
        if (fatal) {
            fprintf(stderr,
                    "pool sent invalid protocol data %u times; exiting "
                    "for operator review\n",
                    protocol_errors.consecutive_failures());
            exit_code = tari_miner::POOL_PROTOCOL_EXIT_CODE;
            return true;
        }
        fprintf(stderr, "pool protocol error (%u/%u); retrying in 5s\n",
                protocol_errors.consecutive_failures(),
                tari_miner::MAX_PROTOCOL_ERRORS);
        return false;
    };

    while (true) {
        double elapsed = now_sec() - start;
        if (opt.max_runtime_sec > 0 && elapsed >= opt.max_runtime_sec) break;

        std::string login = opt.wallet + opt.login_separator + opt.worker;
        printf("connecting to %s as %s\n", opt.pool.c_str(), login.c_str());

        PoolClient pool;
        if (!pool.connect_login(opt.pool, login, opt.pass)) {
            pool_silence.reset();
            protocol_errors.reset();
            fprintf(stderr, "pool connection/login send failed; retrying in 5s\n");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        Job job;
        tari_miner::JobWaitOutcome wait_outcome =
            pool.wait_for_job(job, 20000);
        if (wait_outcome != tari_miner::JobWaitOutcome::Job) {
            if (wait_outcome == tari_miner::JobWaitOutcome::LoginRejected) {
                pool.stop();
                pool_silence.reset();
                protocol_errors.reset();
                bool fatal = login_failures.record_failure();
                if (fatal) {
                    fprintf(stderr,
                            "pool login rejected %u times; check the wallet address, "
                            "worker name, password, and login separator\n",
                            login_failures.consecutive_failures());
                    exit_code = tari_miner::LOGIN_FAILURE_EXIT_CODE;
                    break;
                }
                fprintf(stderr,
                        "pool login rejected (%u/%u); retrying in 5s\n",
                        login_failures.consecutive_failures(),
                        tari_miner::MAX_LOGIN_FAILURES);
                std::this_thread::sleep_for(
                    std::chrono::seconds(tari_miner::LOGIN_RETRY_SECONDS));
                continue;
            }
            pool.stop();
            if (tari_miner::counts_as_pool_silence(wait_outcome)) {
                protocol_errors.reset();
                // The connection stayed open but sent no job and no error, so
                // there is nothing for the login policy to count. Back off, and
                // give up eventually rather than reconnecting forever in silence.
                bool silent_fatal = pool_silence.record_silence();
                if (silent_fatal) {
                    fprintf(stderr,
                            "no job received from %s after %u attempts; exiting for "
                            "supervisor restart\n",
                            opt.pool.c_str(), pool_silence.consecutive_silences());
                    exit_code = tari_miner::POOL_SILENT_EXIT_CODE;
                    break;
                }
                unsigned backoff = pool_silence.backoff_seconds();
                fprintf(stderr, "no job received (%u/%u); reconnecting in %us\n",
                        pool_silence.consecutive_silences(),
                        tari_miner::MAX_SILENT_CYCLES, backoff);
                std::this_thread::sleep_for(std::chrono::seconds(backoff));
                continue;
            }
            pool_silence.reset();
            if (wait_outcome == tari_miner::JobWaitOutcome::ProtocolError) {
                if (record_protocol_error())
                    break;
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
            protocol_errors.reset();
            fprintf(stderr,
                    "pool disconnected before the first valid job; retrying in 5s\n");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        login_failures.record_success();
        pool_silence.record_job();
        protocol_errors.record_valid_job(job.seq);

        uint64_t last_seq = 0;
        uint64_t base = 0, mask = ~0ULL, counter = 0;

        auto derive_keys = [&](uint64_t nonce, const Job &j) -> siphash_keys {
            tari_siphash_keys k;
            tari_c29_derive_keys(nonce, j.mining_hash, &k);
            siphash_keys sk;
            sk.k0 = k.k0;
            sk.k1 = k.k1;
            sk.k2 = k.k2;
            sk.k3 = k.k3;
            return sk;
        };

        auto inject_keys = [&](SolverCtx *c, uint64_t nonce, const Job &j) {
            c->trimmer.sipkeys = derive_keys(nonce, j);
            c->sols.clear();
        };

        auto consume_solutions = [&](SolverCtx *c, const Job &sol_job, uint64_t nonce) {
            int nsols = (int)(c->sols.size() / TARI_C29_PROOFSIZE);
            for (int s = 0; s < nsols; ++s) {
                uint32_t edges[TARI_C29_PROOFSIZE];
                for (int i = 0; i < TARI_C29_PROOFSIZE; ++i)
                    edges[i] = c->sols[s * TARI_C29_PROOFSIZE + i];
                int rc = tari_c29_verify(nonce, sol_job.mining_hash, edges);
                if (rc != TARI_C29_OK) {
                    verify_failures++;
                    continue;
                }
                cycles++;
                uint64_t diff = tari_c29_check(nonce, sol_job.mining_hash, edges);
                if (diff >= sol_job.target_diff) {
                    printf("share diff=%llu target=%llu nonce=%s\n",
                           (unsigned long long)diff, (unsigned long long)sol_job.target_diff,
                           nonce_hex_be(nonce).c_str());
                    if (pool.submit_share(sol_job, nonce, edges)) submitted++;
                }
            }
        };

        auto report_speed = [&]() {
            double t = now_sec();
            if (t - last_report >= 15.0) {
                double gsec = graphs / (t - start);
                printf("speed %.2f g/s | graphs=%llu cycles=%llu submitted=%llu accepted=%llu rejected=%llu\n",
                       gsec, (unsigned long long)graphs, (unsigned long long)cycles,
                       (unsigned long long)submitted, (unsigned long long)pool.accepted(),
                       (unsigned long long)pool.rejected());
                last_report = t;
            }
        };

        // Duty-cycle throttle. Sleeps in proportion to the time worked since the
        // previous call, so intensity 50 leaves the card idle roughly half the
        // time and 100 never sleeps at all. Pipelined callers first wait for
        // queued trims so the GPU is actually idle during this sleep.
        double last_resume = now_sec();
        auto throttle = [&]() {
            if (opt.intensity >= 100) return;
            double now = now_sec();
            double worked = now - last_resume;
            if (worked > 0.0) {
                double idle = worked * (100.0 - opt.intensity) / opt.intensity;
                double deadline = now + idle;
                while (pool.alive()) {
                    double remaining = deadline - now_sec();
                    if (remaining <= 0.0) break;
                    if (opt.max_runtime_sec > 0 && now_sec() - start >= opt.max_runtime_sec) break;
                    Job latest = pool.current_job();
                    if (latest.seq != 0 && latest.seq != last_seq) break;
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(std::min(remaining, 0.1)));
                }
            }
            last_resume = now_sec();
        };

        if (opt.pipeline <= 1) while (pool.alive()) {
            elapsed = now_sec() - start;
            if (opt.max_runtime_sec > 0 && elapsed >= opt.max_runtime_sec) break;

            job = pool.current_job();
            if (job.seq == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (job.seq != last_seq) {
                // Do not reset on the initial job: a pool that sends one valid
                // job and then malformed updates on every reconnect must still
                // reach the protocol-error limit. A later valid update proves
                // the connection has recovered and breaks that streak.
                protocol_errors.record_valid_job(job.seq);
                last_seq = job.seq;
                base = nonce_prefix_base(job.xn_hex, &mask);
                counter = ((uint64_t)(now_sec() * 1000000.0)) & mask;
            }

            uint64_t nonce = base | (counter++ & mask);
            inject_keys(ctx, nonce, job);

            SolverTrimResult trim = ctx->trim_copy_checked(opt.device);
            if (trim.cuda_error == cudaSuccess)
                graphs++;
            if (!observe_trim(0, trim))
                break;
            if (trim.nedges) {
                int cycle_rc = ctx->findcycles_copied_status(trim.nedges);
                if (cycle_rc != cudaSuccess) {
                    report_cuda_failure(
                        opt.device, 0, "cycle recovery", (cudaError_t)cycle_rc
                    );
                    exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
                    break;
                }
            }
            consume_solutions(ctx, job, nonce);
            report_speed();
            throttle();
        } else {
            struct PendingTrim {
                std::future<SolverTrimResult> future;
                Job job;
                uint64_t nonce = 0;
                bool active = false;
            };
            std::vector<PendingTrim> pending((size_t)opt.pipeline);
            uint64_t done = 0;

            auto launch_trim = [&](int slot) -> bool {
                double launch_elapsed = now_sec() - start;
                if (opt.max_runtime_sec > 0 && launch_elapsed >= opt.max_runtime_sec) return false;
                if (!pool.alive()) return false;
                Job launch_job = pool.current_job();
                if (launch_job.seq == 0) return false;
                if (launch_job.seq != last_seq) {
                    protocol_errors.record_valid_job(launch_job.seq);
                    last_seq = launch_job.seq;
                    base = nonce_prefix_base(launch_job.xn_hex, &mask);
                    counter = ((uint64_t)(now_sec() * 1000000.0)) & mask;
                }
                uint64_t nonce = base | (counter++ & mask);
                SolverCtx *slot_ctx = contexts[(size_t)slot];
                inject_keys(slot_ctx, nonce, launch_job);
                pending[(size_t)slot].job = launch_job;
                pending[(size_t)slot].nonce = nonce;
                pending[(size_t)slot].active = true;
                pending[(size_t)slot].future = std::async(std::launch::async, [slot_ctx, device = opt.device]() {
                    return slot_ctx->trim_copy_checked(device);
                });
                return true;
            };

            auto drain_pending = [&]() {
                for (int slot = 0; slot < opt.pipeline; ++slot) {
                    if (pending[(size_t)slot].active) {
                        SolverTrimResult trim =
                            pending[(size_t)slot].future.get();
                        pending[(size_t)slot].active = false;
                        if (trim.cuda_error == cudaSuccess)
                            graphs++;
                        if (!exit_code)
                            observe_trim(slot, trim);
                    }
                }
            };

            auto wait_pending = [&]() {
                if (opt.intensity >= 100) return;
                for (int slot = 0; slot < opt.pipeline; ++slot) {
                    if (pending[(size_t)slot].active)
                        pending[(size_t)slot].future.wait();
                }
            };

            for (int slot = 0; slot < opt.pipeline; ++slot) {
                if (!launch_trim(slot)) break;
            }

            while (pool.alive()) {
                elapsed = now_sec() - start;
                if (opt.max_runtime_sec > 0 && elapsed >= opt.max_runtime_sec) break;

                int slot = (int)(done % (uint64_t)opt.pipeline);
                if (!pending[(size_t)slot].active) {
                    if (!launch_trim(slot)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                }

                SolverCtx *slot_ctx = contexts[(size_t)slot];
                SolverTrimResult trim = pending[(size_t)slot].future.get();
                pending[(size_t)slot].active = false;
                if (trim.cuda_error == cudaSuccess)
                    graphs++;
                if (!observe_trim(slot, trim))
                    break;
                if (trim.nedges) {
                    int cycle_rc = slot_ctx->findcycles_copied_status(trim.nedges);
                    if (cycle_rc != cudaSuccess) {
                        report_cuda_failure(
                            opt.device, slot, "cycle recovery", (cudaError_t)cycle_rc
                        );
                        exit_code = tari_miner::SOLVER_FAILURE_EXIT_CODE;
                        break;
                    }
                }
                consume_solutions(slot_ctx, pending[(size_t)slot].job, pending[(size_t)slot].nonce);
                done++;
                launch_trim(slot);
                report_speed();
                wait_pending();
                throttle();
            }
            drain_pending();
        }
        if (exit_code)
            break;
        if (pool.protocol_error()) {
            pool.stop();
            // The reader can parse a valid update and malformed data in the
            // same receive batch before the mining loop observes that update.
            // Consult the synchronized final job state before counting the
            // protocol failure so recovery does not depend on thread timing.
            protocol_errors.record_valid_job(pool.current_job().seq);
            if (record_protocol_error())
                break;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        // A connection that ends without malformed data also breaks the
        // protocol-error streak.
        protocol_errors.reset();
    }

    double elapsed = now_sec() - start;
    printf("\n--- summary ---\n");
    printf("graphs=%llu elapsed=%.2fs speed=%.3f g/s cycles=%llu submitted=%llu verify_failures=%llu\n",
           (unsigned long long)graphs, elapsed, graphs / elapsed,
           (unsigned long long)cycles, (unsigned long long)submitted,
           (unsigned long long)verify_failures);
    for (SolverCtx *c : contexts)
        destroy_solver_ctx(c);
    socket_cleanup();
    if (exit_code) return exit_code;
    return verify_failures ? 3 : 0;
}
