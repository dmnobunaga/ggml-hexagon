#pragma once

// Opt-in, read-only Linux/Android memory snapshots. No allocation or paging of
// model weights; values are KiB and -1 means not readable on this device.
// dmaBufGlobalKiB is SYSTEM-WIDE, not process-owned: do not add it to VmRSS.

#include <string>

#if defined(__linux__) || defined(__ANDROID__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <dirent.h>
#include <time.h>
#include <unistd.h>

namespace ggml_mem_snapshot_detail {

struct metrics {
    long long mem_available = -1;
    long long mem_free = -1;
    long long cached = -1;
    long long shmem = -1;
    long long sreclaimable = -1;
    long long sunreclaim = -1;
    long long swap_free = -1;
    long long vm_rss = -1;
    long long vm_hwm = -1;
    long long rss_anon = -1;
    long long rss_file = -1;
    long long rss_shmem = -1;
    long long vm_swap = -1;
    long long pss = -1;
    long long dmabuf_kib = -1;
    long long dmabuf_count = -1;
};

inline void load_proc_file(const char * path, metrics & m) {
    FILE * f = std::fopen(path, "r");
    if (!f) {
        return;
    }
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        char key[64];
        long long val = 0;
        if (std::sscanf(line, " %63[^:]: %lld", key, &val) != 2) {
            continue;
        }
        if      (!std::strcmp(key, "MemAvailable")) m.mem_available = val;
        else if (!std::strcmp(key, "MemFree"))      m.mem_free = val;
        else if (!std::strcmp(key, "Cached"))       m.cached = val;
        else if (!std::strcmp(key, "Shmem"))        m.shmem = val;
        else if (!std::strcmp(key, "SReclaimable")) m.sreclaimable = val;
        else if (!std::strcmp(key, "SUnreclaim"))   m.sunreclaim = val;
        else if (!std::strcmp(key, "SwapFree"))     m.swap_free = val;
        else if (!std::strcmp(key, "VmRSS"))        m.vm_rss = val;
        else if (!std::strcmp(key, "VmHWM"))        m.vm_hwm = val;
        else if (!std::strcmp(key, "RssAnon"))      m.rss_anon = val;
        else if (!std::strcmp(key, "RssFile"))      m.rss_file = val;
        else if (!std::strcmp(key, "RssShmem"))     m.rss_shmem = val;
        else if (!std::strcmp(key, "VmSwap"))       m.vm_swap = val;
        else if (!std::strcmp(key, "Pss"))          m.pss = val;
    }
    std::fclose(f);
}

inline void load_dmabuf_global(metrics & m) {
    DIR * dir = opendir("/sys/kernel/dmabuf/buffers");
    if (!dir) {
        return;
    }
    long long total_bytes = 0;
    long long count = 0;
    while (const dirent * ent = readdir(dir)) {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9') {
            continue;
        }
        char path[256];
        int len = std::snprintf(path, sizeof(path), "/sys/kernel/dmabuf/buffers/%s/size", ent->d_name);
        if (len <= 0 || (size_t) len >= sizeof(path)) {
            continue;
        }
        FILE * f = std::fopen(path, "r");
        if (!f) {
            continue;
        }
        long long size_bytes = 0;
        bool parsed = std::fscanf(f, "%lld", &size_bytes) == 1;
        std::fclose(f);
        if (parsed && size_bytes >= 0 && total_bytes <= LLONG_MAX - size_bytes) {
            total_bytes += size_bytes;
            ++count;
        }
    }
    closedir(dir);
    if (count > 0) {
        m.dmabuf_kib = total_bytes / 1024;
        m.dmabuf_count = count;
    }
}

} // namespace ggml_mem_snapshot_detail

inline std::string ggml_mem_snapshot(const std::string & stage) {
    const char * enabled = std::getenv("GGML_MEM_SNAPSHOT");
    if (!enabled || std::strcmp(enabled, "1") != 0) {
        return {};
    }

    using namespace ggml_mem_snapshot_detail;
    metrics m;
    load_proc_file("/proc/meminfo", m);
    load_proc_file("/proc/self/status", m);
    load_proc_file("/proc/self/smaps_rollup", m);
    load_dmabuf_global(m);

    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const long long t_ms = (long long) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;

    char line[1200];
    std::snprintf(line, sizeof(line),
        "[MEM] stage=%s t_ms=%lld pid=%lld "
        "MemAvailableKiB=%lld MemFreeKiB=%lld CachedKiB=%lld ShmemKiB=%lld "
        "SReclaimableKiB=%lld SUnreclaimKiB=%lld SwapFreeKiB=%lld "
        "VmRSSKiB=%lld VmHWMKiB=%lld RssAnonKiB=%lld RssFileKiB=%lld "
        "RssShmemKiB=%lld VmSwapKiB=%lld PssKiB=%lld "
        "dmaBufGlobalKiB=%lld dmaBufCount=%lld",
        stage.c_str(), t_ms, (long long) getpid(),
        m.mem_available, m.mem_free, m.cached, m.shmem,
        m.sreclaimable, m.sunreclaim, m.swap_free,
        m.vm_rss, m.vm_hwm, m.rss_anon, m.rss_file,
        m.rss_shmem, m.vm_swap, m.pss, m.dmabuf_kib, m.dmabuf_count);
    return line;
}

#else
inline std::string ggml_mem_snapshot(const std::string &) { return {}; }
#endif
