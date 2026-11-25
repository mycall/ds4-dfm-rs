#pragma once
#include "../file.h"
#include <limits.h>
#include <time.h>
#include <process.h>

typedef intptr_t ssize_t;
#define STDERR_FILENO 2
#define STDOUT_FILENO 1
#define PATH_MAX 32768
#define R_OK 4
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#define SSIZE_MAX INTPTR_MAX
#define off_t ds4_file_offset
#define stat _stat64
#define fstat _fstat64
#define mkstemp ds4_mkstemp
#define ftruncate _chsize_s
#define lseek _lseeki64

#define _SC_PAGESIZE 1
#define _SC_PHYS_PAGES 2
#define _SC_NPROCESSORS_ONLN 3
static inline int64_t sysconf(int key) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (key == _SC_PAGESIZE) return si.dwPageSize;
    if (key == _SC_NPROCESSORS_ONLN) return GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (key == _SC_PHYS_PAGES) {
        MEMORYSTATUSEX m = {sizeof(m)};
        return GlobalMemoryStatusEx(&m) ? (int64_t)(m.ullTotalPhys / si.dwPageSize) : -1;
    }
    errno = EINVAL;
    return -1;
}

static inline ssize_t pread(int fd, void *dst, size_t len, int64_t off) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || off < 0) { errno = EINVAL; return -1; }
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)off;
    ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);
    DWORD count;
    if (ReadFile(h, dst, (DWORD)(len > MAXDWORD ? MAXDWORD : len), &count, &ov)) return count;
    if (GetLastError() == ERROR_HANDLE_EOF) return 0;
    errno = EIO;
    return -1;
}

#define CLOCK_MONOTONIC 1
#define CLOCK_THREAD_CPUTIME_ID 2
static inline int clock_gettime(int which, struct timespec *ts) {
    if (which == CLOCK_MONOTONIC) {
        LARGE_INTEGER t, hz;
        if (!QueryPerformanceCounter(&t) || !QueryPerformanceFrequency(&hz)) return -1;
        ts->tv_sec = t.QuadPart / hz.QuadPart;
        ts->tv_nsec = (long)((t.QuadPart % hz.QuadPart) * 1000000000LL / hz.QuadPart);
        return 0;
    }
    FILETIME created, exited, kernel, user;
    if (which != CLOCK_THREAD_CPUTIME_ID ||
        !GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return -1;
    uint64_t ticks = (((uint64_t)kernel.dwHighDateTime << 32) | kernel.dwLowDateTime) +
                     (((uint64_t)user.dwHighDateTime << 32) | user.dwLowDateTime);
    ts->tv_sec = ticks / 10000000;
    ts->tv_nsec = (long)(ticks % 10000000) * 100;
    return 0;
}

static inline int nanosleep(const struct timespec *ts, struct timespec *remaining) {
    if (ts->tv_sec < 0 || ts->tv_nsec < 0 || ts->tv_nsec >= 1000000000) { errno = EINVAL; return -1; }
    uint64_t ms = (uint64_t)ts->tv_sec * 1000 + (ts->tv_nsec + 999999) / 1000000;
    while (ms >= MAXDWORD) { Sleep(MAXDWORD - 1); ms -= MAXDWORD - 1; }
    Sleep((DWORD)ms);
    return 0;
}

#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#define MADV_WILLNEED 1
static inline int madvise(void *addr, size_t len, int advice) {
    if (advice != MADV_WILLNEED) { errno = EINVAL; return -1; }
    WIN32_MEMORY_RANGE_ENTRY range = {addr, len};
    return PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) ? 0 : -1;
}

static inline ssize_t pwrite(int fd, const void *src, size_t len, int64_t off) {
    OVERLAPPED ov = {0};
    if (off < 0) { errno = EINVAL; return -1; }
    ov.Offset = (DWORD)off;
    ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);
    DWORD count;
    if (WriteFile((HANDLE)_get_osfhandle(fd), src, (DWORD)(len > MAXDWORD ? MAXDWORD : len),
                  &count, &ov)) return count;
    errno = EIO;
    return -1;
}
