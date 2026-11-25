/* Page-aligned contiguous GGUF shard views. No file data is copied. */
#pragma once
#include <sys/mman.h>
#include <unistd.h>

static inline void *ds4_reserve_shards(size_t bytes) {
#ifdef _WIN32
    void *p = VirtualAlloc2(NULL, NULL, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                            PAGE_NOACCESS, NULL, 0);
    if (!p) { errno = ENOMEM; return MAP_FAILED; }
    return p;
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    return mmap(NULL, bytes, PROT_NONE, flags, -1, 0);
#endif
}

/* span and remaining include page padding. MapViewOfFile3 permits page
 * alignment when replacing a placeholder, so the Rust/Linux shard bases agree. */
static inline void *ds4_map_shard(void *at, size_t bytes, size_t span, size_t remaining,
                                  int flags, int fd) {
#ifdef _WIN32
    (void)flags;
    if (span < remaining && !VirtualFree(at, span, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
        errno = EINVAL;
        return MAP_FAILED;
    }
    HANDLE section = CreateFileMappingW((HANDLE)_get_osfhandle(fd), NULL, PAGE_READONLY, 0, 0, NULL);
    if (!section) { errno = EIO; return MAP_FAILED; }
    /* Pass the exact file length. The view covers its rounded page span;
     * requesting bytes beyond EOF fails for a read-only section. */
    void *view = MapViewOfFile3(section, GetCurrentProcess(), at, 0, bytes,
                                MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, NULL, 0);
    CloseHandle(section);
    if (!view) { errno = EIO; return MAP_FAILED; }
    return view;
#else
    (void)span;
    (void)remaining;
    return mmap(at, bytes, PROT_READ, flags | MAP_FIXED, fd, 0);
#endif
}

static inline void ds4_unmap_shards(void *base, size_t bytes) {
#ifdef _WIN32
    char *at = (char *)base;
    char *end = at + bytes;
    while (at < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(at, &info, sizeof(info)) || info.AllocationBase != at) return;
        char *next = at + info.RegionSize;
        while (next < end && VirtualQuery(next, &info, sizeof(info)) && info.AllocationBase == at) {
            next += info.RegionSize;
        }
        if (!UnmapViewOfFile(at)) return;
        at = next;
    }
#else
    munmap(base, bytes);
#endif
}
