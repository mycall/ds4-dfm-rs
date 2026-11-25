/* Compare sparse reads through the engine's contiguous mapping with the
 * backing files. Supply GGUF shard paths in model order. */
#include "../native/bridge/ds4_shard_map.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    size_t count = (size_t)argc - 1;
    struct shard { int fd; size_t bytes, span; } *shards = calloc(count, sizeof(*shards));
    if (!shards) return 1;
    size_t page = (size_t)sysconf(_SC_PAGESIZE), total = 0;
    for (size_t i = 0; i < count; ++i) {
        struct stat st;
        shards[i].fd = open(argv[i + 1], O_RDONLY);
        if (shards[i].fd < 0 || fstat(shards[i].fd, &st) || st.st_size < 32) return 1;
        shards[i].bytes = (size_t)st.st_size;
        shards[i].span = (shards[i].bytes + page - 1) / page * page;
        if (shards[i].span > SIZE_MAX - total) return 1;
        total += shards[i].span;
    }
    unsigned char *region = ds4_reserve_shards(total);
    if (region == MAP_FAILED) return 1;
    size_t base = 0;
    for (size_t i = 0; i < count; ++i) {
        struct shard s = shards[i];
        unsigned char *view = ds4_map_shard(region + base, s.bytes, s.span, total - base,
                                            MAP_PRIVATE, s.fd);
        if (view != region + base) {
#ifdef _WIN32
            fprintf(stderr, "Win32 error %lu, size %zu span %zu remaining %zu\n",
                    GetLastError(), s.bytes, s.span, total - base);
#endif
            fprintf(stderr, "cannot map shard %zu at offset %zu\n", i + 1, base);
            return 1;
        }
        unsigned char reference[32];
        size_t offsets[] = {0, s.bytes / 2 - 16, s.bytes - sizeof(reference)};
        for (size_t j = 0; j < sizeof(offsets) / sizeof(*offsets); ++j) {
            if (pread(s.fd, reference, sizeof(reference), offsets[j]) != sizeof(reference) ||
                memcmp(view + offsets[j], reference, sizeof(reference))) return 1;
        }
        base += s.span;
    }
    ds4_unmap_shards(region, total);
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION info;
    if (!VirtualQuery(region, &info, sizeof(info)) || info.State != MEM_FREE ||
        info.RegionSize < total) return 1;
#endif
    for (size_t i = 0; i < count; ++i) close(shards[i].fd);
    free(shards);
    printf("PASS %zu contiguous shard views, %zu bytes of address space\n", count, total);
    return 0;
}
