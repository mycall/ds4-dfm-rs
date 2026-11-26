#include "llama-mmap.h"
#include "ds41_expert_read.hpp"

#include <cstdlib>
#include <cstring>
#include <malloc.h>

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    _putenv_s("DS4_N1X_DIRECT_EXPERT_IO", "1");
    llama_file file(argv[1], "rb");
    llama_mmap mapping(&file, 0, false);
    constexpr size_t capacity = 32768;
    std::unique_ptr<void, decltype(&_aligned_free)> staging(_aligned_malloc(capacity, 4096), &_aligned_free);
    if (!staging || file.size() < capacity) return 1;

    const auto * mapped = static_cast<const uint8_t *>(mapping.addr());
    for (const auto [offset, bytes] : {std::pair<size_t, size_t>{0, 4096}, {31, 12543},
                                      {4093, 8198}, {8192, 17317}, {file.size() - 4432, 4432}}) {
        const void * data = mapping.read_block(offset, bytes, staging.get(), capacity);
        if (!data || std::memcmp(data, mapped + offset, bytes) != 0) {
            std::fprintf(stderr, "direct shard read mismatch at offset %zu, length %zu\n", offset, bytes);
            return 1;
        }
    }
    if (mapping.read_block(31, 8192, staging.get(), 512) != nullptr) return 1;

    decltype(staging) second(_aligned_malloc(capacity, 4096), &_aligned_free);
    if (!second) return 1;
    ds41_expert_read first_read, second_read;
    for (const size_t offset : {size_t(97), file.size() - 4913}) {
        const void * a = mapping.read_block(offset, 4913, staging.get(), capacity, &first_read);
        const void * b = mapping.read_block(12289, 9000, second.get(), capacity, &second_read);
        if (!a || !b || !second_read.finish() || !first_read.finish() ||
            std::memcmp(a, mapped + offset, 4913) || std::memcmp(b, mapped + 12289, 9000)) {
            std::fprintf(stderr, "overlapping shard reads failed at offset %zu\n", offset);
            return 1;
        }
    }
    // A sector-sized read may succeed at EOF with fewer bytes than requested.
    if (mapping.read_block(file.size() - 15, 16, staging.get(), capacity, &first_read) && first_read.finish()) return 1;

    _putenv_s("DS4_N1X_DIRECT_EXPERT_IO", "0");
    llama_mmap fallback(&file, 0, false);
    return fallback.read_block(4096, 4096, staging.get(), capacity) != nullptr;
}
