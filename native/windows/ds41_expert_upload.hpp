#pragma once

#include "ds41_expert_read.hpp"
#include "ggml-cpp.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// Three pinned slots allow two disk reads alongside an outstanding CUDA copy.
class ds41_expert_upload {
    static constexpr size_t chunk = 16ULL << 20;
    static constexpr size_t alignment = 4096;
    static constexpr size_t stride = chunk + 2*alignment;
    static constexpr int slots = 3;

    struct slot {
        ds41_expert_read read;
        ggml_backend_event_ptr copied;
        bool copying = false;
        ggml_tensor *dst;
        const uint8_t *src;
        const void *staged;
        size_t offset, bytes;
    };

    ggml_backend_buffer_ptr buffer;
    slot ring[slots];
    uint8_t *base = nullptr;
    int oldest = 0, next = 0, queued = 0, depth = 2;

    void submit_oldest(ggml_backend_t backend) {
        auto &s = ring[oldest];
        const void *data = s.staged && s.read.finish() ? s.staged : s.src;
        ggml_backend_tensor_set_async(backend, s.dst, data, s.offset, s.bytes);
        ggml_backend_event_record(s.copied.get(), backend);
        s.copying = true;
        oldest = (oldest + 1) % slots;
        --queued;
    }

public:
    explicit ds41_expert_upload(ggml_backend_dev_t device) {
        const char *setting = std::getenv("DS4_N1X_EXPERT_READ_DEPTH");
        if (setting && std::strcmp(setting, "1") == 0) depth = 1;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        if (!props.caps.async || !props.caps.events || !props.caps.host_buffer) return;
        const auto type = ggml_backend_dev_host_buffer_type(device);
        if (!type) return;
        for (auto &s : ring) {
            s.copied.reset(ggml_backend_event_new(device));
            if (!s.copied) return;
        }
        buffer.reset(ggml_backend_buft_alloc_buffer(type, slots*stride + alignment));
        if (!buffer || ggml_backend_buffer_get_type(buffer.get()) != type) {
            buffer.reset();
            return;
        }
        const auto ptr = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer.get()));
        base = reinterpret_cast<uint8_t *>((ptr + alignment - 1) & ~uintptr_t(alignment - 1));
    }

    ~ds41_expert_upload() {
        for (auto &s : ring) {
            if (s.copying) ggml_backend_event_synchronize(s.copied.get());
        }
    }

    // Enqueue all remaining GPU copies before a consumer can use this batch.
    void flush(ggml_backend_t backend) {
        while (queued) submit_oldest(backend);
    }

    void send(const llama_model &model, ggml_backend_t backend, ggml_tensor *dst,
              const uint8_t *src, size_t offset, size_t bytes) {
        while (base && bytes >= 65536) {
            auto &s = ring[next];
            if (s.copying) {
                ggml_backend_event_synchronize(s.copied.get());
                s.copying = false;
            }
            s.src = src;
            s.dst = dst;
            s.offset = offset;
            s.bytes = std::min(bytes, chunk);
            s.staged = model.read_mapped_block(src, s.bytes, base + next*stride, stride, &s.read);
            next = (next + 1) % slots;
            ++queued;
            if (queued == depth) submit_oldest(backend);
            src += s.bytes;
            offset += s.bytes;
            bytes -= s.bytes;
        }
        if (bytes) {
            flush(backend);
            ggml_backend_tensor_set_async(backend, dst, src, offset, bytes);
        }
    }
};
