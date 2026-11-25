// N1X adapter for llama.cpp's DeepSeek 4.1 GGUF graph. The Rust host owns
// prompt rendering, tokenization, sampling and output.
#define DS4_GGUF_BUILD
#include "../bridge/ds4_bridge.h"
#include <llama.h>
#include <ggml-backend.h>
#include <stdio.h>
#include <exception>
#include <memory>
#include "working_set.hpp"

struct ds4_bridge_gguf {
    struct llama_model *model;
    struct llama_context *ctx;
    struct llama_batch_ext *batch;
    llama_pos next_pos;
    n1x_working_set working_set;
};

void ds4_bridge_gguf_close(ds4_bridge_gguf *s) {
    if (!s) return;
    if (s->batch) llama_batch_ext_free(s->batch);
    if (s->ctx) llama_free(s->ctx);
    if (s->model) llama_model_free(s->model);
    delete s;
}

ds4_bridge_gguf *ds4_bridge_gguf_open(const char *path, int gpu, int ctx,
                                      int threads, int vocab) try {
    llama_log_set([](ggml_log_level level, const char *msg, void *) {
        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR)
            fputs(msg, stderr);
    }, nullptr);
    llama_backend_init();
    struct llama_model_params p = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu || (gpu && !llama_supports_gpu_offload())) {
        fprintf(stderr, "ds41: requested native backend is unavailable\n");
        return nullptr;
    }
    ggml_backend_dev_t cpu_only[] = {nullptr};
    struct llama_model_tensor_buft_override placement[] = {
        {"ffn_.*_exps\\.weight", ggml_backend_dev_buffer_type(cpu)},
        {"per_layer_token_embd\\.weight", ggml_backend_dev_buffer_type(cpu)},
        {nullptr, nullptr},
    };
    p.load_mode = LLAMA_LOAD_MODE_MMAP;
    p.lazy_mode = LLAMA_LAZY_MODE_ON;
    p.use_extra_bufts = false;
    p.no_host = true;
    p.n_gpu_layers = gpu ? -1 : 0;
    if (gpu) p.tensor_buft_overrides = placement;
    else p.devices = cpu_only;
    std::unique_ptr<ds4_bridge_gguf, decltype(&ds4_bridge_gguf_close)> s(
        new ds4_bridge_gguf{}, ds4_bridge_gguf_close);
    // Discard cold mapped expert pages before a large prefill displaces Windows.
    // Dedicated GPU memory is outside the pageable process working set.
    constexpr size_t GiB = size_t(1) << 30;
    if (!s->working_set.apply()) return nullptr;
    s->model = llama_model_load_from_file(path, p);
    if (!s->model || llama_vocab_n_tokens(llama_model_get_vocab(s->model)) != vocab)
        return nullptr;
    struct llama_context_params c = llama_context_default_params();
    if (gpu) {
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!device) return nullptr;
        size_t free_gpu, total_gpu;
        ggml_backend_dev_memory(device, &free_gpu, &total_gpu);
        c.moe_cache_size = free_gpu / 2;
        fprintf(stderr, "ds41: CUDA expert cache %.2f GiB, Windows working set limit %.2f GiB\n",
                (double)c.moe_cache_size / GiB, (double)s->working_set.max_ws / GiB);
    }
    c.n_ctx = ctx;
    c.n_batch = c.n_ubatch = 32;
    if (threads > 0) c.n_threads = c.n_threads_batch = threads;
    s->ctx = llama_init_from_model(s->model, c);
    if (!s->ctx) return nullptr;
    s->batch = llama_batch_ext_init(s->ctx);
    return s->batch ? s.release() : nullptr;
} catch (const std::exception &e) {
    fprintf(stderr, "ds41: opening graph: %s\n", e.what());
    return nullptr;
}

// Logits remain valid until the next eval or close. Each call appends to
// sequence zero. Only the prompt frontier needs an output projection.
const float *ds4_bridge_gguf_eval(ds4_bridge_gguf *s, const int *ids, int count) try {
    while (count > 0) {
        int n = count < 32 ? count : 32;
        llama_batch_ext_clear(s->batch);
        for (int i = 0; i < n; ++i) {
            int idx = llama_batch_ext_add_token(s->batch, 0, ids[i]);
            const llama_pos pos = s->next_pos + i;
            if (idx < 0 || !llama_batch_ext_set_pos(s->batch, idx, &pos) ||
                !llama_batch_ext_set_output_logits(s->batch, idx, i == count - 1))
                return nullptr;
        }
        if (llama_process(s->ctx, LLAMA_PROCESS_TYPE_DECODE, s->batch) != 0) return nullptr;
        s->next_pos += n;
        ids += n;
        count -= n;
    }
    return llama_get_logits_ith(s->ctx, -1);
} catch (const std::exception &e) {
    fprintf(stderr, "ds41: evaluating graph: %s\n", e.what());
    return nullptr;
}
