// Test-only logits oracle using the pinned llama.cpp C API.
#include <llama.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <io.h>

int main(int argc, char **argv) {
    if (argc < 4 || (std::strcmp(argv[2], "f32") && std::strcmp(argv[2], "f16")))
        return 2;
    _setmode(_fileno(stdout), _O_BINARY);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu[] = {nullptr};
    mp.devices = cpu;
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.n_batch = cp.n_ubatch = 32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = std::strcmp(argv[2], "f32") == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 1;
    auto *batch = llama_batch_ext_init(ctx);
    if (!batch)
        return 1;
    auto n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 3; i < argc; ++i) {
        char *end = nullptr;
        const long id = std::strtol(argv[i], &end, 10);
        if (*end || id < 0 || id >= n_vocab)
            return 2;
        llama_batch_ext_clear(batch);
        int idx = llama_batch_ext_add_token(batch, 0, (int)id);
        const llama_pos pos = i - 3;
        if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
            !llama_batch_ext_set_output_logits(batch, idx, true) ||
            llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
            return 1;
        const float *row = llama_get_logits_ith(ctx, -1);
        if (!row || fwrite(row, sizeof(float), n_vocab, stdout) != (size_t)n_vocab)
            return 1;
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
