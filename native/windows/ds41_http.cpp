// Native single-slot HTTP host for the pinned DeepSeek 4.1 GGUF backend.
#include "working_set.hpp"
#include <ggml-backend.h>
#include <llama.h>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

int llama_server(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc != 5 || (std::string(argv[2]) != "cuda" && std::string(argv[2]) != "cpu")) {
        fprintf(stderr, "usage: ds4-gguf-server MODEL cuda|cpu CONTEXT PORT\n");
        return 2;
    }
    n1x_working_set working_set;
    if (!working_set.apply()) return 1;
    // Tool calls belong to the HTTP client even when the operator also uses
    // a separate llama-server with its own tools or MCP configuration.
    for (const char *setting : {"LLAMA_ARG_TOOLS", "LLAMA_ARG_MCP_SERVERS_CONFIG", "LLAMA_ARG_MCP_SERVERS_JSON"})
        _putenv_s(setting, "");
    const bool cuda = std::string(argv[2]) == "cuda";
    // Wider CUDA batches let more prompt tokens share each expert-weight upload.
    std::vector<std::string> options = {
        argv[0], "-m", argv[1], "--host", "127.0.0.1", "--port", argv[4],
        "--alias", "DeepSeek-V4.1-Flash-1017-unsloth-UD-IQ2_XXS",
        "-c", argv[3], "-np", "1", "-t", "12", "-b", cuda ? "2048" : "1024", "-ub", cuda ? "2048" : "512",
        "--fit", "off", "--no-host", "--no-repack", "--load-mode", "mmap", "--lazy-mode", "on",
        "--cache-prompt", "--no-cache-idle-slots", "--ctx-checkpoints", "4", "--checkpoint-min-step", "512",
        "--jinja", "--reasoning", "off", "--no-warmup", "--offline", "--no-ui",
        "--cors-origins", "localhost", "--timeout", "3600",
    };
    if (!GetEnvironmentVariableA("LLAMA_ARG_CACHE_RAM", nullptr, 0)) {
        options.insert(options.end(), {"--cache-ram", "0"});
    }
    if (cuda) {
        char io[2]{};
        const DWORD length = GetEnvironmentVariableA("DS4_N1X_DIRECT_EXPERT_IO", io, sizeof(io));
        const bool direct = length != 1 || io[0] != '0';
        if (length == 0) _putenv_s("DS4_N1X_DIRECT_EXPERT_IO", "1");
        if (!GetEnvironmentVariableA("DS4_N1X_PREFETCH_EXPERTS", nullptr, 0))
            _putenv_s("DS4_N1X_PREFETCH_EXPERTS", direct ? "0" : "1");
        llama_backend_init();
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!dev) { fprintf(stderr, "CUDA device unavailable\n"); return 1; }
        size_t avail, total;
        ggml_backend_dev_memory(dev, &avail, &total);
        constexpr size_t GiB = size_t(1) << 30;
        // Leave room for the dense weights and CUDA workspaces before sizing the expert cache.
        if (avail <= 10 * GiB) { fprintf(stderr, "Insufficient free CUDA memory\n"); return 1; }
        const char * placement = GetEnvironmentVariableA("LLAMA_ARG_N_CPU_MOE", nullptr, 0)
            ? "per_layer_token_embd\\.weight=CPU"
            : "ffn_.*_exps\\.weight=CPU,per_layer_token_embd\\.weight=CPU";
        options.insert(options.end(), {"-ngl", "all", "-ot", placement});
        if (!GetEnvironmentVariableA("LLAMA_ARG_MOE_CACHE_MIB", nullptr, 0)) {
            auto mib = (avail - 8 * GiB) / (2 << 20);
            options.insert(options.end(), {"--moe-cache-mib", std::to_string(mib)});
        }
    } else {
        options.insert(options.end(), {"--device", "none", "-ngl", "0"});
    }
    std::vector<char *> args;
    for (auto &option : options) args.push_back(option.data());
    args.push_back(nullptr);
    // Windows may charge driver allocations outside the process working set.
    std::thread([] {
        for (;;) {
            MEMORYSTATUSEX mem{sizeof(mem)};
            if (GlobalMemoryStatusEx(&mem) && (mem.ullAvailPhys < (2ULL << 30) || mem.ullAvailPageFile < (2ULL << 30))) {
                fprintf(stderr, "ds41: stopping resident model at the 2 GiB Windows memory floor\n");
                fflush(stderr);
                TerminateProcess(GetCurrentProcess(), 2);
            }
            Sleep(100);
        }
    }).detach();
    return llama_server(static_cast<int>(options.size()), args.data());
}
