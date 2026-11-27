#define NOMINMAX
#include "draft.hpp"
#include "layers.hpp"
#include "chat.h"
#include "llama-model.h"
#include "speculative.h"
#include "working_set.hpp"
#include <chrono>
#include <cstdio>
#include <thread>
#include <iostream>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
enum class Feature { Window, Indices, Compressed, Hidden };
struct Tap {
    std::string name;
    Feature feature;
    int slot = 0;
};
static std::vector<Tap> target_taps(const llama_hparams &hp) {
    const int source = dfly_last_visit(hp, 20);
    std::vector<Tap> taps{{"dfly_comp_row-" + std::to_string(source), Feature::Compressed},
                          {"result_norm", Feature::Hidden}};
    const std::array<int, 3> layers{28, 32, 36};
    for (int slot = 0; slot < 3; ++slot) {
        const int logical = dfly_last_visit(hp, layers[slot]);
        require(hp.dsv41_kv_src[logical] == source && hp.dsv41_topk_src[logical] == logical,
                "unexpected DFly2 target KV or index source");
        taps.push_back({"kv-" + std::to_string(logical), Feature::Window, slot});
        taps.push_back({"idx_top_k-" + std::to_string(logical), Feature::Indices, slot});
    }
    return taps;
}
struct Capture {
    // Single sequence and fresh contiguous positions. Only the current batch can be rolled back.
    int start = 0, n = 0;
    std::array<std::vector<float>, 3> window;
    std::vector<float> global, hidden;
    std::array<std::vector<std::vector<int>>, 3> picks;
    void collect(Tensor *t, const Tap &tap) {
        const auto &name = tap.name;
        bool comp = tap.feature == Feature::Compressed;
        bool hid = tap.feature == Feature::Hidden;
        require(ggml_is_contiguous(t), "unexpected non-contiguous capture " + name);
        if (tap.feature == Feature::Indices) {
            int count = int(t->ne[0]), rows = int(ggml_nelements(t) / count);
            require(t->type == GGML_TYPE_I32 && rows == n, "index rows " + name);
            std::vector<int> data(rows * count);
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            auto &dst = picks[tap.slot];
            dst.resize(start + rows);
            for (int r = 0; r < rows; ++r) {
                dst[start + r].assign(data.begin() + r * count, data.begin() + (r + 1) * count);
            }
        } else {
            int width = int(t->ne[0]), rows = int(ggml_nelements(t) / width);
            int stride = hid ? 5120 : 512;
            require(t->type == GGML_TYPE_F32 && (rows == n || (hid && rows == 1)) && width >= stride,
                    "unexpected rows for " + name);
            std::vector<float> data(rows * width);
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            auto &dst = hid ? hidden : comp ? global : window[tap.slot];
            const int offset = start + (n - rows);
            dst.resize((offset + rows) * stride);
            for (int r = 0; r < rows; ++r) {
                std::copy_n(data.data() + r * width, stride, dst.data() + (offset + r) * stride);
            }
        }
    }
    ContextRows context(int pos) const {
        ContextRows r;
        require(hidden.size() >= size_t(pos) * 5120 && global.size() >= size_t(pos) * 512,
                "missing hidden or compressed target rows");
        r.hidden.assign(hidden.begin() + (pos - 1) * 5120, hidden.begin() + pos * 5120);
        for (int i = 0; i < 3; ++i) {
            require(window[i].size() >= size_t(pos) * 512 && picks[i].size() >= size_t(pos),
                    "missing index or window target rows");
            auto &k = r.keys[i];
            int begin = std::max(0, pos - 128);
            k.insert(k.end(), window[i].begin() + begin * 512, window[i].begin() + pos * 512);
            for (int idx : picks[i][pos - 1]) {
                if (idx >= 0 && idx < pos) {
                    k.insert(k.end(), global.begin() + idx * 512, global.begin() + (idx + 1) * 512);
                }
            }
        }
        return r;
    }
};
int main(int argc, char **argv) try {
    if (argc < 6 || argc > 8) {
        fprintf(stderr, "usage: probe TARGET SAFETENSORS_DIR|DSPARK_GGUF|- PROMPTS_JSON TOKENS CACHE_MIB [DRAFT_MAX [CTX]]\n");
        return 2;
    }
    n1x_working_set ws;
    require(ws.apply(), "host memory admission");
    std::thread([] {
        for (;;) {
            MEMORYSTATUSEX ms{sizeof(ms)};
            if (GlobalMemoryStatusEx(&ms) &&
                std::min(ms.ullAvailPhys, ms.ullAvailPageFile) < (2ull << 30)) {
                fprintf(stderr, "memory floor reached\n");
                TerminateProcess(GetCurrentProcess(), 75);
            }
            Sleep(100);
        }
    }).detach();
    nlohmann::json prompts;
    std::ifstream(argv[3]) >> prompts;
    require(prompts.is_array() && !prompts.empty(), "expected an array of prompts");
    bool replay = prompts.front().contains("proposals");
    const int limit = std::stoi(argv[4]);
    const int draft_max = argc >= 7 ? std::stoi(argv[6]) : 5;
    const int context = argc >= 8 ? std::stoi(argv[7]) : 32768;
    require(context >= 32 && context <= 32768, "context out of range");
    require(draft_max >= 1 && draft_max <= 5, "expected one to five draft tokens");
    require(limit >= 2 && limit < 32768, "output token budget out of range");
    bool active = std::string(argv[2]) != "-" && std::filesystem::is_directory(argv[2]);
    bool dspark = std::string(argv[2]) != "-" && !active;
    require(!replay || (!active && !dspark), "replay takes target only");
    llama_log_set(
        [](ggml_log_level level, const char *message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                fputs(message, stderr);
            }
        },
        nullptr);
    llama_backend_init();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    auto host = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(dev && host, "missing CUDA or CPU backend");
    auto bp = ggml_backend_dev_buffer_type(host);
    llama_model_tensor_buft_override placement[] = {
        {"ffn_.*_exps\\.weight", bp}, {"per_layer_token_embd\\.weight", bp}, {nullptr, nullptr}};
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.tensor_buft_overrides = placement;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    mp.no_host = true;
    mp.use_extra_bufts = false;
    auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "target model load");
    auto vocab = llama_model_get_vocab(model.get());
    const auto taps = active ? target_taps(model->hparams) : std::vector<Tap>{};
    const bool chat = prompts.front().contains("user_prompt");
    auto chat_template = chat ? common_chat_templates_init(model.get(), "") : nullptr;
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    Backend gpu_backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    Backend cpu_backend(ggml_backend_dev_init(host, nullptr), ggml_backend_free);
    require(gpu_backend && cpu_backend, "backend initialization failed");
    ggml_backend_t backends[] = {gpu_backend.get(), cpu_backend.get()};
    Weights w;
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        nullptr, ggml_backend_sched_free);
    if (active) {
        w.load(argv[2], gpu_backend.get());
        scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true));
        require(bool(scheduler), "draft scheduler initialization failed");
        require(model->tok_embd && model->output && model->tok_embd->ne[0] == 5120 &&
                    model->output->ne[1] == 129280,
                "base embedding/head mismatch");
        fprintf(stderr, "loaded %zu packed draft tensors\n", w.tensors.size());
    }
    Capture cap;
    auto p = llama_context_default_params();
    p.n_ctx = context;
    p.n_batch = p.n_ubatch = 512;
    p.n_threads = p.n_threads_batch = 12;
    p.moe_cache_size = std::stoull(argv[5]) << 20;
    p.n_rs_seq = (active || dspark || replay) ? 6 : 0;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    using GetOutput = Tensor *(*)(llama_context *, const char *);
    auto get_output =
        reinterpret_cast<GetOutput>(GetProcAddress(GetModuleHandleW(L"llama.dll"), "ds4_dfly_output"));
    require(!active || get_output, "DFly2 requires the benchmark llama.dll");
    _putenv_s("DS4_N1X_DFLY2_OUTPUTS", active ? "1" : "");
    auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(
        llama_init_from_model(model.get(), p), llama_free);
    require(bool(ctx), "target context");
    common_params dsparams;
    common_speculative_init_result_ptr dscontexts;
    common_speculative_ptr spec;
    if (dspark) {
        dsparams.speculative.draft.mparams.path = argv[2];
        dsparams.speculative.draft.n_max = draft_max;
        dsparams.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
        dsparams.n_gpu_layers = -2;
        dsparams.n_parallel = 1;
        dsparams.n_batch = dsparams.n_ubatch = 512;
        dsparams.n_outputs_max = dsparams.n_outputs_max_per_seq = 6;
        dsparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        dsparams.n_ctx = p.n_ctx;
        dscontexts = std::make_unique<common_speculative_init_result>(dsparams, model.get(), ctx.get());
        require(dscontexts->context(), "DSpark context failed");
        // DSpark's metadata names checkpoint layers. Its feature extractor takes logical indices.
        for (auto &layer : dscontexts->model()->target_layer_ids) {
            layer = dfly_last_visit(model->hparams, layer);
        }
        dsparams.speculative.draft.ctx_tgt = ctx.get();
        dsparams.speculative.draft.ctx_dft = dscontexts->context();
        spec.reset(common_speculative_init(dsparams.speculative, 1));
        require(bool(spec), "DSpark initialization failed");
    }
    common_batch batch(ctx.get());
    std::vector<float> gaps;
    auto eval = [&](int pos, const std::vector<int> &tokens, bool all_logits) {
        cap.start = pos;
        cap.n = (int)tokens.size();
        batch.clear();
        for (int i = 0; i < (int)tokens.size(); ++i) {
            batch.add(tokens[i], pos + i, 0, all_logits || i + 1 == (int)tokens.size());
        }
        require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode");
        if (spec) {
            require(common_speculative_process(spec.get(), batch), "DSpark target feature injection");
        }
        std::vector<int> next;
        gaps.clear();
        for (int i = all_logits ? 0 : int(tokens.size()) - 1; i < (int)tokens.size(); ++i) {
            auto *logits = llama_get_logits_ith(ctx.get(), i);
            require(logits, "no logits");
            next.push_back(
                (int)(std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits));
            if (pos < 100) {
                int id = next.back();
                float second = -INFINITY;
                for (int k = 0; k < llama_vocab_n_tokens(vocab); ++k) {
                    if (k != id) {
                        second = std::max(second, logits[k]);
                    }
                }
                gaps.push_back(logits[id] - second);
            }
        }
        if (active) {
            for (const auto &tap : taps) {
                auto *t = get_output(ctx.get(), tap.name.c_str());
                require(t && t->buffer, "missing saved output " + tap.name);
                cap.collect(t, tap);
            }
        }
        return next;
    };
    for (auto &entry : prompts) {
        auto mem = llama_get_memory(ctx.get());
        llama_memory_clear(mem, true);
        cap = Capture{};
        if (dspark) {
            llama_memory_clear(llama_get_memory(dscontexts->context()), true);
        }
        std::string text;
        if (chat) {
            common_chat_templates_inputs input;
            // Match ds4-n1x.ps1 run (--nothink) and ShadowArgs::default().system.
            input.messages = {{"system", "You are a helpful assistant"},
                              {"user", entry.at("user_prompt").get<std::string>()}};
            input.enable_thinking = false;
            input.chat_template_kwargs["enable_thinking"] = "false";
            text = common_chat_templates_apply(chat_template.get(), input).prompt.str();
        } else {
            text = (replay ? entry.at("prompt") : entry).get<std::string>();
        }
        std::vector<int> tokens(text.size() + 32);
        int n = llama_tokenize(vocab, text.data(), (int)text.size(), tokens.data(), (int)tokens.size(),
                               !chat, chat);
        require(n > 0 && n + limit + 5 < (int)p.n_ctx, "prompt exceeds context");
        tokens.resize(n);
        auto start = Clock::now();
        int anchor = -1;
        for (int offset = 0; offset < n; offset += 512) {
            std::vector<int> part(tokens.begin() + offset, tokens.begin() + std::min(n, offset + 512));
            anchor = eval(offset, part, false).back();
        }
        if (spec) {
            common_speculative_begin(spec.get(), 0, tokens);
        }
        auto prefill = elapsed(start);
        int pos = n, accepted = 0, proposed = 0, rounds = 0;
        std::vector<std::vector<int>> proposals;
        std::vector<int> generated{anchor};
        double draft_s = 0, verify_s = 0;
        std::vector<float> accepted_gaps;
        if (!gaps.empty()) {
            accepted_gaps.push_back(gaps.back());
        }
        start = Clock::now();
        while ((int)generated.size() < limit && !(chat && llama_vocab_is_eog(vocab, generated.back()))) {
            std::vector<int> cand;
            if (active) {
                auto t = Clock::now();
                auto rows = cap.context(pos);
                DraftGraph g(w, model->tok_embd, model->output);
                auto *out = g.build(anchor, pos, rows, draft_max);
                auto *sched = scheduler.get();
                ggml_backend_sched_reset(sched);
                require(ggml_backend_sched_alloc_graph(sched, g.graph), "allocate draft graph");
                for (auto &[key, data] : g.inputs) {
                    ggml_backend_tensor_set(key, data.data(), 0, data.size());
                }
                require(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS,
                        "draft compute");
                cand.resize(draft_max);
                ggml_backend_tensor_get(out, cand.data(), 0, cand.size() * sizeof(int));
                ggml_backend_sched_reset(sched);
                draft_s += elapsed(t);
            } else if (spec) {
                auto t = Clock::now();
                auto &dp = common_speculative_get_draft_params(spec.get(), 0);
                dp.drafting = true;
                dp.n_max = draft_max;
                dp.pos0 = pos;
                dp.id_last = anchor;
                dp.prompt = &tokens;
                dp.result = &cand;
                dp.temp = 0;
                common_speculative_draft(spec.get());
                require(!cand.empty(), "DSpark returned no proposals");
                require(llama_memory_seq_rm(llama_get_memory(dscontexts->context()), 0, pos, -1),
                        "DSpark draft rewind failed");
                draft_s += elapsed(t);
            } else if (replay) {
                cand = entry.at("proposals").at(rounds).get<std::vector<int>>();
                require(cand.size() <= 5, "invalid replay block size");
                for (auto id : cand) {
                    require(id >= 0 && id < llama_vocab_n_tokens(vocab), "invalid replay token");
                }
            }
            cand.resize(std::min<size_t>(cand.size(), limit - generated.size() - 1));
            proposals.push_back(cand);
            std::vector<int> block{anchor};
            block.insert(block.end(), cand.begin(), cand.end());
            auto t = Clock::now();
            auto verified = eval(pos, block, true);
            size_t accepted_here = 0;
            while (accepted_here < cand.size() && verified[accepted_here] == cand[accepted_here]) {
                ++accepted_here;
                if (chat && llama_vocab_is_eog(vocab, cand[accepted_here - 1])) {
                    break;
                }
            }
            const bool stop = chat && accepted_here && llama_vocab_is_eog(vocab, cand[accepted_here - 1]);
            if (!gaps.empty()) {
                accepted_gaps.insert(accepted_gaps.end(), gaps.begin(),
                                     gaps.begin() + accepted_here + (stop ? 0 : 1));
            }
            generated.insert(generated.end(), cand.begin(), cand.begin() + accepted_here);
            if (!stop) {
                anchor = verified[accepted_here];
                generated.push_back(anchor);
            }
            pos += 1 + (int)accepted_here;
            tokens.insert(tokens.end(), block.begin(), block.begin() + 1 + accepted_here);
            if (spec) {
                common_speculative_accept(spec.get(), 0, (uint16_t)accepted_here);
            }
            if (accepted_here < cand.size()) {
                require(llama_memory_seq_rm(mem, 0, pos, -1), "target rollback failed");
            }
            if (dspark && accepted_here < cand.size()) {
                require(llama_memory_seq_rm(llama_get_memory(dscontexts->context()), 0, pos, -1),
                        "DSpark rollback failed");
            }
            verify_s += elapsed(t);
            accepted += (int)accepted_here;
            proposed += (int)cand.size();
            ++rounds;
        }
        auto sec = elapsed(start);
        nlohmann::json report = {{"prompt", text},           {"tokens", generated},
                                 {"prefill_s", prefill},     {"decode_s", sec},
                                 {"tps", rounds ? (generated.size() - 1) / sec : 0.0}, {"rounds", rounds},
                                 {"proposed", proposed},     {"accepted", accepted},
                                 {"draft_s", draft_s},       {"target_s", verify_s}};
        if (replay) {
            require(report.at("tokens") == entry.at("tokens"),
                    "target-only replay returned a different continuation");
        }
        report["proposals"] = proposals;
        report["logical_layers"] = llama_model_n_layer(model.get());
        if (active) {
            for (const auto &tap : taps) { report["target_taps"].push_back(tap.name); }
        } else if (dspark) {
            report["target_input_layers"] = dscontexts->model()->target_layer_ids;
        }
        if (chat) {
            report["user_prompt"] = entry.at("user_prompt");
            report["prompt_ids"] = std::vector<int>(tokens.begin(), tokens.begin() + n);
            auto visible = generated;
            if (llama_vocab_is_eog(vocab, visible.back())) { visible.pop_back(); }
            report["answer"] = common_detokenize(vocab, visible, false);
            report["output_budget"] = limit;
        }
        report["draft_max"] = draft_max;
        report["top1_margin"] = accepted_gaps;
        report["prompt_tokens"] = n;
        std::cout << report.dump() << std::endl;
    }
    return 0;
} catch (const std::exception &e) {
    fprintf(stderr, "probe failed: %s\n", e.what());
    return 1;
}
