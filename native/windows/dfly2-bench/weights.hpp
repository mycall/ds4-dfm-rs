#pragma once
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline void require(bool ok, const std::string &message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
struct Source {
    std::filesystem::path file;
    size_t offset, size;
    std::string dtype;
    std::vector<int64_t> shape;
    void read(void *dest) const {
        std::ifstream f(file, std::ios::binary);
        f.seekg(offset);
        require(bool(f.read(static_cast<char *>(dest), size)), "incomplete tensor in " + file.string());
    }
};
using Tensor = ggml_tensor;
struct Weights {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, Tensor *> tensors;
    Tensor *at(const std::string &name) const { return tensors.at(name); }
    ~Weights() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
    void load(const std::filesystem::path &dir, ggml_backend_t backend) {
        nlohmann::json config;
        std::ifstream(dir / "config.json") >> config;
        const auto &draft = config.at("dfly_config");
        const auto layers = nlohmann::json::array({28, 32, 36});
        require(config.at("architectures") == nlohmann::json::array({"DFlyDeepseekV41DraftModel"}) &&
                    config.at("text_config").at("hidden_size") == 5120 && draft.at("block_size") == 6 &&
                    draft.at("target_kv_layers") == layers &&
                    draft.at("target_index_layers") == layers && draft.at("mask_token_id") == 128799 &&
                    draft.at("causal_block") == false,
                "expected the DS-V4.1-Flash-DFly2-B6 checkpoint");
        std::map<std::string, Source> entries;
        for (auto &de : std::filesystem::directory_iterator(dir)) {
            if (de.path().extension() != ".safetensors") {
                continue;
            }
            std::ifstream f(de.path(), std::ios::binary);
            uint64_t len = 0;
            f.read(reinterpret_cast<char *>(&len), 8);
            require(f && len < 16 * 1024 * 1024 && len + 8 <= de.file_size(), "invalid tensor header");
            std::string header(len, ' ');
            f.read(header.data(), len);
            auto j = nlohmann::json::parse(header);
            for (const auto &[name, meta] : j.items()) {
                if (name == "__metadata__") {
                    continue;
                }
                auto offsets = meta["data_offsets"].get<std::array<size_t, 2>>();
                require(offsets[1] >= offsets[0] && offsets[1] <= de.file_size() - 8 - len,
                        "tensor bounds");
                require(
                    entries
                        .emplace(name, Source{de.path(), len + 8 + offsets[0], offsets[1] - offsets[0],
                                              meta["dtype"].get<std::string>(),
                                              meta["shape"].get<std::vector<int64_t>>()})
                        .second,
                    "duplicate tensor");
            }
        }
        ctx = ggml_init({ggml_tensor_overhead() * 256 + 4096, nullptr, true});
        require(ctx, "draft weight metadata allocation failed");
        for (auto &[name, s] : entries) {
            if (name.find(".experts.") != std::string::npos) {
                continue;
            }
            require(s.dtype == "F32" || s.dtype == "BF16", "unknown precision " + name);
            auto dims = s.shape;
            std::reverse(dims.begin(), dims.end());
            if (name.ends_with(".wo_a.weight")) {
                dims = {4096, 1024, 8};
            }
            require(!dims.empty() && dims.size() <= GGML_MAX_DIMS, "unexpected tensor rank " + name);
            size_t bytes = s.dtype == "F32" ? 4 : 2;
            for (auto dim : dims) {
                require(dim > 0 && size_t(dim) <= s.size / bytes, "invalid tensor dimensions " + name);
                bytes *= size_t(dim);
            }
            require(bytes == s.size, "unexpected tensor byte count " + name);
            bool convert = s.dtype == "BF16" && (dims.size() == 1 || name.ends_with(".ffn.gate.weight"));
            auto *t =
                ggml_new_tensor(ctx, (s.dtype == "F32" || convert) ? GGML_TYPE_F32 : GGML_TYPE_BF16,
                                (int)dims.size(), dims.data());
            tensors.emplace(name, t);
        }
        for (int l = 0; l < 3; ++l) {
            for (int w = 1; w <= 3; ++w) {
                auto name = "model.layers." + std::to_string(l) + ".ffn.experts.w" + std::to_string(w);
                tensors[name] = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, w == 2 ? 2304 : 5120,
                                                   w == 2 ? 5120 : 2304, 128);
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "drafter weight allocation failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto &[name, t] : tensors) {
            ggml_set_name(t, name.c_str());
            if (auto ix = entries.find(name); ix != entries.end()) {
                auto &s = ix->second;
                std::vector<uint8_t> v(s.size);
                s.read(v.data());
                if (s.dtype == "BF16" && t->type == GGML_TYPE_F32) {
                    std::vector<float> dst(s.size / 2);
                    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(v.data()), dst.data(),
                                          dst.size());
                    ggml_backend_tensor_set(t, dst.data(), 0, ggml_nbytes(t));
                } else {
                    require(v.size() == ggml_nbytes(t), "weight size " + name);
                    ggml_backend_tensor_set(t, v.data(), 0, v.size());
                }
                continue;
            }
            auto split = name.rfind(".w");
            const size_t nblocks = t->ne[0] * t->ne[1] / 32;
            std::vector<uint8_t> packed(nblocks * 16), scales(nblocks), dest(nblocks * 17);
            for (int e = 0; e < 128; ++e) {
                auto prefix = name.substr(0, split) + "." + std::to_string(e) + name.substr(split);
                auto &sw = entries.at(prefix + ".weight");
                auto &ss = entries.at(prefix + ".scale");
                require(sw.size == packed.size() && ss.size == scales.size() && sw.dtype == "I8" &&
                            ss.dtype == "F8_E8M0",
                        "expert layout " + prefix);
                sw.read(packed.data());
                ss.read(scales.data());
                // HF stores adjacent nibbles together. GGML pairs element i with i+16 in each group.
                for (size_t b = 0; b < nblocks; ++b) {
                    auto *dst = dest.data() + b * 17;
                    const auto *src = packed.data() + b * 16;
                    dst[0] = scales[b];
                    for (int i = 0; i < 8; ++i) {
                        dst[1 + 2 * i] = (src[i] & 15) | (src[i + 8] << 4);
                        dst[2 + 2 * i] = (src[i] >> 4) | (src[i + 8] & 240);
                    }
                }
                ggml_backend_tensor_set(t, dest.data(), e * dest.size(), dest.size());
            }
        }
    }
};
