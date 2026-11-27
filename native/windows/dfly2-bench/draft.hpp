#pragma once
#include "weights.hpp"
#include <cmath>
#include <cstring>
struct ContextRows {
    std::array<std::vector<float>, 3> keys;
    std::vector<float> hidden;
};
struct DraftGraph {
    ggml_context *c;
    ggml_cgraph *graph;
    const Weights &weights;
    Tensor *embedding, *head;
    std::map<Tensor *, std::vector<char>> inputs;
    DraftGraph(const Weights &w, Tensor *e, Tensor *h) : weights(w), embedding(e), head(h) {
        c = ggml_init(
            {ggml_tensor_overhead() * 5000 + ggml_graph_overhead_custom(4096, false), nullptr, true});
        graph = ggml_new_graph_custom(c, 4096, false);
    }
    ~DraftGraph() { ggml_free(c); }
    Tensor *w(const std::string &key) { return weights.at("model." + key); }
    Tensor *mm(const std::string &key, Tensor *x) { return ggml_mul_mat(c, w(key + ".weight"), x); }
    Tensor *norm(const std::string &key, Tensor *x) {
        return ggml_mul(c, ggml_rms_norm(c, x, 1e-20f), w(key + ".weight"));
    }
    Tensor *input(ggml_type dt, int width, int rows, const void *data) {
        auto *t = ggml_new_tensor_2d(c, dt, width, rows);
        ggml_set_input(t);
        inputs[t].assign((const char *)data, (const char *)data + ggml_nbytes(t));
        return t;
    }
    Tensor *ids(const std::vector<int> &v) { return input(GGML_TYPE_I32, (int)v.size(), 1, v.data()); }
    Tensor *row(Tensor *x, int i) {
        return ggml_view_2d(c, x, x->ne[0], 1, x->nb[1], size_t(i) * x->nb[1]);
    }
    Tensor *trans(Tensor *x) { return ggml_cont(c, ggml_transpose(c, x)); }
    Tensor *rope(Tensor *x, Tensor *p, bool inverse = false) {
        const float m = 1.f / (1.f + 0.1f * std::log(16.f));
        auto *z = inverse
                      ? ggml_rope_ext_back(c, x, p, nullptr, 64, 0, 65536, 160000, 1.f / 16, 1, m, 32, 1)
                      : ggml_rope_ext(c, x, p, nullptr, 64, 0, 65536, 160000, 1.f / 16, 1, m, 32, 1);
        return ggml_rope_set_offset(z, 448);
    }
    Tensor *attention(Tensor *q, Tensor *k, Tensor *sink) {
        auto *kp = ggml_reshape_3d(c, ggml_cast(c, k, GGML_TYPE_F16), 512, k->ne[1], 1);
        auto *qp = ggml_permute(c, q, 0, 2, 1, 3);
        auto *out = ggml_flash_attn_ext(c, qp, kp, kp, nullptr, 1.f / std::sqrt(512.f), 0, 0);
        ggml_flash_attn_ext_add_sinks(out, sink);
        return out;
    }
    Tensor *project(const std::string &path, Tensor *x, Tensor *pos) {
        int n = (int)x->ne[2];
        x = rope(x, pos, true);
        x = ggml_permute(c, ggml_reshape_3d(c, x, 4096, 8, n), 0, 2, 1, 3);
        x = mm(path + "wo_a", x);
        return mm(path + "wo_b", ggml_cont_2d(c, ggml_permute(c, x, 0, 2, 1, 3), 8192, n));
    }
    struct Mix {
        Tensor *pre, *post, *comb;
    };
    Mix mix(const std::string &base, Tensor *x) {
        auto *f = ggml_reshape_2d(c, x, 5120 * 4, 6);
        auto *v = ggml_mul_mat(c, w(base + "_fn"), ggml_rms_norm(c, f, 1e-20f));
        auto *s = w(base + "_scale"), *b = w(base + "_base");
        auto part = [&](int i) {
            auto *vs = ggml_view_2d(c, v, 4, 6, v->nb[1], i * 4 * sizeof(float));
            auto *bs = ggml_view_1d(c, b, 4, i * 4 * sizeof(float));
            auto *ss = ggml_view_1d(c, s, 1, i * sizeof(float));
            return ggml_sigmoid(c, ggml_add(c, ggml_mul(c, vs, ss), bs));
        };
        return {ggml_scale_bias(c, part(0), 1, 1e-6f), ggml_scale(c, part(1), 2),
                ggml_dsv4_hc_comb(c, v, s, b, 1e-6f, 20)};
    }
    Tensor *ffn(const std::string &p, Tensor *x) {
        auto *prob = ggml_sqrt(c, ggml_softplus(c, mm(p + "gate", x)));
        auto *chosen = ggml_top_k(c, ggml_add(c, prob, w(p + "gate.bias")), 3);
        auto *scale =
            ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, prob, 1, 128, 6), chosen), 3, 6);
        auto *denominator = ggml_scale_bias(c, ggml_sum_rows(c, scale), 1.f, 1e-20f);
        scale = ggml_reshape_3d(c, ggml_scale(c, ggml_div(c, scale, denominator), 1.5f), 1, 3, 6);
        auto *inp = ggml_reshape_3d(c, x, 5120, 1, 6);
        auto exp = [&](int i, Tensor *y) {
            return ggml_mul_mat_id(c, w(p + "experts.w" + std::to_string(i)), y, chosen);
        };
        auto act = [&](Tensor *g, Tensor *u) {
            return ggml_mul(c, ggml_silu(c, ggml_clamp(c, g, -INFINITY, 10.f)),
                            ggml_clamp(c, u, -10.f, 10.f));
        };
        auto *experts = ggml_mul(c, exp(2, act(exp(1, inp), exp(3, inp))), scale);
        auto er = [&](int i) {
            return ggml_view_2d(c, experts, 5120, 6, experts->nb[2], i * experts->nb[1]);
        };
        auto *r = ggml_add(c, ggml_add(c, er(0), er(1)), er(2));
        auto s = p + "shared_experts.";
        return ggml_add(c, r, mm(s + "w2", act(mm(s + "w1", x), mm(s + "w3", x))));
    }
    Tensor *build(int anchor, int position, const ContextRows &rows, int draft_max) {
        auto *token = ids({anchor, 128799, 128799, 128799, 128799, 128799});
        auto *emb = ggml_get_rows(c, embedding, token);
        auto *prev = input(GGML_TYPE_F32, 5120, 1, rows.hidden.data());
        auto *anchor_emb = row(emb, 0);
        auto *fusion = mm("mtp_fusion.eh_proj", ggml_concat(c, norm("mtp_fusion.enorm", anchor_emb),
                                                            norm("mtp_fusion.hnorm", prev), 0));
        auto *first = ggml_add(c, anchor_emb, ggml_mul(c, fusion, w("mtp_fusion.gate")));
        auto *rest = ggml_view_2d(c, emb, 5120, 5, emb->nb[1], emb->nb[1]);
        auto *collapsed = ggml_concat(c, first, rest, 1);
        // The six slots share target keys and attend to one another without a causal mask.
        auto *h = ggml_repeat_4d(c, ggml_reshape_3d(c, collapsed, 5120, 1, 6), 5120, 4, 6, 1);
        std::vector<int> positions(6);
        for (int i = 0; i < 6; ++i) {
            positions[i] = position + i;
        }
        auto *pos = ids(positions);
        Tensor *last_context = nullptr;
        for (int i = 0; i < 3; ++i) {
            auto p = "layers." + std::to_string(i) + ".", a = p + "attn.";
            auto m = mix(p + "hc_attn", h);
            auto *x = norm(p + "attn_norm", collapsed);
            auto *q = rope(
                ggml_reshape_3d(c, mm(a + "wq_b", norm(a + "q_norm", mm(a + "wq_a", x))), 512, 64, 6),
                pos);
            auto *own = rope(ggml_reshape_3d(c, norm(a + "kv_norm", mm(a + "wkv", x)), 512, 1, 6), pos);
            require(!rows.keys[i].empty() && rows.keys[i].size() % 512 == 0, "missing target keys");
            last_context =
                input(GGML_TYPE_F32, 512, int(rows.keys[i].size() / 512), rows.keys[i].data());
            auto *keys = ggml_concat(c, last_context, ggml_reshape_2d(c, own, 512, 6), 1);
            x = project(a, attention(q, keys, w(a + "attn_sink")), pos);
            h = ggml_dsv4_hc_post(c, x, h, m.post, m.comb);
            // AngelSpec's mHC carries the previous sublayer's pre-mix into this sublayer.
            auto f = mix(p + "hc_ffn", h);
            x = ffn(p + "ffn.", norm(p + "ffn_norm", ggml_dsv4_hc_pre(c, h, m.pre)));
            h = ggml_dsv4_hc_post(c, x, h, f.post, f.comb);
            collapsed = ggml_dsv4_hc_pre(c, h, f.pre);
        }
        auto *block = norm("final_norm", collapsed);
        const std::string p = "hidden_correction.", a = p + "attention.";
        Tensor *previous = ids({anchor}), *block_keys = nullptr, *out = nullptr;
        // Correction is causal. Slot zero supplies a key, then each corrected argmax feeds its
        // embedding into the following slot. All five proposals use the target's output head.
        for (int slot = 0; slot <= draft_max; ++slot) {
            auto *hidden = row(block, slot);
            auto *inp = ggml_concat(c, norm(p + "hidden_norm", hidden),
                                    norm(p + "embed_norm", ggml_get_rows(c, embedding, previous)), 0);
            auto *slot_pos = ids({position + slot});
            auto *bk =
                rope(ggml_reshape_3d(c, norm(a + "kv_norm", mm(a + "kv_in", inp)), 512, 1, 1), slot_pos);
            block_keys = block_keys ? ggml_concat(c, block_keys, bk, 1) : bk;
            if (slot == 0) {
                continue;
            }
            auto *q = rope(
                ggml_reshape_3d(c, mm(a + "wq_b", norm(a + "q_norm", mm(a + "q_in", inp))), 512, 64, 1),
                slot_pos);
            auto *context_read = attention(q, last_context, w(a + "attn_sink"));
            auto *scores = ggml_scale(c, ggml_mul_mat(c, block_keys, q), 1.f / std::sqrt(512.f));
            std::vector<int> distances;
            for (int i = slot; i >= 0; --i) {
                distances.push_back(i);
            }
            auto *bias = trans(ggml_get_rows(c, trans(w(a + "block_rel_bias")), ids(distances)));
            scores = ggml_add(c, scores, bias);
            auto *sink = ggml_reshape_2d(c, w(a + "block_sink"), 1, 64);
            auto *probs = ggml_soft_max(c, ggml_concat(c, scores, sink, 0));
            probs = ggml_cont(c, ggml_view_2d(c, probs, slot + 1, 64, probs->nb[1], 0));
            auto *br = ggml_mul_mat(c, trans(block_keys), probs);
            auto *gb = ggml_get_rows(c, trans(w(a + "block_gate_bias")), ids({slot}));
            auto *gate =
                ggml_reshape_2d(c, ggml_tanh(c, ggml_add(c, mm(a + "block_gate", inp), gb)), 1, 64);
            auto *read = project(a, ggml_add(c, context_read, ggml_mul(c, br, gate)), slot_pos);
            auto *delta = mm(p + "down_proj", ggml_mul(c, ggml_silu(c, mm(p + "gate_proj", inp)),
                                                       mm(p + "up_proj", inp)));
            previous =
                ggml_argmax(c, ggml_mul_mat(c, head, ggml_add(c, hidden, ggml_add(c, read, delta))));
            out = out ? ggml_concat(c, out, previous, 0) : previous;
        }
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        return out;
    }
};
