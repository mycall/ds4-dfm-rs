//! Text-only Varloop reference session. Weights are shared by physical layer.
//! Each visit owns its causal attention and short-convolution histories.
//! Forward reference: llama.cpp 244640b7, src/models/inkling2.cpp (layer)
//! and src/models/inkling2-loop.cpp (cross-pass bank and cache slots).
mod math;
mod weights;

use crate::{GgufFile, TensorInventory};
use math::{add, clip, mix, norm, sigmoid, softmax};
use std::collections::VecDeque;
use std::path::Path;
use weights::{Bank, Layer, Matrix};

type Row = Vec<f32>;
type Result<T> = std::result::Result<T, String>;

struct Shape {
    width: usize,
    heads: usize,
    kv: usize,
    head: usize,
    rel: usize,
    extent: usize,
    window: usize,
    taps: usize,
    experts: usize,
    used: usize,
    ff: usize,
    rank: usize,
    block: usize,
    start: usize,
    end: usize,
    passes: usize,
    layers: usize,
    ctx: usize,
    vocab: usize,
    padded: usize,
    swa: Vec<bool>,
    eps: f32,
    qclip: f32,
    rclip: f32,
    floor: usize,
    alpha: f32,
    gate_cap: f32,
    up_cap: f32,
    route: f32,
    logit_cap: f32,
}

impl Shape {
    fn read(g: &GgufFile, passes: Option<usize>) -> Result<Self> {
        let key = |s: &str| format!("inkling2_loop.{s}");
        let uint = |k: &str| {
            g.get_u32(&key(k))
                .filter(|n| *n > 0)
                .map(|n| n as usize)
                .ok_or_else(|| format!("invalid {k}"))
        };
        let float = |k: &str| {
            g.get_f32_compat(&key(k))
                .filter(|x| x.is_finite())
                .ok_or_else(|| format!("invalid {k}"))
        };
        if g.get_string("general.architecture") != Some(b"inkling2_loop")
            || g.get_bool(&key("loop.varying")) != Some(true)
            || g.split_count() > 1
        {
            return Err("expected a single variable-loop Inkling GGUF".into());
        }
        let pattern = g
            .get_array(&key("attention.sliding_window_pattern"))
            .ok_or("missing sliding window pattern")?;
        let swa = g.array_bools(&pattern).map_err(|e| e.to_string())?;
        // The exporter writes per-layer head counts even when every layer has
        // the same geometry, as on the released d24 checkpoint.
        let per_layer = |k: &str| -> Result<usize> {
            let Some(a) = g.get_array(&key(k)) else {
                return uint(k);
            };
            if a.len != swa.len() as u64 {
                return Err(format!("invalid {k} length"));
            }
            let counts = g.array_le_u32s(&a).map_err(|e| e.to_string())?;
            let first = counts
                .first()
                .copied()
                .filter(|&n| n != 0 && counts.iter().all(|&v| v == n));
            first
                .map(|n| n as usize)
                .ok_or_else(|| format!("expected uniform {k}"))
        };
        let s = Self {
            width: uint("embedding_length")?,
            heads: per_layer("attention.head_count")?,
            kv: per_layer("attention.head_count_kv")?,
            head: uint("attention.key_length")?,
            rel: uint("attention.relative_length")?,
            extent: uint("attention.relative_extent")?,
            window: uint("attention.sliding_window")?,
            taps: uint("shortconv.l_cache")?,
            experts: uint("expert_count")?,
            used: uint("expert_used_count")?,
            ff: uint("expert_feed_forward_length")?,
            rank: uint("loop.gate_rank")?,
            block: uint("attn_res.block_size")?,
            start: uint("loop.start")?,
            end: uint("loop.end")?,
            passes: passes.unwrap_or(uint("loop.count")?),
            layers: uint("block_count")?,
            vocab: uint("vocab_size_unpadded")?,
            padded: uint("vocab_size")?,
            ctx: uint("context_length")?,
            swa,
            eps: float("attention.layer_norm_rms_epsilon")?,
            qclip: float("attention.qk_clip")?,
            rclip: float("attention.relative_clip")?,
            floor: uint("attention.log_scale_floor")?,
            alpha: float("attention.log_scale_alpha")?,
            gate_cap: float("activation.situ_beta")?,
            up_cap: float("activation.situ_linear_beta")?,
            route: float("expert_weights_scale")?,
            logit_cap: float("final_logit_softcapping")?,
        };
        if !(1..=5).contains(&s.passes)
            || s.start >= s.end
            || s.end > s.layers
            || s.start % s.block != 0
            || s.end % s.block != 0
            || s.layers != s.swa.len()
            || s.layers > 256
            || s.taps < 2
            || s.taps > 64
            || s.heads % s.kv != 0
            || s.used > s.experts
            || s.vocab > s.padded
            || s.head != uint("attention.value_length")?
            || [s.eps, s.qclip, s.rclip, s.gate_cap, s.up_cap, s.route]
                .iter()
                .any(|v| *v <= 0.)
            || s.alpha < 0.
            || s.logit_cap < 0.
        {
            return Err(
                "invalid Varloop dimensions, scales or pass count (supported 1..=5)".into(),
            );
        }
        Ok(s)
    }
}

#[derive(Default)]
struct History {
    keys: VecDeque<Row>,
    values: VecDeque<Row>,
    conv: [VecDeque<Row>; 4],
}

/// Incremental single-sequence text forward pass. Open a new session to change depth.
pub struct Session {
    bank: Bank,
    shape: Shape,
    layers: Vec<Layer>,
    embedding: Matrix,
    emb_norm: Row,
    output: Matrix,
    out_query: Row,
    out_norm: Row,
    state: Vec<History>,
    position: usize,
}

impl Session {
    pub fn open(path: &Path, passes: Option<usize>, cuda: bool) -> Result<Self> {
        let g = GgufFile::open(path).map_err(|e| e.to_string())?;
        let s = Shape::read(&g, passes)?;
        let tensors = TensorInventory::from_file(path, &g).map_err(|e| e.to_string())?;
        let mut bank = Bank::new(g, tensors, cuda)?;
        let embedding = bank.tensor("token_embd.weight", &[s.width, s.padded], false)?;
        let output = bank.tensor("output.weight", &[s.width, s.padded], true)?;
        let emb_norm = bank.values("token_embd_norm.weight", &[s.width])?;
        let out_query = bank.values("output_res_query.weight", &[s.width])?;
        let out_norm = bank.values("output_norm.weight", &[s.width])?;
        let layers = (0..s.layers)
            .map(|i| Layer::load(&mut bank, &s, i))
            .collect::<Result<_>>()?;
        let visits = s.layers + (s.passes - 1) * (s.end - s.start);
        Ok(Self {
            bank,
            shape: s,
            layers,
            embedding,
            emb_norm,
            output,
            out_query,
            out_norm,
            state: (0..visits).map(|_| History::default()).collect(),
            position: 0,
        })
    }

    /// Append one token. Only project the vocabulary when the caller needs logits.
    pub fn step(&mut self, token: u32, logits: bool) -> Result<Option<Row>> {
        let s = &self.shape;
        if token as usize >= s.vocab || self.position >= s.ctx {
            return Err("token or position outside model".into());
        }
        let mut cur = self.bank.row(&self.embedding, token as usize);
        norm(&mut cur, Some(&self.emb_norm), s.eps);
        let mut blocks = vec![];
        let mut visit = 0;
        // Physical weights repeat. Temporal state follows execution order and persists across tokens.
        let mut layer = |i: usize, cur: Row, blocks: &mut Vec<Row>, previous: &[Row]| {
            let value = self.layers[i].forward(
                &self.bank,
                s,
                i,
                self.position,
                cur,
                blocks,
                previous,
                &mut self.state[visit],
            );
            visit += 1;
            value
        };
        let run = || -> Result<_> {
            for i in 0..s.start {
                cur = layer(i, cur, &mut blocks, &[])?;
            }
            let prelude = cur.clone();
            let saved = blocks.clone();
            let mut prior = Vec::new();
            for pass in 0..s.passes {
                if pass > 0 {
                    // The first core block's input is the prelude itself. The bank contains
                    // the remaining block inputs and the last core output from the previous pass.
                    prior = blocks.split_off(saved.len() + 1);
                    prior.push(cur);
                    cur = prelude.clone();
                    blocks.clone_from(&saved);
                }
                for i in s.start..s.end {
                    cur = layer(i, cur, &mut blocks, &prior)?;
                }
            }
            for i in s.end..s.layers {
                cur = layer(i, cur, &mut blocks, &[])?;
            }
            if !logits {
                return Ok(None);
            }
            let mut last = mix(&blocks, &cur, &self.out_query);
            norm(&mut last, Some(&self.out_norm), s.eps);
            let mut out = self.bank.mv(&self.output, &last, 0)?;
            out.truncate(s.vocab);
            if s.logit_cap > 0. {
                for z in &mut out {
                    *z = s.logit_cap * (*z / s.logit_cap).tanh();
                }
            }
            Ok(Some(out))
        };
        let result = run();
        // A device error can occur after a subset of the histories have advanced.
        self.position = if result.is_ok() {
            self.position + 1
        } else {
            s.ctx
        };
        result
    }
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        b: &Bank,
        s: &Shape,
        i: usize,
        pos: usize,
        cur: Row,
        blocks: &mut Vec<Row>,
        prior: &[Row],
        h: &mut History,
    ) -> Result<Row> {
        let mut input: [Row; 3] = std::array::from_fn(|j| match &self.attn_query {
            Some(q) => {
                let n = if s.swa[i] { 0 } else { j };
                mix(blocks, &cur, &q[n * s.width..(n + 1) * s.width])
            }
            None => cur.clone(),
        });
        for (j, x) in input.iter_mut().enumerate() {
            self.control(b, x, prior, j)?;
            norm(
                x,
                Some(&self.input_norm[if s.swa[i] { 0 } else { j }]),
                s.eps,
            );
        }
        let [xq, xk, xv] = &input;
        let mut q = b.mv(&self.q, xq, 0)?;
        let mut k = b.mv(&self.k, xk, 0)?;
        let mut v = b.mv(&self.v, xv, 0)?;
        let mut r = b.mv(&self.rel, xq, 0)?;
        let dynamic =
            |j: usize, x: &Row| self.dynamic.as_ref().map(|d| b.mv(&d[j], x, 0)).transpose();
        math::conv(
            &mut k,
            &self.taps[0],
            dynamic(0, xk)?.as_deref(),
            s,
            &mut h.conv[0],
        );
        math::conv(
            &mut v,
            &self.taps[1],
            dynamic(1, xv)?.as_deref(),
            s,
            &mut h.conv[1],
        );
        let tau = if s.swa[i] {
            1.
        } else {
            1. + s.alpha * ((pos + 1) as f32 / s.floor as f32).max(1.).ln()
        };
        clip(&mut q, Some(&self.q_clip), s.head, s.qclip, tau);
        clip(&mut k, None, s.head, s.qclip, 1.);
        clip(&mut r, Some(&self.r_clip), s.rel, s.rclip, tau);
        for row in v.chunks_exact_mut(s.head) {
            norm(row, None, s.eps);
        }
        if s.swa[i] && h.keys.len() == s.window {
            h.keys.pop_front();
            h.values.pop_front();
        }
        h.keys.push_back(k);
        h.values.push_back(v);
        let sinks = b.mv(&self.sink, xq, 0)?;
        let gates = b.mv(&self.gate, xq, 0)?;
        let mut attn = vec![0.; q.len()];
        for head in 0..s.heads {
            let qi = head * s.head;
            let ki = (head / (s.heads / s.kv)) * s.head;
            let rel = &r[head * s.rel..(head + 1) * s.rel];
            let mut scores = vec![sinks[head]];
            for (t, key) in h.keys.iter().enumerate() {
                let lag = h.keys.len() - 1 - t;
                let bias = self
                    .rel_bias
                    .chunks_exact(s.rel)
                    .nth(lag)
                    .map_or(0., |w| math::dot(w, rel));
                scores.push(
                    math::dot(&q[qi..qi + s.head], &key[ki..ki + s.head]) / (s.head as f32).sqrt()
                        + bias,
                );
            }
            softmax(&mut scores);
            for (p, val) in scores[1..].iter().zip(&h.values) {
                for d in 0..s.head {
                    attn[qi + d] += p * val[ki + d];
                }
            }
        }
        for (a, g) in attn.iter_mut().zip(gates) {
            *a *= sigmoid(g);
        }
        let mut res = b.mv(&self.o, &attn, 0)?;
        math::conv(&mut res, &self.taps[2], None, s, &mut h.conv[2]);
        if i % s.block == 0 {
            blocks.push(cur);
        } else {
            add(&mut res, &cur);
        }
        let mut x = mix(blocks, &res, &self.ff_query);
        self.control(b, &mut x, prior, 3)?;
        norm(&mut x, Some(&self.ff_norm), s.eps);
        let mut ff = self.moe(b, s, &x)?;
        math::conv(&mut ff, &self.taps[3], None, s, &mut h.conv[3]);
        add(&mut res, &ff);
        Ok(res)
    }

    fn control(&self, b: &Bank, x: &mut Row, prior: &[Row], j: usize) -> Result<()> {
        let Some((last, rest)) = prior.split_last() else {
            return Ok(());
        };
        let c = &self.controls[j];
        let message = mix(rest, last, &c.query);
        let mut both = x.clone();
        norm(&mut both, None, 1e-6);
        let mut msg = message.clone();
        norm(&mut msg, None, 1e-6);
        both.extend(msg);
        let low = b.mv(&c.down, &both, 0)?;
        let gain = b.mv(&c.gain, &low, 0)?;
        let forget = b.mv(&c.forget, &low, 0)?;
        for j in 0..x.len() {
            x[j] = (1. - forget[j].tanh()) * x[j] + gain[j].tanh() * message[j];
        }
        Ok(())
    }

    fn moe(&self, b: &Bank, s: &Shape, x: &Row) -> Result<Row> {
        let prob: Row = b.mv(&self.router, x, 0)?.into_iter().map(sigmoid).collect();
        let mut ids: Vec<_> = (0..s.experts).collect();
        let score = |i: usize| prob[i] + self.bias[i];
        ids.sort_unstable_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
        let ids = &ids[..s.used];
        let sum = ids.iter().map(|&id| prob[id]).sum::<f32>().max(1e-20);
        let mut out = vec![0.; s.width];
        for &id in ids {
            let gu = b.mv(&self.gu, x, id)?;
            let (g, u) = gu.split_at(s.ff);
            let act = g
                .iter()
                .zip(u)
                .map(|(&g, &u)| {
                    s.gate_cap
                        * (g / s.gate_cap).tanh()
                        * sigmoid(g)
                        * s.up_cap
                        * (u / s.up_cap).tanh()
                })
                .collect::<Row>();
            let down = b.mv(&self.down, &act, id)?;
            let weight = s.route * prob[id] / sum;
            for (v, z) in out.iter_mut().zip(down) {
                *v += weight * z;
            }
        }
        Ok(out)
    }
}
