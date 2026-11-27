use super::{Result, Row, Shape};
use crate::tensors::TensorInfo;
use crate::{f16_to_f32, GgufFile, TensorInventory};
use std::collections::HashMap;

// GGUF FP16 and Q8_0 scales use IEEE half, including the subnormal interval.
fn gguf_half(bits: u16) -> f32 {
    if bits & 0x7c00 == 0 {
        let magnitude = f32::from(bits & 0x3ff) / 16_777_216.;
        if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    } else {
        f16_to_f32(bits)
    }
}

pub(super) struct Matrix {
    kind: u32,
    offset: usize,
    stride: usize,
    width: usize,
    rows: usize,
    #[cfg(feature = "varloop-cuda")]
    device: Option<ds4_sys::varloop::Weight>,
}

pub(super) struct Bank {
    file: GgufFile,
    names: HashMap<String, TensorInfo>,
    #[cfg(feature = "varloop-cuda")]
    gpu: Option<ds4_sys::varloop::Device>,
}

impl Bank {
    pub fn new(file: GgufFile, inv: TensorInventory, cuda: bool) -> Result<Self> {
        #[cfg(not(feature = "varloop-cuda"))]
        if cuda {
            return Err("build with varloop-cuda for CUDA projections".into());
        }
        let mut names = HashMap::new();
        for t in inv.tensors {
            if names.insert(t.name.clone(), t).is_some() {
                return Err("duplicate tensor name".into());
            }
        }
        Ok(Self {
            file,
            names,
            #[cfg(feature = "varloop-cuda")]
            gpu: if cuda {
                Some(ds4_sys::varloop::Device::new()?)
            } else {
                None
            },
        })
    }

    pub fn tensor(&self, name: &str, dims: &[usize], _project: bool) -> Result<Matrix> {
        let t = self
            .names
            .get(name)
            .ok_or_else(|| format!("missing {name}"))?;
        // A GGUF vector may omit its trailing singleton query dimension.
        let actual = &t.dim[..t.ndim as usize];
        if (0..actual.len().max(dims.len())).any(|i| {
            actual.get(i).copied().unwrap_or(1) != dims.get(i).copied().unwrap_or(1) as u64
        }) {
            return Err(format!("{name} has wrong shape, expected {dims:?}"));
        }
        let width = dims[0];
        let stride = match t.typ {
            0 => width * 4,
            1 | 30 => width * 2,
            8 if width % 32 == 0 => width / 32 * 34,
            _ => {
                return Err(format!(
                    "{name} has unsupported GGUF type {} or row width",
                    t.typ
                ))
            }
        };
        let bytes = self
            .file
            .as_bytes()
            .get(t.abs_offset as usize..)
            .and_then(|b| b.get(..t.bytes as usize))
            .ok_or_else(|| format!("truncated {name}"))?;
        let size = dims[1..].iter().try_fold(stride, |n, d| n.checked_mul(*d));
        if size != Some(bytes.len()) {
            return Err(format!("{name} storage mismatch"));
        }
        Ok(Matrix {
            kind: t.typ,
            offset: t.abs_offset as usize,
            stride,
            width,
            rows: dims.get(1).copied().unwrap_or(1),
            #[cfg(feature = "varloop-cuda")]
            device: if _project && self.gpu.is_some() {
                Some(ds4_sys::varloop::Weight::upload(
                    bytes,
                    width,
                    dims.get(1).copied().unwrap_or(1),
                    t.typ,
                )?)
            } else {
                None
            },
        })
    }

    pub fn row(&self, t: &Matrix, r: usize) -> Row {
        let bytes = self.row_bytes(t, r);
        match t.kind {
            0 => bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
            1 | 30 => bytes
                .chunks_exact(2)
                .map(|v| {
                    let n = u16::from_le_bytes(v.try_into().unwrap());
                    if t.kind == 1 {
                        gguf_half(n)
                    } else {
                        f32::from_bits((n as u32) << 16)
                    }
                })
                .collect(),
            8 => bytes
                .chunks_exact(34)
                .flat_map(|block| {
                    let scale = gguf_half(u16::from_le_bytes([block[0], block[1]]));
                    block[2..].iter().map(move |&v| (v as i8 as f32) * scale)
                })
                .collect(),
            _ => unreachable!(),
        }
    }

    fn row_bytes(&self, t: &Matrix, row: usize) -> &[u8] {
        let start = t.offset + row * t.stride;
        &self.file.as_bytes()[start..start + t.stride]
    }

    pub fn values(&self, name: &str, dims: &[usize]) -> Result<Row> {
        let t = self.tensor(name, dims, false)?;
        Ok((0..dims[1..].iter().product())
            .flat_map(|r| self.row(&t, r))
            .collect())
    }

    pub fn mv(&self, t: &Matrix, x: &[f32], expert: usize) -> Result<Row> {
        debug_assert_eq!(x.len(), t.width);
        #[cfg(feature = "varloop-cuda")]
        if let (Some(gpu), Some(w)) = (&self.gpu, &t.device) {
            return gpu.mv(w, x, expert);
        }
        let first = expert * t.rows;
        let dot = |r| {
            let bytes = self.row_bytes(t, r);
            match t.kind {
                0 => packed_f32_dot(bytes, x),
                8 => packed_q8_dot(bytes, x),
                _ => super::math::dot(x, &self.row(t, r)),
            }
        };
        Ok((first..first + t.rows).map(dot).collect())
    }
}

// Four independent FP32 sums allow the ARM64 compiler to vectorize without
// enabling relaxed floating-point semantics for the rest of the graph.
fn packed_f32_dot(bytes: &[u8], x: &[f32]) -> f32 {
    let mut sums = [0.; 4];
    let mut weights = bytes.chunks_exact(16);
    let mut input = x.chunks_exact(4);
    for (w, v) in weights.by_ref().zip(input.by_ref()) {
        for (lane, b) in w.chunks_exact(4).enumerate() {
            sums[lane] += f32::from_le_bytes(b.try_into().unwrap()) * v[lane];
        }
    }
    let tail = weights
        .remainder()
        .chunks_exact(4)
        .zip(input.remainder())
        .map(|(b, v)| f32::from_le_bytes(b.try_into().unwrap()) * v)
        .sum::<f32>();
    sums.iter().sum::<f32>() + tail
}

fn packed_q8_dot(bytes: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.;
    for (block, input) in bytes.chunks_exact(34).zip(x.chunks_exact(32)) {
        let mut lanes = [0.; 4];
        for (q, v) in block[2..].chunks_exact(4).zip(input.chunks_exact(4)) {
            for i in 0..4 {
                lanes[i] += (q[i] as i8 as f32) * v[i];
            }
        }
        let scale = gguf_half(u16::from_le_bytes([block[0], block[1]]));
        sum += scale * lanes.iter().sum::<f32>();
    }
    sum
}

pub(super) struct Controller {
    pub query: Row,
    pub down: Matrix,
    pub gain: Matrix,
    pub forget: Matrix,
}
pub(super) struct Layer {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub rel: Matrix,
    pub sink: Matrix,
    pub gate: Matrix,
    pub q_clip: Row,
    pub r_clip: Row,
    pub rel_bias: Row,
    pub taps: [Row; 4],
    pub dynamic: Option<[Matrix; 2]>,
    pub input_norm: Vec<Row>,
    pub attn_query: Option<Row>,
    pub ff_query: Row,
    pub ff_norm: Row,
    pub router: Matrix,
    pub bias: Row,
    pub gu: Matrix,
    pub down: Matrix,
    pub controls: Vec<Controller>,
}

impl Layer {
    pub fn load(b: &mut Bank, s: &Shape, i: usize) -> Result<Self> {
        let key = |n: &str| format!("blk.{i}.{n}.weight");
        let m = |n: &str, d: &[usize]| b.tensor(&key(n), d, true);
        let v = |n: &str, d: &[usize]| b.values(&key(n), d);
        let w = s.width;
        let q = s.head * s.heads;
        let kv = s.head * s.kv;
        let swa = s.swa[i];
        let mut controls = Vec::new();
        if (s.start..s.end).contains(&i) {
            let queries = v("loop_msg_query", &[w, 4])?;
            for (tag, query) in ["q", "k", "v", "ffn"]
                .into_iter()
                .zip(queries.chunks_exact(w))
            {
                let ctrl = |p: &str, d: &[usize]| m(&format!("loop_{tag}_ctrl_{p}"), d);
                controls.push(Controller {
                    query: query.to_vec(),
                    down: ctrl("down", &[2 * w, s.rank])?,
                    gain: ctrl("gain", &[s.rank, w])?,
                    forget: ctrl("forget", &[s.rank, w])?,
                });
            }
        }
        let bias = format!("blk.{i}.exp_probs_b.bias");
        Ok(Self {
            q: m("attn_q", &[w, q])?,
            k: m("attn_k", &[w, kv])?,
            v: m("attn_v", &[w, kv])?,
            o: m("attn_output", &[q, w])?,
            rel: m("attn_rel", &[w, s.rel * s.heads])?,
            gate: m("attn_gate", &[w, q])?,
            sink: m("attn_sink_proj", &[w, s.heads])?,
            q_clip: v("attn_q_clip", &[s.head, s.heads])?,
            r_clip: v("attn_rel_clip", &[s.rel, s.heads])?,
            rel_bias: v(
                "attn_rel_bias",
                &[s.rel, if swa { s.window } else { s.extent }],
            )?,
            taps: [
                v("attn_k_conv", &[s.taps, kv])?,
                v("attn_v_conv", &[s.taps, kv])?,
                v("attn_post_conv", &[s.taps, w])?,
                v("ffn_post_conv", &[s.taps, w])?,
            ],
            dynamic: if swa {
                None
            } else {
                Some([
                    m("attn_k_conv_proj", &[w, s.kv * s.taps])?,
                    m("attn_v_conv_proj", &[w, s.kv * s.taps])?,
                ])
            },
            input_norm: if swa {
                vec![v("attn_norm", &[w])?]
            } else {
                vec![
                    v("attn_q_inp_norm", &[w])?,
                    v("attn_k_inp_norm", &[w])?,
                    v("attn_v_inp_norm", &[w])?,
                ]
            },
            attn_query: if i == 0 {
                None
            } else {
                Some(v("attn_res_query", &[w, if swa { 1 } else { 3 }])?)
            },
            ff_query: v("ffn_res_query", &[w])?,
            ff_norm: v("ffn_norm", &[w])?,
            router: m("ffn_gate_inp", &[w, s.experts])?,
            bias: if b.names.contains_key(&bias) {
                b.values(&bias, &[s.experts])?
            } else {
                vec![0.; s.experts]
            },
            gu: m("ffn_gate_up_exps", &[w, 2 * s.ff, s.experts])?,
            down: m("ffn_down_exps", &[s.ff, w, s.experts])?,
            controls,
        })
    }
}
