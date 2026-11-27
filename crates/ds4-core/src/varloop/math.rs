use super::{Row, Shape};
use std::collections::VecDeque;

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub(super) fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}
pub(super) fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub(super) fn norm(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let scale = (dot(x, x) / x.len() as f32 + eps).sqrt().recip();
    for (i, z) in x.iter_mut().enumerate() {
        *z *= scale * w.map_or(1., |v| v[i]);
    }
}

pub(super) fn clip(x: &mut [f32], w: Option<&[f32]>, head: usize, limit: f32, tau: f32) {
    if let Some(weights) = w {
        for (v, w) in x.iter_mut().zip(weights) {
            *v *= w;
        }
    }
    for row in x.chunks_exact_mut(head) {
        let scale = tau * limit / dot(row, row).sqrt().max(limit);
        for v in row {
            *v *= scale;
        }
    }
}

pub(super) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}

// Depth attention normalizes both keys and values. 'partial' is the latest block.
pub(super) fn mix(blocks: &[Row], partial: &[f32], query: &[f32]) -> Row {
    let width = partial.len();
    let mut scores = Vec::with_capacity(blocks.len() + 1);
    let mut scales = Vec::with_capacity(scores.capacity());
    let states = || {
        blocks
            .iter()
            .map(Vec::as_slice)
            .chain(std::iter::once(partial))
    };
    for state in states() {
        let scale = (dot(state, state) / width as f32 + 1e-6).sqrt().recip();
        scales.push(scale);
        scores.push(dot(state, query) * scale / (width as f32).sqrt());
    }
    softmax(&mut scores);
    let mut out = vec![0.; width];
    for ((state, p), scale) in states().zip(scores).zip(scales) {
        for (value, x) in out.iter_mut().zip(state) {
            *value += x * (p * scale);
        }
    }
    out
}

// GGUF's static taps run oldest to newest. Dynamic taps run newest to oldest.
pub(super) fn conv(
    x: &mut Row,
    w: &[f32],
    dynamic: Option<&[f32]>,
    s: &Shape,
    past: &mut VecDeque<Row>,
) {
    let mut out = x.clone();
    let den = ((if dynamic.is_some() { 2 } else { 1 }) * s.taps + 1) as f32;
    for (lag, row) in std::iter::once(&*x).chain(past.iter().rev()).enumerate() {
        for (i, v) in row.iter().enumerate() {
            let tap = w[i * s.taps + s.taps - 1 - lag]
                + dynamic.map_or(0., |d| d[(i / s.head) * s.taps + lag]);
            out[i] += v * tap;
        }
    }
    for v in &mut out {
        *v /= den.sqrt();
    }
    past.push_back(std::mem::replace(x, out));
    if past.len() >= s.taps {
        past.pop_front();
    }
}
