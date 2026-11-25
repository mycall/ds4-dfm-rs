use crate::{sample::draw, Sampler};
use ds4_sys::n1x_gguf::*;
use std::ffi::{c_void, CString};
use std::ptr::NonNull;

/// Owns the native graph and its current sequence. A logits row borrows the
/// graph until the next eval, so consume it before returning to the caller.
pub struct Session {
    handle: NonNull<c_void>,
    vocab: usize,
}

impl Session {
    pub fn open(path: &str, gpu: bool, ctx: i32, threads: i32, vocab: i32) -> Result<Self, String> {
        let path = CString::new(path).map_err(|e| e.to_string())?;
        if ctx <= 0 || vocab <= 0 {
            return Err("invalid model dimensions".into());
        }
        let handle =
            unsafe { ds4_bridge_gguf_open(path.as_ptr(), gpu.into(), ctx, threads, vocab) };
        Ok(Self {
            handle: NonNull::new(handle).ok_or("DeepSeek 4.1 GGUF initialization failed")?,
            vocab: vocab as usize,
        })
    }

    pub fn sample(
        &mut self,
        tokens: &[i32],
        sampler: &Sampler,
        rng: &mut u64,
    ) -> Result<i32, String> {
        let count = i32::try_from(tokens.len()).map_err(|e| e.to_string())?;
        if count == 0 || tokens.iter().any(|&id| id < 0 || id as usize >= self.vocab) {
            return Err("invalid DeepSeek 4.1 input token sequence".into());
        }
        let row = unsafe { ds4_bridge_gguf_eval(self.handle.as_ptr(), tokens.as_ptr(), count) };
        if row.is_null() {
            return Err("DeepSeek 4.1 GGUF forward pass failed".into());
        }
        // Open checked the Rust vocabulary length against the native graph.
        let logits = unsafe { std::slice::from_raw_parts(row, self.vocab) };
        if !logits.iter().any(|x| x.is_finite()) {
            return Err("DeepSeek 4.1 GGUF produced no finite logits".into());
        }
        let id = draw(logits, sampler, rng, [-1, -1]);
        if id < 0 {
            return Err("no token available to sample".into());
        }
        Ok(id)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { ds4_bridge_gguf_close(self.handle.as_ptr()) }
    }
}
