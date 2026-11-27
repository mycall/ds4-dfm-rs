//! Safe ownership of the native single-row projection kernel and its scratch.
use std::{cell::Cell, ffi::c_void, marker::PhantomData, ptr::NonNull};

extern "C" {
    fn ds4_bridge_varloop_device() -> *mut c_void;
    fn ds4_bridge_varloop_close(d: *mut c_void);
    fn ds4_bridge_varloop_upload(p: *const u8, n: usize) -> *mut c_void;
    fn ds4_bridge_varloop_free(p: *mut c_void);
    fn ds4_bridge_varloop_mv(
        d: *mut c_void,
        p: *const c_void,
        row: usize,
        kind: u32,
        width: i32,
        rows: i32,
        x: *const f32,
        out: *mut f32,
    ) -> i32;
}

pub struct Device(NonNull<c_void>, PhantomData<Cell<()>>);
pub struct Weight {
    ptr: NonNull<c_void>,
    width: i32,
    rows: i32,
    groups: usize,
    kind: u32,
}
impl Device {
    pub fn new() -> Result<Self, String> {
        // SAFETY: native returns an owned, empty scratch context or null.
        NonNull::new(unsafe { ds4_bridge_varloop_device() })
            .map(|p| Self(p, PhantomData))
            .ok_or("CUDA scratch allocation failed".into())
    }
    pub fn mv(&self, w: &Weight, x: &[f32], group: usize) -> Result<Vec<f32>, String> {
        if x.len() != w.width as usize || group >= w.groups {
            return Err("invalid CUDA projection".into());
        }
        let mut out = vec![0.; w.rows as usize];
        // SAFETY: checked dimensions and group cover the uploaded byte range. The call is synchronous.
        let status = unsafe {
            ds4_bridge_varloop_mv(
                self.0.as_ptr(),
                w.ptr.as_ptr(),
                group * w.rows as usize,
                w.kind,
                w.width,
                w.rows,
                x.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(format!("CUDA projection failed (runtime status {status})"));
        }
        Ok(out)
    }
}
impl Weight {
    pub fn upload(bytes: &[u8], width: usize, rows: usize, kind: u32) -> Result<Self, String> {
        let row = match kind {
            0 => width.checked_mul(4),
            1 | 30 => width.checked_mul(2),
            8 if width % 32 == 0 => (width / 32).checked_mul(34),
            _ => None,
        };
        let group = row
            .and_then(|n| n.checked_mul(rows))
            .filter(|&n| n > 0 && bytes.len() % n == 0 && !bytes.is_empty())
            .ok_or("invalid CUDA matrix")?;
        let (width, rows) = (
            i32::try_from(width).map_err(|_| "matrix too wide")?,
            i32::try_from(rows).map_err(|_| "too many rows")?,
        );
        // SAFETY: native copies bytes before returning, retaining only device memory.
        let ptr = NonNull::new(unsafe { ds4_bridge_varloop_upload(bytes.as_ptr(), bytes.len()) })
            .ok_or("CUDA weight upload failed")?;
        Ok(Self {
            ptr,
            width,
            rows,
            groups: bytes.len() / group,
            kind,
        })
    }
}
impl Drop for Weight {
    fn drop(&mut self) {
        // SAFETY: this handle exclusively owns the uploaded allocation.
        unsafe { ds4_bridge_varloop_free(self.ptr.as_ptr()) }
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: this handle exclusively owns the scratch context.
        unsafe { ds4_bridge_varloop_close(self.0.as_ptr()) }
    }
}
