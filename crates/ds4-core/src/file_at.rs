//! Positional reads for disk-backed tensor rows.

#[cfg(unix)]
pub(crate) use std::os::unix::fs::FileExt;

#[cfg(windows)]
pub(crate) trait FileExt {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
}

#[cfg(windows)]
impl FileExt for std::fs::File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        use std::os::windows::fs::FileExt as _;

        while !buf.is_empty() {
            let count = self.seek_read(buf, offset)?;
            if count == 0 {
                return Err(ErrorKind::UnexpectedEof.into());
            }
            offset = offset
                .checked_add(count as u64)
                .ok_or_else(|| Error::from(ErrorKind::InvalidInput))?;
            buf = &mut buf[count..];
        }
        Ok(())
    }
}

/// Stable identity for a disk-backed Engram shard, taken from its open handle.
/// ChangeTime also distinguishes in-place edits that preserve LastWriteTime.
#[cfg(windows)]
pub(crate) fn identity(file: &std::fs::File) -> std::io::Result<[u64; 5]> {
    use std::mem::{size_of_val, zeroed};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, GetFileInformationByHandle, GetFileInformationByHandleEx,
        BY_HANDLE_FILE_INFORMATION, FILE_BASIC_INFO,
    };
    let handle = file.as_raw_handle();
    // SAFETY: both calls fill their zeroed Win32 output structures synchronously.
    unsafe {
        let mut info: BY_HANDLE_FILE_INFORMATION = zeroed();
        let mut basic: FILE_BASIC_INFO = zeroed();
        if GetFileInformationByHandle(handle, &mut info) == 0
            || GetFileInformationByHandleEx(
                handle,
                FileBasicInfo,
                (&mut basic as *mut FILE_BASIC_INFO).cast(),
                size_of_val(&basic) as u32,
            ) == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok([
            info.dwVolumeSerialNumber.into(),
            (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
            (u64::from(info.nFileSizeHigh) << 32) | u64::from(info.nFileSizeLow),
            basic.LastWriteTime as u64,
            basic.ChangeTime as u64,
        ])
    }
}
